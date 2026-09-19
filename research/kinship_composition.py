"""Cortex kinship relational-composition research prototype.

This module tests a narrow form of logical reasoning: learn a typed relation
composition law from solved short paths, then recursively apply that learned law
to unseen entities, unseen graphs, and longer paths.

The learner never receives the canonical kinship meanings used by the synthetic
world generator. Primitive relation tokens and answer labels are independently
permuted for every seed.

This is a structured-input benchmark. It isolates relational composition from
natural-language parsing and should not be interpreted as a CLUTRR replacement.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from random import Random
from typing import Iterable, Sequence


UP = "UP"
DOWN = "DOWN"

KINSHIP_STATES = (
    "parent",
    "ancestor",
    "child",
    "descendant",
    "sibling",
    "aunt_uncle",
    "niece_nephew",
    "cousin",
)


@dataclass(frozen=True)
class GraphExample:
    facts: tuple[tuple[str, str, str], ...]
    query_source: str
    query_target: str
    answer: str
    path_length: int


@dataclass(frozen=True)
class Inference:
    answer: str | None
    confidence: float
    path: tuple[str, ...]
    resolved: bool


def canonical_relation(sequence: Sequence[str]) -> str:
    """Ground-truth relation for the synthetic one-parent family path."""
    if not sequence:
        raise ValueError("query path must be non-empty")

    up = 0
    while up < len(sequence) and sequence[up] == UP:
        up += 1
    if any(token == UP for token in sequence[up:]):
        raise ValueError("sequence must be a canonical UP* DOWN* geodesic")
    down = len(sequence) - up

    if up == 0 and down == 1:
        return "parent"
    if up == 0 and down >= 2:
        return "ancestor"
    if up == 1 and down == 0:
        return "child"
    if up >= 2 and down == 0:
        return "descendant"
    if up == 1 and down == 1:
        return "sibling"
    if up == 1 and down >= 2:
        return "aunt_uncle"
    if up >= 2 and down == 1:
        return "niece_nephew"
    if up >= 2 and down >= 2:
        return "cousin"
    raise AssertionError((up, down))


def canonical_sequences(length: int) -> list[tuple[str, ...]]:
    if length < 1:
        raise ValueError("length must be positive")
    return [
        (UP,) * up + (DOWN,) * (length - up)
        for up in range(length + 1)
    ]


def random_aliases(
    rng: Random,
) -> tuple[dict[str, str], dict[str, str]]:
    primitive_names = ["R0", "R1"]
    rng.shuffle(primitive_names)
    primitive_alias = {
        UP: primitive_names[0],
        DOWN: primitive_names[1],
    }

    answer_names = [f"K{i}" for i in range(len(KINSHIP_STATES))]
    rng.shuffle(answer_names)
    answer_alias = dict(zip(KINSHIP_STATES, answer_names))
    return primitive_alias, answer_alias


def make_example(
    sequence: Sequence[str],
    *,
    primitive_alias: dict[str, str],
    answer_alias: dict[str, str],
    rng: Random,
    namespace: str,
    distractor_branches: int = 5,
    disconnected_edges: int = 7,
) -> GraphExample:
    """Create a unique directed query graph with irrelevant facts."""
    if distractor_branches < 0 or disconnected_edges < 0:
        raise ValueError("distractor counts must be non-negative")

    nodes = [
        f"{namespace}_p{i}_{rng.getrandbits(48):012x}"
        for i in range(len(sequence) + 1)
    ]
    facts: list[tuple[str, str, str]] = []
    for index, token in enumerate(sequence):
        facts.append(
            (
                nodes[index],
                primitive_alias[token],
                nodes[index + 1],
            )
        )

    relation_values = list(primitive_alias.values())
    for index in range(distractor_branches):
        source = rng.choice(nodes)
        target = f"{namespace}_branch{index}_{rng.getrandbits(48):012x}"
        facts.append((source, rng.choice(relation_values), target))

    for index in range(disconnected_edges):
        source = f"{namespace}_noiseA{index}_{rng.getrandbits(48):012x}"
        target = f"{namespace}_noiseB{index}_{rng.getrandbits(48):012x}"
        facts.append((source, rng.choice(relation_values), target))

    rng.shuffle(facts)
    truth = canonical_relation(sequence)
    return GraphExample(
        facts=tuple(facts),
        query_source=nodes[0],
        query_target=nodes[-1],
        answer=answer_alias[truth],
        path_length=len(sequence),
    )


def extract_relation_path(example: GraphExample) -> tuple[str, ...]:
    """Return the unique directed relation path between query entities."""
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for source, relation, target in example.facts:
        adjacency[source].append((target, relation))

    queue = deque([(example.query_source, tuple())])
    seen = {example.query_source}
    while queue:
        node, path = queue.popleft()
        if node == example.query_target:
            return path
        for target, relation in adjacency.get(node, ()):
            if target in seen:
                continue
            seen.add(target)
            queue.append((target, path + (relation,)))
    raise ValueError("query target is unreachable")


class CortexRelationAlgebra:
    """Evidence-based finite relation algebra induced from solved examples."""

    START = "__START__"

    def __init__(
        self,
        *,
        minimum_transition_evidence: int = 4,
        confidence_threshold: float = 0.90,
        margin_threshold: float = 0.60,
    ) -> None:
        if minimum_transition_evidence < 1:
            raise ValueError("minimum_transition_evidence must be positive")
        self.minimum_transition_evidence = minimum_transition_evidence
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.path_answers: dict[tuple[str, ...], Counter[str]] = defaultdict(
            Counter
        )
        self.transitions: dict[
            tuple[str, str], Counter[str]
        ] = defaultdict(Counter)
        self._finalized = False

    def observe_solved(self, example: GraphExample) -> None:
        if self._finalized:
            raise RuntimeError("cannot add examples after finalize")
        path = extract_relation_path(example)
        self.path_answers[path][example.answer] += 1

    @staticmethod
    def _dominant(counter: Counter[str]) -> tuple[str, int, float, float]:
        total = sum(counter.values())
        if total <= 0:
            raise ValueError("empty evidence counter")
        ordered = counter.most_common()
        best_label, best_count = ordered[0]
        second_count = ordered[1][1] if len(ordered) > 1 else 0
        confidence = best_count / total
        margin = (best_count - second_count) / total
        return best_label, total, confidence, margin

    def finalize(self) -> None:
        if self._finalized:
            return

        solved: dict[tuple[str, ...], tuple[str, int]] = {}
        for path, counter in self.path_answers.items():
            label, total, confidence, margin = self._dominant(counter)
            if confidence < self.confidence_threshold:
                continue
            if margin < self.margin_threshold:
                continue
            solved[path] = (label, total)

        for path, (label, support) in solved.items():
            if len(path) == 1:
                self.transitions[(self.START, path[0])][label] += support

        for path, (state, _support) in solved.items():
            for extension, (next_state, next_support) in solved.items():
                if len(extension) != len(path) + 1:
                    continue
                if extension[:-1] != path:
                    continue
                token = extension[-1]
                self.transitions[(state, token)][next_state] += next_support

        self._finalized = True

    def transition_evidence(
        self,
    ) -> dict[tuple[str, str], dict[str, int]]:
        self.finalize()
        return {
            key: dict(counter)
            for key, counter in self.transitions.items()
        }

    def _step(self, state: str, token: str) -> tuple[str | None, float]:
        counter = self.transitions.get((state, token))
        if not counter:
            return None, 0.0
        best, total, confidence, margin = self._dominant(counter)
        if total < self.minimum_transition_evidence:
            return None, confidence
        if confidence < self.confidence_threshold:
            return None, confidence
        if margin < self.margin_threshold:
            return None, confidence
        return best, confidence

    def infer(self, example: GraphExample) -> Inference:
        self.finalize()
        path = extract_relation_path(example)
        state = self.START
        confidence = 1.0
        for token in path:
            next_state, step_confidence = self._step(state, token)
            confidence = min(confidence, step_confidence)
            if next_state is None:
                return Inference(None, confidence, path, False)
            state = next_state
        return Inference(state, confidence, path, True)


class ExactSequenceMemory:
    """Non-compositional exact lookup baseline."""

    def __init__(self) -> None:
        self.answers: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

    def observe_solved(self, example: GraphExample) -> None:
        self.answers[extract_relation_path(example)][example.answer] += 1

    def infer(self, example: GraphExample) -> Inference:
        path = extract_relation_path(example)
        counter = self.answers.get(path)
        if not counter:
            return Inference(None, 0.0, path, False)
        label, count = counter.most_common(1)[0]
        confidence = count / sum(counter.values())
        return Inference(label, confidence, path, True)


class LastRelationBaseline:
    """Weak baseline conditioning only on the final primitive relation."""

    def __init__(self) -> None:
        self.answers: dict[str, Counter[str]] = defaultdict(Counter)

    def observe_solved(self, example: GraphExample) -> None:
        path = extract_relation_path(example)
        self.answers[path[-1]][example.answer] += 1

    def infer(self, example: GraphExample) -> Inference:
        path = extract_relation_path(example)
        counter = self.answers.get(path[-1])
        if not counter:
            return Inference(None, 0.0, path, False)
        label, count = counter.most_common(1)[0]
        confidence = count / sum(counter.values())
        return Inference(label, confidence, path, True)


class ClippedCountBaseline:
    """Hand-engineered sufficient-statistic baseline."""

    def __init__(self) -> None:
        self.raw: list[tuple[tuple[str, ...], str]] = []
        self.tokens: tuple[str, str] | None = None
        self.answers: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
        self._finalized = False

    def observe_solved(self, example: GraphExample) -> None:
        if self._finalized:
            raise RuntimeError("cannot add examples after finalize")
        self.raw.append((extract_relation_path(example), example.answer))

    def finalize(self) -> None:
        if self._finalized:
            return
        token_set = sorted(
            {token for path, _answer in self.raw for token in path}
        )
        if len(token_set) != 2:
            raise ValueError("expected exactly two primitive relation aliases")
        self.tokens = (token_set[0], token_set[1])
        for path, answer in self.raw:
            feature = tuple(
                min(2, path.count(token))
                for token in self.tokens
            )
            self.answers[feature][answer] += 1
        self._finalized = True

    def infer(self, example: GraphExample) -> Inference:
        self.finalize()
        assert self.tokens is not None
        path = extract_relation_path(example)
        feature = tuple(
            min(2, path.count(token))
            for token in self.tokens
        )
        counter = self.answers.get(feature)
        if not counter:
            return Inference(None, 0.0, path, False)
        label, count = counter.most_common(1)[0]
        confidence = count / sum(counter.values())
        return Inference(label, confidence, path, True)


def train_examples(
    *,
    seed: int,
    max_length: int = 5,
    repetitions: int = 8,
    omitted_sequence: tuple[str, ...] | None = None,
) -> tuple[list[GraphExample], dict[str, str], dict[str, str]]:
    if max_length < 1 or repetitions < 1:
        raise ValueError("invalid training configuration")
    rng = Random(seed)
    primitive_alias, answer_alias = random_aliases(rng)
    examples: list[GraphExample] = []
    index = 0
    for length in range(1, max_length + 1):
        for sequence in canonical_sequences(length):
            if omitted_sequence is not None and sequence == omitted_sequence:
                continue
            for _ in range(repetitions):
                examples.append(
                    make_example(
                        sequence,
                        primitive_alias=primitive_alias,
                        answer_alias=answer_alias,
                        rng=rng,
                        namespace=f"train{seed}_{index}",
                    )
                )
                index += 1
    rng.shuffle(examples)
    return examples, primitive_alias, answer_alias


def test_examples(
    *,
    seed: int,
    primitive_alias: dict[str, str],
    answer_alias: dict[str, str],
    min_length: int = 6,
    max_length: int = 10,
    repetitions: int = 4,
    require_prefix: tuple[str, ...] | None = None,
) -> list[GraphExample]:
    if min_length < 1 or max_length < min_length or repetitions < 1:
        raise ValueError("invalid test configuration")
    rng = Random(seed + 1_000_003)
    examples: list[GraphExample] = []
    index = 0
    for length in range(min_length, max_length + 1):
        for sequence in canonical_sequences(length):
            if require_prefix is not None:
                if sequence[: len(require_prefix)] != require_prefix:
                    continue
            for _ in range(repetitions):
                examples.append(
                    make_example(
                        sequence,
                        primitive_alias=primitive_alias,
                        answer_alias=answer_alias,
                        rng=rng,
                        namespace=f"test{seed}_{index}",
                    )
                )
                index += 1
    rng.shuffle(examples)
    return examples


def score(
    model: object,
    examples: Iterable[GraphExample],
) -> dict[str, float | dict[int, float]]:
    total = 0
    correct = 0
    resolved = 0
    by_length: dict[int, list[int]] = defaultdict(list)

    infer = getattr(model, "infer")
    for example in examples:
        prediction: Inference = infer(example)
        total += 1
        if prediction.resolved:
            resolved += 1
        hit = int(prediction.answer == example.answer)
        correct += hit
        by_length[example.path_length].append(hit)

    return {
        "accuracy": correct / max(1, total),
        "coverage": resolved / max(1, total),
        "resolved_accuracy": correct / max(1, resolved),
        "accuracy_by_length": {
            length: sum(values) / len(values)
            for length, values in sorted(by_length.items())
        },
    }
