"""Cross-algebra transfer of discovered Cortex equation forms.

The transfer layer deliberately reuses only canonical equation forms and
source-world evidence summaries. Source element labels, source table cells, and
source applicability scopes are never transferred.

Target worlds must re-ground every imported form from target evidence before it
can participate in closure.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Iterable, Sequence

from algebra_form_induction import (
    CandidateForm,
    CortexFuzzyLawInducer,
    Evidence,
    FormInference,
    LawScope,
    Table,
    _close_forms,
    _crossfit_predictive_evidence,
    _fold_tables,
    _structural_evidence,
)


@dataclass(frozen=True)
class TransferPrior:
    """Reusable equation form distilled from several source worlds."""

    form: CandidateForm
    source_worlds: int
    source_selected_worlds: int
    source_support_rate: float
    mean_structural_membership: float
    mean_predictive_membership: float

    @property
    def strength(self) -> float:
        # Transfer carries confidence in the *form*, not source-world
        # applicability. Predictive applicability must be re-earned on target.
        return min(
            self.source_support_rate,
            self.mean_structural_membership,
        )


@dataclass(frozen=True)
class TransferLaw:
    prior: TransferPrior
    discovery: Evidence
    validation: Evidence
    scope: LawScope
    target_structural_membership: float
    target_predictive_membership: float
    structural_membership: float
    membership: float
    score: float
    active: bool
    predictive_active: bool


@dataclass(frozen=True)
class TransferInference:
    value: str | None
    resolved: bool
    provenance: tuple[str, ...]


def build_transfer_library(
    source_models: Sequence[CortexFuzzyLawInducer],
    *,
    minimum_source_support: float = 0.75,
    minimum_source_membership: float = 0.98,
    maximum_forms: int = 24,
) -> tuple[TransferPrior, ...]:
    """Distill reusable canonical forms from independently fitted source worlds."""
    if not source_models:
        raise ValueError("source_models must not be empty")
    if not 0.0 <= minimum_source_support <= 1.0:
        raise ValueError("minimum_source_support must lie in [0,1]")
    if not 0.0 <= minimum_source_membership <= 1.0:
        raise ValueError("minimum_source_membership must lie in [0,1]")
    if maximum_forms < 1:
        raise ValueError("maximum_forms must be positive")

    per_key: dict[str, list] = {}
    forms: dict[str, CandidateForm] = {}
    predictive_witnesses: dict[str, int] = {}

    for model in source_models:
        supported = {
            law.form.key: law
            for law in model.laws
            if law.active
        }
        for key, law in supported.items():
            forms[key] = law.form
            per_key.setdefault(key, []).append(law)
            predictive_witnesses[key] = (
                predictive_witnesses.get(key, 0)
                + law.validation.positive
            )

    priors: list[TransferPrior] = []
    total_worlds = len(source_models)
    for key, laws in per_key.items():
        selected_worlds = len(laws)
        support_rate = selected_worlds / total_worlds
        if support_rate < minimum_source_support:
            continue

        structural = sum(
            law.structural_membership
            for law in laws
        ) / selected_worlds
        predictive = sum(
            law.predictive_membership
            for law in laws
        ) / selected_worlds

        if structural < minimum_source_membership:
            continue

        priors.append(
            TransferPrior(
                form=forms[key],
                source_worlds=total_worlds,
                source_selected_worlds=selected_worlds,
                source_support_rate=support_rate,
                mean_structural_membership=structural,
                mean_predictive_membership=predictive,
            )
        )

    priors.sort(
        key=lambda prior: (
            -prior.strength,
            prior.form.complexity,
            -predictive_witnesses.get(prior.form.key, 0),
            prior.form.key,
        )
    )
    return tuple(priors[:maximum_forms])


def _target_minimum_witnesses(variable_count: int) -> int:
    if variable_count <= 1:
        return 2
    if variable_count == 2:
        return 2
    return 4


def _evidence_membership(
    evidence: Evidence,
    *,
    minimum_witnesses: int,
) -> float:
    if evidence.total == 0:
        return 0.0
    mass = min(1.0, evidence.total / max(1, minimum_witnesses))
    return evidence.consistency * mass


def _universal_target_scope(
    form: CandidateForm,
    elements: Sequence[str],
) -> LawScope:
    """Universal target-scope hypothesis for a transferred equation form.

    No source entity scope is reused. Instead, cross-world structural support
    permits the *hypothesis* that the canonical equation is universally
    quantified over the new target's elements. Target-only structural and
    predictive evidence must still accept that hypothesis before closure.
    """
    all_elements = frozenset(elements)
    unary = tuple(
        (name, all_elements)
        for name in form.variables
    )
    pairwise = tuple()
    if len(form.variables) >= 3:
        all_pairs = frozenset(product(elements, repeat=2))
        pairwise = tuple(
            (
                left_name,
                right_name,
                all_pairs,
            )
            for index, left_name in enumerate(form.variables)
            for right_name in form.variables[index + 1 :]
        )
    return LawScope(
        unary=unary,
        pairwise=pairwise,
    )


def _target_crossfit_lawset_evidence(
    forms: Sequence[CandidateForm],
    elements: Sequence[str],
    observed: Table,
) -> tuple[Evidence, int]:
    """Leave-one-cell-out target validation for sparse transfer.

    Every observed target cell is predicted from all other observed cells.
    This remains strictly out-of-sample while avoiding the evidence collapse
    caused by removing a quarter of an already sparse target table.
    """
    if len(observed) < 2:
        return Evidence(0, 0, 2), 0

    positive = 0
    negative = 0
    conflicts = 0
    folds = len(observed)

    for train, validation in _fold_tables(
        elements,
        observed,
        folds=folds,
    ):
        scoped_forms = [
            (
                form,
                _universal_target_scope(form, elements),
            )
            for form in forms
        ]
        completed, _provenance, fold_conflicts, _rounds = _close_forms(
            elements,
            train,
            scoped_forms,
        )
        conflicts += fold_conflicts
        for pair, truth in validation.items():
            prediction = completed.get(pair)
            if prediction is None:
                continue
            if prediction == truth:
                positive += 1
            else:
                negative += 1

    return Evidence(
        positive=positive,
        negative=negative,
        minimum_witnesses=2,
    ), conflicts


class CortexTransferredLawInducer:
    """Re-ground transferred forms in a new target algebra.

    This class never enumerates a target grammar. It scores only the forms in
    the supplied transfer library.
    """

    def __init__(
        self,
        elements: Iterable[str],
        observed: Table,
        library: Sequence[TransferPrior],
        *,
        membership_threshold: float = 0.98,
        complexity_penalty: float = 0.01,
        maximum_active_laws: int = 24,
    ) -> None:
        self.elements = tuple(sorted(elements))
        self.observed = dict(observed)
        self.library = tuple(library)

        if not self.elements:
            raise ValueError("elements must not be empty")
        if not 0.0 <= membership_threshold <= 1.0:
            raise ValueError("membership_threshold must lie in [0,1]")
        if complexity_penalty < 0.0 or not isfinite(complexity_penalty):
            raise ValueError("complexity_penalty must be finite and non-negative")
        if maximum_active_laws < 1:
            raise ValueError("maximum_active_laws must be positive")

        self.membership_threshold = membership_threshold
        self.complexity_penalty = complexity_penalty
        self.maximum_active_laws = maximum_active_laws

        self.laws = self._score_library()
        eligible = [
            law
            for law in self.laws
            if law.active
        ]
        eligible.sort(
            key=lambda law: (
                -law.score,
                law.prior.form.complexity,
                law.prior.form.key,
            )
        )

        candidates = list(
            eligible[:maximum_active_laws]
        )
        self.joint_validation = Evidence(
            positive=0,
            negative=0,
            minimum_witnesses=2,
        )
        self.joint_validation_conflicts = 0
        self.selection_mode = "unresolved"

        def validate_bundle(
            laws: Sequence[TransferLaw],
        ) -> tuple[Evidence, int]:
            return _target_crossfit_lawset_evidence(
                [law.prior.form for law in laws],
                self.elements,
                self.observed,
            )

        def is_safe(
            evidence: Evidence,
            conflicts: int,
        ) -> bool:
            return (
                conflicts == 0
                and evidence.negative == 0
                and evidence.membership
                >= membership_threshold
            )

        accepted: list[TransferLaw] = []

        # First look for a target-validated single transferred law. This is the
        # most conservative reusable abstraction.
        single_seeds = []
        for law in candidates:
            evidence, conflicts = validate_bundle([law])
            if is_safe(evidence, conflicts):
                single_seeds.append(
                    (
                        -evidence.positive,
                        law.prior.form.complexity,
                        law.prior.form.key,
                        law,
                        evidence,
                        conflicts,
                    )
                )

        if single_seeds:
            single_seeds.sort(
                key=lambda item: item[:3]
            )
            (
                _neg_positive,
                _complexity,
                _key,
                seed,
                evidence,
                conflicts,
            ) = single_seeds[0]
            accepted = [seed]
            self.joint_validation = evidence
            self.joint_validation_conflicts = conflicts
            self.selection_mode = "single-seed"
        else:
            # Some laws can become predictive only compositionally. Search
            # pairs before giving up, bounded by the 24-form transfer library.
            pair_seeds = []
            for left_index in range(len(candidates)):
                for right_index in range(
                    left_index + 1,
                    len(candidates),
                ):
                    pair = [
                        candidates[left_index],
                        candidates[right_index],
                    ]
                    evidence, conflicts = validate_bundle(pair)
                    if not is_safe(evidence, conflicts):
                        continue
                    pair_seeds.append(
                        (
                            -evidence.positive,
                            sum(
                                law.prior.form.complexity
                                for law in pair
                            ),
                            tuple(
                                law.prior.form.key
                                for law in pair
                            ),
                            pair,
                            evidence,
                            conflicts,
                        )
                    )

            if pair_seeds:
                pair_seeds.sort(
                    key=lambda item: item[:3]
                )
                (
                    _neg_positive,
                    _complexity,
                    _keys,
                    accepted,
                    evidence,
                    conflicts,
                ) = pair_seeds[0]
                self.joint_validation = evidence
                self.joint_validation_conflicts = conflicts
                self.selection_mode = "pair-seed"

        # Grow the accepted bundle only when another law gives strictly more
        # correct cross-fitted target predictions without errors or conflicts.
        while accepted and len(accepted) < maximum_active_laws:
            accepted_keys = {
                law.prior.form.key
                for law in accepted
            }
            current_positive = self.joint_validation.positive
            improvements = []

            for law in candidates:
                if law.prior.form.key in accepted_keys:
                    continue
                trial = accepted + [law]
                evidence, conflicts = validate_bundle(trial)
                if not is_safe(evidence, conflicts):
                    continue
                if evidence.positive <= current_positive:
                    continue
                improvements.append(
                    (
                        -evidence.positive,
                        law.prior.form.complexity,
                        law.prior.form.key,
                        law,
                        evidence,
                        conflicts,
                    )
                )

            if not improvements:
                break

            improvements.sort(
                key=lambda item: item[:3]
            )
            (
                _neg_positive,
                _complexity,
                _key,
                law,
                evidence,
                conflicts,
            ) = improvements[0]
            accepted.append(law)
            self.joint_validation = evidence
            self.joint_validation_conflicts = conflicts

        self.active_laws = tuple(accepted)
        self.completed = dict(self.observed)
        self.provenance: dict[tuple[str, str], tuple[str, ...]] = {}
        self.conflicts = 0
        self.rounds = 0
        self._closed = False

    def _score_library(self) -> tuple[TransferLaw, ...]:
        out: list[TransferLaw] = []
        for prior in self.library:
            form = prior.form
            discovery = _structural_evidence(
                form,
                self.elements,
                self.observed,
            )
            validation = _crossfit_predictive_evidence(
                form,
                self.elements,
                self.observed,
            )
            scope = _universal_target_scope(
                form,
                self.elements,
            )

            target_structural = _evidence_membership(
                discovery,
                minimum_witnesses=_target_minimum_witnesses(
                    len(form.variables)
                ),
            )
            target_predictive = _evidence_membership(
                validation,
                minimum_witnesses=1,
            )

            structural = min(
                prior.strength,
                target_structural,
            )
            membership = min(
                structural,
                target_predictive,
            )
            score = (
                structural
                - self.complexity_penalty * form.complexity
            )

            out.append(
                TransferLaw(
                    prior=prior,
                    discovery=discovery,
                    validation=validation,
                    scope=scope,
                    target_structural_membership=target_structural,
                    target_predictive_membership=target_predictive,
                    structural_membership=structural,
                    membership=membership,
                    score=score,
                    active=structural >= self.membership_threshold,
                    predictive_active=membership >= self.membership_threshold,
                )
            )
        return tuple(out)

    def structurally_supported_keys(self) -> set[str]:
        return {
            law.prior.form.key
            for law in self.laws
            if law.active
        }

    def selected_keys(self) -> set[str]:
        return {
            law.prior.form.key
            for law in self.active_laws
        }

    def close(self) -> None:
        if self._closed:
            return

        (
            self.completed,
            self.provenance,
            self.conflicts,
            self.rounds,
        ) = _close_forms(
            self.elements,
            self.observed,
            [
                (law.prior.form, law.scope)
                for law in self.active_laws
            ],
        )
        self._closed = True

    def infer(self, left: str, right: str) -> TransferInference:
        self.close()
        pair = (left, right)
        value = self.completed.get(pair)
        if value is None:
            return TransferInference(None, False, tuple())
        if pair in self.observed:
            return TransferInference(value, True, ("observed",))
        return TransferInference(
            value,
            True,
            self.provenance.get(pair, ("derived",)),
        )


def score_transfer_heldout(
    model: object,
    table: Table,
    heldout: Iterable[tuple[str, str]],
) -> dict[str, float | int]:
    total = 0
    resolved = 0
    correct = 0

    infer = getattr(model, "infer")
    for pair in heldout:
        total += 1
        result = infer(*pair)
        if result.resolved:
            resolved += 1
            correct += int(result.value == table[pair])

    return {
        "heldout": total,
        "resolved": resolved,
        "correct": correct,
        "wrong_resolved": resolved - correct,
        "coverage": resolved / max(1, total),
        "accuracy": correct / max(1, total),
        "resolved_accuracy": correct / max(1, resolved),
    }
