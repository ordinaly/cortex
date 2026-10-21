"""Equal acquisition budgets, shared free context/outcomes, frozen evaluation."""
import argparse
from dataclasses import dataclass
from fractions import Fraction as F
import json
from pathlib import Path
from random import Random

from question_learning import QUESTIONS
from question_learning_campaign import episodes
from selective_training import Acquisition, Counts, PartialEvidence, validate


@dataclass(frozen=True)
class AcquisitionRecord:
    episode_id: str
    context: int
    questions: tuple[str, ...]
    reason: str


def train(stream, policy, budget, seed):
    stream = tuple(stream)
    if policy not in ('active', 'random', 'exhaustive') or budget < 0 or budget > len(stream):
        raise ValueError('invalid training configuration')
    counts, history, queries = Counts(), [], 0
    rng, acquisition = Random(seed), Acquisition(seed)
    # Exhaustive samples episodes uniformly, acquiring both properties, with an
    # optional single-query episode for an odd budget. All see every free outcome.
    chosen = rng.sample(range(len(stream)), (budget+1)//2) if policy == 'exhaustive' else []
    full = set(chosen[:budget//2])
    single = chosen[-1] if policy == 'exhaustive' and budget % 2 else None
    for index, episode in enumerate(stream):
        if policy == 'active' and queries < budget:
            q, reason = acquisition.choose(counts.snapshot(), episode.context)
            questions = (q,)
        elif policy == 'random' and queries < budget:
            questions, reason = (rng.choice(QUESTIONS),), 'random'
        elif policy == 'exhaustive' and index in full:
            questions, reason = QUESTIONS, 'exhaustive'
        elif policy == 'exhaustive' and index == single:
            questions, reason = (rng.choice(QUESTIONS),), 'remainder'
        else:
            questions, reason = (), 'no_acquisition'
        # Immutable choice precedes both the current sensor answers and outcome.
        history.append(AcquisitionRecord(episode.episode_id, episode.context, questions, reason))
        answers = [None, None]
        for q in questions:
            answers[QUESTIONS.index(q)] = episode.ask(q)
            queries += 1
        counts.update(PartialEvidence(episode.episode_id, episode.context, tuple(answers), episode.reveal()))
    return counts.snapshot(), tuple(history), queries


def run_case(seed, policy, budget, signal):
    snapshot, trace, discovery_cost = train(episodes(seed, 'discovery', budget, signal=signal), policy, budget, seed+313)
    proposal = snapshot.propose()  # Freeze before any validation observation.
    validation_rows = []
    for e in episodes(seed, 'validation', 64, signal=signal):
        answers = [None, None]
        answers[QUESTIONS.index(proposal)] = e.ask(proposal)
        validation_rows.append(PartialEvidence(e.episode_id, e.context, tuple(answers), e.reveal()))
    decision = validate(snapshot, proposal, validation_rows)
    # Identical evaluation acquisition: one proposed query per episode, even if
    # validation rejected it (then ignored). Isolates learning under equal ALL costs.
    errors, brier, forecasts = 0, 0., []
    cache = {(c, v): snapshot.predict(c, decision.adopted, v if decision.adopted else None)
             for c in (0, 1) for v in (0, 1)}
    for e in episodes(seed, 'evaluation', 512, signal=signal):
        if e.episode_id in snapshot.ids | decision.ids:
            raise ValueError('evaluation evidence reused')
        answer = e.ask(proposal)
        p = cache[e.context, answer]
        forecasts.append((e.episode_id, str(p)))
        y = e.reveal()
        errors += int(p >= F(1, 2)) != y
        brier += (float(p)-y)**2
    cost = discovery_cost + len(validation_rows) + 512
    return {'seed': seed, 'policy': policy, 'budget': budget, 'signal': signal,
            'proposed': proposal, 'adopted': decision.adopted, 'validation_gain': float(decision.gain),
            'discovery_queries': discovery_cost, 'validation_queries': len(validation_rows),
            'evaluation_queries': 512, 'total_queries': cost, 'error_rate': errors/512,
            'brier': brier/512, 'total_loss': (errors + .1*cost)/512,
            'exploration_queries': sum(r.reason == 'explore' for r in trace),
            'allocation': {str(c): {q: sum(r.context == c and q in r.questions for r in trace)
                                    for q in QUESTIONS} for c in (0, 1)}}


def campaign():
    cases = [run_case(seed, policy, budget, signal) for signal in (True, False)
             for budget in (64, 128, 256) for seed in range(40, 50)
             for policy in ('active', 'random', 'exhaustive')]
    summary = []
    for signal in (True, False):
        for budget in (64, 128, 256):
            for policy in ('active', 'random', 'exhaustive'):
                rows = [r for r in cases if (r['signal'], r['budget'], r['policy']) == (signal, budget, policy)]
                summary.append({'signal': signal, 'budget': budget, 'policy': policy,
                                'adopted': sum(r['adopted'] is not None for r in rows),
                                **{key: sum(r[key] for r in rows)/len(rows)
                                   for key in ('error_rate', 'brier', 'total_queries', 'total_loss')}})
    return {'version': 'selective-training-v0.3', 'seeds': list(range(40, 50)),
            'validation_n': 64, 'evaluation_n': 512, 'epsilon': .2, 'bonus': .1,
            'margin': .02, 'min_cell': 4, 'summary': summary, 'cases': cases}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = campaign()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2))
