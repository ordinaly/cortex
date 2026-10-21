"""Reproducible bounded question-learning experiment; standard library only."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction as F
import json
from math import log
from pathlib import Path
from random import Random

from question_learning import Evidence, QUESTIONS, fit


class Episode:
    """Charged sensor access. Outcome is exposed only after the decision.

    Underscores express an API boundary, not a security sandbox. The learner
    receives only context, acquired answers, and the subsequently revealed outcome.
    """
    def __init__(self, episode_id, context, answers, outcome):
        self.episode_id, self.context = episode_id, context
        self._answers, self._outcome = tuple(answers), outcome
        self._asked = set()
        self._closed = False

    @property
    def query_count(self):
        return len(self._asked)

    def ask(self, question):
        if self._closed or question in self._asked:
            raise ValueError('episode closed or question already acquired')
        if question not in QUESTIONS:
            raise ValueError('inaccessible question')
        self._asked.add(question)
        return self._answers[QUESTIONS.index(question)]

    def reveal(self):
        if self._closed:
            raise ValueError('outcome already revealed')
        self._closed = True
        return self._outcome


def episodes(seed, phase, count, *, signal=True):
    """Only the world driver knows which probe matters; identities swap by seed."""
    offsets = {'discovery': 101, 'validation': 211, 'evaluation': 307}
    rng = Random(seed * 100003 + offsets[phase])
    relevant = seed % 2
    result = []
    for i in range(count):
        context = rng.randrange(2)
        answers = (rng.randrange(2), rng.randrange(2))
        p = .1 if context == 0 else (.9 if answers[relevant] else .1) if signal else .5
        result.append(Episode(f'{seed}:{phase}:{i}', context, answers, int(rng.random() < p)))
    return result


def collect(stream):
    records = []
    queries = 0
    for episode in stream:
        answers = tuple(episode.ask(q) for q in QUESTIONS)
        outcome = episode.reveal()
        records.append(Evidence(episode.episode_id, episode.context, answers, outcome))
        queries += episode.query_count
    return tuple(records), queries


@dataclass(frozen=True)
class Forecast:
    episode_id: str
    context: int
    question: str | None
    probability: F
    decision: int


def evaluate(model, stream, policy, cost, *, random_quota=0, seed=0):
    """Frozen prospective evaluation: no outcomes update the fitted model.

    Matched random receives only the total active query count, chooses episodes
    uniformly, and then chooses a probe uniformly. It never receives locations.
    """
    if policy not in ('active', 'random_matched', 'none', 'all'):
        raise ValueError('unknown policy')
    stream = tuple(stream)
    cost = F(cost)
    if not stream or cost < 0 or not 0 <= random_quota <= len(stream):
        raise ValueError('invalid evaluation configuration')
    ids = [episode.episode_id for episode in stream]
    if len(set(ids)) != len(ids) or set(ids) & model.evidence_ids:
        raise ValueError('evaluation evidence must be fresh and unique')
    rng = Random(seed)
    indices = set(rng.sample(range(len(stream)), random_quota)) if policy == 'random_matched' else set()
    chosen = {c: model.choose_question(c, cost) for c in model.contexts}
    probabilities = {(c, None, None): model.probability(c) for c in model.contexts}
    for c in model.contexts:
        for q in QUESTIONS:
            for answer in (0, 1):
                probabilities[c, q, answer] = model.probability(c, q, answer)
    forecasts, outcomes = [], []
    queries = 0
    for index, episode in enumerate(stream):
        question, answer = None, None
        if policy == 'active':
            question = chosen[episode.context]
        elif policy == 'random_matched' and index in indices:
            question = rng.choice(QUESTIONS)
        elif policy == 'all':
            answers = {q: episode.ask(q) for q in QUESTIONS}
            question = model.decision.adopted
            if question is not None:
                answer = answers[question]
        if question is not None and policy != 'all':
            answer = episode.ask(question)
        p = probabilities[episode.context, question, answer]
        # Freeze forecast before asking the world for the outcome.
        forecasts.append(Forecast(episode.episode_id, episode.context, question, p, int(p >= F(1, 2))))
        outcomes.append(episode.reveal())
        queries += episode.query_count
    errors = sum(f.decision != y for f, y in zip(forecasts, outcomes))
    brier = sum((float(f.probability)-y)**2 for f, y in zip(forecasts, outcomes)) / len(stream)
    log_loss = -sum(log(float(f.probability if y else 1-f.probability))
                    for f, y in zip(forecasts, outcomes)) / len(stream)
    return {'episodes': len(stream), 'queries': queries, 'errors': errors,
            'error_rate': errors/len(stream), 'brier': brier, 'log_loss': log_loss,
            'cost_per_episode': float(cost)*queries/len(stream),
            'total_loss': (errors + float(cost)*queries)/len(stream)}, tuple(forecasts)


def run_seed(seed, *, signal=True, discovery_n=512, validation_n=512, evaluation_n=512, cost=F(1, 10)):
    discovery, dq = collect(episodes(seed, 'discovery', discovery_n, signal=signal))
    validation, vq = collect(episodes(seed, 'validation', validation_n, signal=signal))
    model = fit(discovery, validation)
    results = {}
    results['active'], _ = evaluate(model, episodes(seed, 'evaluation', evaluation_n, signal=signal), 'active', cost)
    for policy in ('random_matched', 'none', 'all'):
        results[policy], _ = evaluate(model, episodes(seed, 'evaluation', evaluation_n, signal=signal), policy, cost,
                                      random_quota=results['active']['queries'], seed=seed+991)
    decision = asdict(model.decision)
    for key in ('discovery_gain', 'validation_gain', 'margin'):
        decision[key] = float(decision[key])
    return {'seed': seed, 'signal': signal, 'decision': decision,
            'acquisition_queries': {'discovery': dq, 'validation': vq},
            'choices': {str(c): model.choose_question(c, cost) for c in model.contexts},
            'policies': results}


def run_campaign(seeds):
    seeds = tuple(seeds)
    cases = [run_seed(seed, signal=signal) for signal in (True, False) for seed in seeds]
    summary = {}
    for signal, label in ((True, 'learnable'), (False, 'no_distinction')):
        rows = [r for r in cases if r['signal'] == signal]
        summary[label] = {
            'adopted': sum(r['decision']['adopted'] is not None for r in rows),
            'cases': len(rows),
            'policies': {policy: {metric: sum(r['policies'][policy][metric] for r in rows)/len(rows)
                                  for metric in ('queries', 'error_rate', 'brier', 'log_loss', 'total_loss')}
                         for policy in ('active', 'random_matched', 'none', 'all')}}
    return {'version': 'question-learning-v0.2', 'seeds': seeds,
            'design': {'discovery_n': 512, 'validation_n': 512, 'evaluation_n': 512,
                       'query_cost': .1, 'validation_margin': .02,
                       'training': 'fixed full-probe exploration, frozen before validation',
                       'evaluation': 'frozen model; matched random acquisition count'},
            'summary': summary, 'cases': cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run_campaign(range(20, 30))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
