"""Prequential synthetic comparison: score each episode before consuming labels."""
from collections import deque
from functools import lru_cache
import json
import random
from compositional_language import Atom, Entity, Fact, Symbol
from pattern_discovery import Episode, discover, evaluate, select
from rule_revision import RevisingLanguage

R, S, T, U = [Symbol(n, 1, ('entity', 'entity')) for n in ('R', 'S', 'target', 'stable')]
HOT, COLD = [Symbol(n, 1, ('entity',)) for n in ('hot', 'cold')]


def sample(name, *, hot=True, positive=True, seed=0):
    rng = random.Random(str(seed)+name)
    a, b, c, u, v = [Entity(f'{name}:{rng.randrange(10**9)}:{i}') for i in range(5)]
    noise = Symbol('noise'+str(rng.randrange(3)), 1, ('entity', 'entity'))
    atoms = [Atom(R, (a,b)), Atom(S, (b,c)), Atom(HOT if hot else COLD, (a,)), Atom(noise,(u,v))]
    rng.shuffle(atoms)
    facts = tuple(Fact(name+':'+str(i), name, atom) for i,atom in enumerate(atoms))
    return Episode(name, facts, (Atom(T,(a,c),1 if positive else -1), Atom(U,(a,c))))


@lru_cache(maxsize=16)
def initial(seed=0):
    proposal = [sample('train'+str(i), hot=bool(i%2), seed=seed) for i in range(4)]
    validation = [sample('val'+str(i), hot=bool(i%2), seed=seed) for i in range(4)]
    return select(discover(proposal), validation)


def stream(kind, seed=0, count=48):
    for i in range(count):
        hot = bool(i%2)
        positive = True
        if i >= 8 and kind == 'scope':
            positive = hot
        elif i >= 8 and kind == 'reversal':
            positive = False
        elif kind == 'isolated_noise' and i == 8:
            positive = False
        yield sample(f'{kind}:{i}', hot=hot, positive=positive, seed=seed)


def run_one(kind, seed):
    base = initial(seed)
    adaptive = RevisingLanguage(base)
    tolerant = RevisingLanguage(base, counterexample_limit=2)
    periodic = base.schemas
    history = deque(maxlen=8)
    stats = {name: dict(correct=0, wrong=0, abstained=0, searches=0, examined=0)
             for name in ('frozen','periodic','targeted','two_strikes')}
    for i, episode in enumerate(stream(kind, seed)):
        for name, rules in [('frozen',base.schemas),('periodic',periodic)]:
            score = evaluate(rules, [episode])
            for key in ('correct','wrong','abstained'):
                stats[name][key] += getattr(score,key)
        result = adaptive.step(episode)
        for key in ('correct','wrong','abstained'):
            stats['targeted'][key] += getattr(result,key)
        tolerant_result = tolerant.step(episode)
        for key in ('correct','wrong','abstained'):
            stats['two_strikes'][key] += getattr(tolerant_result,key)
        history.append(episode)
        if (i+1)%8 == 0:
            block = list(history)
            d = discover(block[:4])
            periodic = select(d,block[4:]).schemas
            stats['periodic']['searches'] += 1
            stats['periodic']['examined'] += d.examined
    stats['targeted']['searches'] = adaptive.searches
    stats['targeted']['examined'] = adaptive.examined
    stats['two_strikes']['searches'] = tolerant.searches
    stats['two_strikes']['examined'] = tolerant.examined
    for row in stats.values():
        row['loss'] = 2*row['wrong']+row['abstained']
    return stats


def run(seeds=3):
    results = {}
    for kind in ('stationary','scope','reversal','isolated_noise'):
        runs = [run_one(kind,s) for s in range(seeds)]
        results[kind] = {method: {key: sum(r[method][key] for r in runs)/seeds
                         for key in runs[0][method]} for method in runs[0]}
    return {'seeds': seeds, 'episodes_per_run':48, 'queries_per_episode':2,
            'error_cost':2, 'abstention_cost':1, 'mean_results':results}


if __name__ == '__main__':
    print(json.dumps(run(),indent=2))
