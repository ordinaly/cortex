"""Synthetic prospective benchmark; no candidate motifs supplied to discovery."""
from dataclasses import asdict
import json
from compositional_language import Atom, Entity, Fact, Symbol
from pattern_discovery import Episode, discover, evaluate, select


def world(prefix, count, *, negative=False, reversed_roles=False):
    r, s, t, distractor = [Symbol(n, 1, ('entity', 'entity')) for n in ('R7', 'R2', 'R9', 'R4')]
    episodes = []
    for i in range(count):
        name = f'{prefix}:{i}'
        a, b, c, u, v = [Entity(f'{name}:{x}') for x in 'abcuv']
        inputs = [Atom(r, (a, b)), Atom(s, (c, b) if reversed_roles else (b, c)),
                  Atom(distractor, (u, v))]
        facts = tuple(Fact(f'{name}:{j}', name, x) for j, x in enumerate(inputs))
        episodes.append(Episode(name, facts, (Atom(t, (a, c), -1 if negative else 1),)))
    return episodes


def run():
    discovery = discover(world('discovery', 4))
    selected = select(discovery, world('validation', 4))
    fresh = world('test', 8)
    return {
        'examined': discovery.examined,
        'motifs_discovered': len(discovery.definitions),
        'rules_proposed': len(discovery.hypotheses),
        'rules_retained': len(selected.schemas),
        'complexity_cost': selected.complexity_cost,
        'validation': asdict(selected.score),
        'test': asdict(selected.test(fresh)),
        'no_rules_ablation': asdict(evaluate([], fresh)),
        'reversed_roles': asdict(selected.test(world('reversed', 8, reversed_roles=True))),
        'changed_world': asdict(selected.test(world('changed', 8, negative=True))),
        'counterexample_validation_retained': len(select(discovery, world('negative', 4, negative=True)).schemas),
        'high_cost_retained': len(select(discovery, world('cost', 4), cost_per_unit=1).schemas),
    }


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
