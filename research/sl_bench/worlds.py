"""Evaluator-owned adversarial sources. Learners receive context/probe/outcome only."""
from fractions import Fraction as F
from .oracle import FiniteWorld, rational


def deterministic(successors, outputs):
    n = len(outputs)
    return FiniteWorld(tuple(tuple(tuple(int(j==a[i]) for j in range(n)) for i in range(n)) for a in successors), tuple((1-rational(p), rational(p)) for p in outputs))


def exact_worlds():
    return {
        'delayed_witness': deterministic(((2,3,2,3),), (0,0,0,1)),
        'action_blindness': deterministic(((0,1,2,3),(2,3,2,3)), (0,0,0,1)),
        'nuisance_duplicates': deterministic(((0,1,2,3),), (F(1,10),F(1,10),F(9,10),F(9,10))),
        'nontransitive_chain': deterministic(((0,1,2),), (F(1,10),F(3,10),F(5,10))),
        'stochastic_aliasing': FiniteWorld((((F(1,2),F(1,2),0),(0,0,1),(0,0,1)),), ((1,0),(1,0),(0,1))),
    }


def stream(kind, rng, length=6000):
    for t in range(length):
        i = rng.randrange(6)
        context = ('observed', i)
        rates = [0.1,0.1,0.9,0.9,0.5,0.5]
        if kind == 'chain':
            rates = [0.1,0.3,0.5,0.5,0.7,0.9]
        if kind == 'drift_recurrence' and length//3 <= t < 2*length//3:
            rates[0], rates[2] = 0.9,0.1
        if kind == 'rare_witness':
            i = 0 if rng.random() < 0.96 else 1
            context = ('observed',i)
            rates[0],rates[1] = 0.1,0.9
        if kind == 'budget_pressure':
            i = rng.randrange(96)
            context = ('observed', i)
            probability = 0.1 if i%2 == 0 else 0.9
        else:
            probability = rates[i]
        yield context, int(rng.random()<probability), probability
