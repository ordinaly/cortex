"""Exact, bounded evaluator. Never imported by the online learner.

Rational controlled Markov sources emit at the current state, then transition.
A probe of length h is an open-loop action word; outcomes are h-symbol words.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product, permutations


def rational(x):
    if isinstance(x, float):
        raise TypeError('Use rational strings or Fraction, not binary floats')
    return F(x)


def canonical(blocks):
    return tuple(sorted(tuple(sorted(b)) for b in blocks))


def partitions(n):
    """Every set partition exactly once (restricted-growth recursion)."""
    if n < 1 or n > 8:
        raise ValueError('Exhaustive partition cap: 1..8 states')
    def visit(i, blocks):
        if i == n:
            yield tuple(tuple(b) for b in blocks)
            return
        for b in blocks:
            b.append(i)
            yield from visit(i + 1, blocks)
            b.pop()
        blocks.append([i])
        yield from visit(i + 1, blocks)
        blocks.pop()
    yield from visit(0, [])


@dataclass(frozen=True)
class FiniteWorld:
    # kernel[action][state][next_state]; emission[state][symbol]
    kernel: tuple
    emission: tuple

    def __post_init__(self):
        e = tuple(tuple(rational(v) for v in row) for row in self.emission)
        k = tuple(tuple(tuple(rational(v) for v in row) for row in a) for a in self.kernel)
        n = len(e)
        if not 1 <= n <= 8 or not k or not e[0]:
            raise ValueError('Require 1..8 states, actions and symbols')
        if any(len(row) != len(e[0]) for row in e) or any(len(a) != n for a in k):
            raise ValueError('Inconsistent dimensions')
        if any(len(row) != n for a in k for row in a):
            raise ValueError('Non-square kernel')
        if any(any(v < 0 for v in row) or sum(row) != 1 for row in (*e, *(r for a in k for r in a))):
            raise ValueError('Rows must be probability distributions')
        object.__setattr__(self, 'kernel', k)
        object.__setattr__(self, 'emission', e)

    @property
    def n(self):
        return len(self.emission)

    def profiles(self, horizon, actions=None):
        """Exact distributions for every declared probe, all lengths 1..H."""
        actions = tuple(range(len(self.kernel))) if actions is None else tuple(actions)
        if not 1 <= horizon <= 5 or not actions or len(set(actions)) != len(actions):
            raise ValueError('Require unique actions and horizon 1..5')
        if any(a < 0 or a >= len(self.kernel) for a in actions):
            raise ValueError('Unknown action')
        m = len(self.emission[0])
        if sum((len(actions)*m)**h for h in range(1, horizon+1))*self.n > 200000:
            raise ValueError('Exact profile resource cap exceeded')
        profiles = [dict() for _ in range(self.n)]
        for h in range(1, horizon + 1):
            for probe in product(actions, repeat=h):
                outcomes = list(product(range(m), repeat=h))
                for initial in range(self.n):
                    probabilities = []
                    for word in outcomes:
                        mass = [F(int(i == initial)) for i in range(self.n)]
                        for a, symbol in zip(probe, word):
                            weighted = [mass[i]*self.emission[i][symbol] for i in range(self.n)]
                            mass = [sum(weighted[i]*self.kernel[a][i][j] for i in range(self.n)) for j in range(self.n)]
                        probabilities.append(sum(mass))
                    profiles[initial][probe] = tuple(probabilities)
        return tuple(profiles)


def tv(p, q):
    if len(p) != len(q):
        raise ValueError('Different outcome domains')
    return sum(abs(a-b) for a,b in zip(p,q))/2


def distances(profiles):
    if not profiles or any(p.keys() != profiles[0].keys() for p in profiles):
        raise ValueError('Profile domains must agree')
    return tuple(tuple(max((tv(p[q],r[q]) for q in p),default=F(0)) for r in profiles) for p in profiles)


def exact_quotient(profiles):
    cells = {}
    for i,p in enumerate(profiles):
        key = tuple(sorted(p.items()))
        cells.setdefault(key, []).append(i)
    return canonical(cells.values())


def validate_partition(partition, n):
    if any(not b for b in partition) or sorted(x for b in partition for x in b) != list(range(n)):
        raise ValueError('Not a partition of the declared carrier')


def distortion(partition, d):
    validate_partition(partition, len(d))
    return max((d[i][j] for b in partition for i in b for j in b), default=F(0))


def frontier(profiles):
    """Minimum achievable diameter for each exact cell count; all partitions."""
    d = distances(profiles)
    best = {}
    for p in partitions(len(profiles)):
        value = distortion(p, d)
        key = len(p)
        if key not in best or (value, canonical(p)) < best[key]:
            best[key] = value, canonical(p)
    return best


def stable_memory(world, initial=None):
    """Coarsest output-respecting strongly lumpable refinement of initial.

    This is NOT equated to finite-H trace equivalence.
    """
    p = canonical(initial if initial is not None else [range(world.n)])
    validate_partition(p, world.n)
    while True:
        groups = {}
        for b in p:
            for i in b:
                signature = (b, world.emission[i], tuple(sum(world.kernel[a][i][j] for j in c) for a in range(len(world.kernel)) for c in p))
                groups.setdefault(signature, []).append(i)
        refined = canonical(groups.values())
        if p == refined:
            return p
        p = refined


def lumpable(world, partition):
    validate_partition(partition, world.n)
    for b in partition:
        for i in b:
            for j in b:
                if world.emission[i] != world.emission[j]:
                    return False
                for a in world.kernel:
                    if any(sum(a[i][x] for x in c) != sum(a[j][x] for x in c) for c in partition):
                        return False
    return True


def semantic_role_distances(relations):
    """min over bijections pi(x)=y of max typed normalized distortion.

    Declared equal weighting of types, not a canonical scalarization claim.
    """
    r = tuple(tuple(tuple(rational(v) for v in row) for row in rel) for rel in relations)
    if not r or not 1 <= len(r[0]) <= 7:
        raise ValueError('Permutation cap: 1..7 objects')
    n = len(r[0])
    if any(len(t)!=n or any(len(row)!=n or any(v<0 or v>1 for v in row) for row in t) for t in r):
        raise ValueError('Require normalized square relations on one carrier')
    d = [[F(1) for _ in range(n)] for _ in range(n)]
    for pi in permutations(range(n)):
        error = max(abs(t[pi[i]][pi[j]]-t[i][j]) for t in r for i in range(n) for j in range(n))
        for i,j in enumerate(pi):
            d[i][j] = min(d[i][j], error)
    return tuple(map(tuple,d))


def audit_partition(profiles, partition, epsilon=F(0)):
    """Machine-readable certificate/counterexamples for any candidate partition.

    A pass is relative to supplied exact profiles only. Every failure contains
    a concrete pair and a declared distinguishing probe.
    """
    epsilon=rational(epsilon)
    if epsilon<0 or epsilon>1:
        raise ValueError('TV budget must lie in [0,1]')
    validate_partition(partition,len(profiles))
    d=distances(profiles)
    witnesses=[]
    for cell in partition:
        for index,i in enumerate(cell):
            for j in cell[index+1:]:
                if d[i][j]>epsilon:
                    probe=max(profiles[i],key=lambda q:tv(profiles[i][q],profiles[j][q]))
                    witnesses.append(dict(pair=(i,j),probe=probe,distance=str(d[i][j])))
    f=frontier(profiles)
    minimum=min(k for k,(error,_) in f.items() if error<=epsilon)
    return dict(passed=not witnesses,epsilon=str(epsilon),distortion=str(distortion(partition,d)),
                cells=len(partition),minimum_feasible_cells=minimum,
                excess_cells=len(partition)-minimum if not witnesses else None,witnesses=witnesses)
