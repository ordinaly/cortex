"""Opt-in bounded predictive memory; no evaluator or hidden-world imports.

Opaque contexts and probes index observed binary outcomes. Each retained context
keeps recent sufficient evidence, even while its predictions share a folded cell.
Guarantees concern empirical cell diameters; confidence bounds additionally need
independent stationary Bernoulli samples. Drift tests are engineering heuristics.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass
from math import log, sqrt, isfinite


@dataclass(frozen=True)
class PredictiveMemoryPolicy:
    epsilon: float = 0.25
    max_contexts: int = 64
    window: int = 128
    rebuild_every: int = 64
    min_samples: int = 8
    confidence: bool = True
    delta: float = 0.05
    folding: bool = True
    complete_link: bool = True
    reopen: bool = True

    def __post_init__(self):
        if not isfinite(self.epsilon) or not 0 <= self.epsilon <= 1:
            raise ValueError('epsilon must lie in [0,1]')
        if any(type(x) is not int or x < 1 for x in (self.max_contexts,self.window,self.rebuild_every,self.min_samples)):
            raise ValueError('Budgets and sample counts must be positive integers')
        if not 0 < self.delta < 1:
            raise ValueError('delta must lie in (0,1)')


class PredictiveMemory:
    """Finite probe family; O(C P W) retained samples and O(C² P) fold search.

    This bounds storage in stream length, not in arbitrary symbol byte length.
    Non-transitive threshold chaining is offered only as an explicit ablation.
    """
    def __init__(self, probes=('next',), policy=None):
        self.probes = tuple(probes)
        if not self.probes or len(set(self.probes)) != len(self.probes):
            raise ValueError('Require a fixed nonempty unique probe domain')
        self.policy = policy or PredictiveMemoryPolicy()
        self.evidence = OrderedDict()
        self.cells = []
        self.successes = {}
        self.updates = self.rebuilds = self.evictions = self.reopenings = 0
        self.comparisons = 0

    def _stats(self, context, probe):
        row = self.evidence[context][probe]
        return self.successes[context][probe], len(row)

    def _radius(self, count):
        if count < self.policy.min_samples:
            return 1.0
        if not self.policy.confidence:
            return 0.0
        # Union over pairs, probes and rebuild index (summable alpha allocation).
        tests = max(1, self.policy.max_contexts**2 * len(self.probes))
        alpha = self.policy.delta / (2 * tests * (self.rebuilds + 1)**2)
        return min(1.0, sqrt(log(2/alpha)/(2*count)))

    def _compatible(self, a, b):
        self.comparisons += 1
        for probe in self.probes:
            sa,na = self._stats(a,probe)
            sb,nb = self._stats(b,probe)
            if min(na,nb) < self.policy.min_samples:
                return False
            if abs(sa/na-sb/nb)+self._radius(na)+self._radius(nb) > self.policy.epsilon:
                return False
        return True

    def rebuild(self):
        old = {frozenset(c) for c in self.cells}
        if not self.policy.folding:
            self.cells = [[x] for x in self.evidence]
            self.rebuilds += 1
            return
        if not self.policy.reopen and self.rebuilds:
            units = [list(c) for c in self.cells]
        else:
            units = [[x] for x in self.evidence]
        cells = []
        for unit in units:
            for cell in cells:
                matches = [self._compatible(context, other) for context in unit for other in cell]
                if all(matches) if self.policy.complete_link else any(matches):
                    cell.extend(unit)
                    break
            else:
                cells.append(unit)
        new_membership = {x:i for i,c in enumerate(cells) for x in c}
        self.reopenings += sum(len({new_membership[x] for x in c if x in new_membership})>1 for c in old)
        self.cells = cells
        self.rebuilds += 1

    def predict(self, context, probe='next'):
        if probe not in self.probes:
            raise ValueError('Probe outside declared validation domain')
        cell = next((c for c in self.cells if context in c), [context])
        s = n = 0
        for member in cell:
            if member in self.evidence:
                a,b = self._stats(member,probe)
                s += a
                n += b
        # Smoothed probability is a forecast, not a semantic truth certificate.
        return (s + 0.5)/(n + 1)

    def observe(self, context, outcome, probe='next'):
        if probe not in self.probes or type(outcome) is not int or outcome not in (0,1):
            raise ValueError('Require a declared probe and binary integer outcome')
        hash(context)
        if context not in self.evidence:
            if len(self.evidence) >= self.policy.max_contexts:
                victim,_ = self.evidence.popitem(last=False)
                self.cells = [[x for x in c if x != victim] for c in self.cells]
                self.cells = [c for c in self.cells if c]
                del self.successes[victim]
                self.evictions += 1
            self.evidence[context] = {p:deque(maxlen=self.policy.window) for p in self.probes}
            self.successes[context] = {p:0 for p in self.probes}
            self.cells.append([context])
        row = self.evidence[context][probe]
        if len(row) == self.policy.window:
            self.successes[context][probe] -= row[0]
        row.append(outcome)
        self.successes[context][probe] += outcome
        self.evidence.move_to_end(context)
        self.updates += 1
        if self.updates % self.policy.rebuild_every == 0:
            self.rebuild()

    @property
    def snapshot(self):
        return dict(contexts=len(self.evidence), cells=len(self.cells),
                    retained_samples=sum(len(row) for c in self.evidence.values() for row in c.values()),
                    updates=self.updates, rebuilds=self.rebuilds, evictions=self.evictions,
                    reopenings=self.reopenings, comparisons=self.comparisons)


class PredictiveEvidenceAdapter:
    """Cortex EvidenceAtom adapter for a declared binary relation domain.

    predict(context) must precede observation. Late events do not train the
    current predictive profile. Frontiers are evicted with their contexts.
    """
    def __init__(self, values, policy=None):
        self.values = tuple(values)
        if len(self.values)!=2 or len(set(self.values))!=2:
            raise ValueError('Exactly two distinct relation values required')
        self.memory = PredictiveMemory(policy=policy or PredictiveMemoryPolicy(folding=False))
        self.frontiers = {}
        self.late = 0

    def predict(self, context):
        return self.memory.predict(context)

    def observe(self, atom):
        if atom.relation not in self.values:
            raise ValueError('Relation outside declared binary channel')
        if atom.event_time < self.frontiers.get(atom.context,-1):
            self.late += 1
            return False
        self.memory.observe(atom.context, self.values.index(atom.relation))
        self.frontiers[atom.context] = atom.event_time
        self.frontiers = {k:v for k,v in self.frontiers.items() if k in self.memory.evidence}
        return True
