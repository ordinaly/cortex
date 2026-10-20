import random
import unittest
from fractions import Fraction as F
from itertools import product
from research.sl_bench.oracle import (FiniteWorld, partitions, exact_quotient, distances,
    distortion, frontier, stable_memory, lumpable, semantic_role_distances, audit_partition)
from research.sl_bench.worlds import exact_worlds, deterministic
from cortex.predictive_memory import PredictiveMemory, PredictiveMemoryPolicy, PredictiveEvidenceAdapter
from cortex.closed_articulation_loop import EvidenceAtom


class OracleTests(unittest.TestCase):
    def test_bell_counts_and_unique_partitions(self):
        for n,b in enumerate([1,2,5,15,52,203],1):
            rows = list(partitions(n))
            self.assertEqual(len(set(rows)),b)

    def test_delayed_and_action_witnesses(self):
        worlds = exact_worlds()
        w = worlds['delayed_witness']
        self.assertEqual(exact_quotient(w.profiles(1)),((0,1,2),(3,)))
        self.assertEqual(exact_quotient(w.profiles(2)),((0,2),(1,),(3,)))
        a = worlds['action_blindness']
        self.assertEqual(exact_quotient(a.profiles(2, actions=(0,))),((0,1,2),(3,)))
        self.assertEqual(exact_quotient(a.profiles(2)),((0,2),(1,),(3,)))

    def test_nontransitive_and_optimal_frontier(self):
        profiles = exact_worlds()['nontransitive_chain'].profiles(1)
        d = distances(profiles)
        self.assertEqual(d[0][1],F(1,5))
        self.assertEqual(d[1][2],F(1,5))
        self.assertEqual(d[0][2],F(2,5))
        f = frontier(profiles)
        self.assertEqual(f[1][0],F(2,5))
        self.assertEqual(f[2][0],F(1,5))
        self.assertEqual(f[3][0],0)

    def test_independent_trace_path_enumeration(self):
        rng = random.Random(81)
        for _ in range(10):
            n=3
            kernel=[]
            for a in range(2):
                rows=[]
                for i in range(n):
                    raw=[rng.randrange(1,5) for j in range(n)]
                    rows.append(tuple(F(v,sum(raw)) for v in raw))
                kernel.append(tuple(rows))
            emission=tuple((F(i+1,5),1-F(i+1,5)) for i in range(n))
            w=FiniteWorld(tuple(kernel),emission)
            profiles=w.profiles(3)
            for initial in range(n):
                for probe, probs in profiles[initial].items():
                    h=len(probe)
                    self.assertEqual(sum(probs),1)
                    for word,actual in zip(product(range(2),repeat=h),probs):
                        expected=F(0)
                        for path in product(range(n),repeat=h):
                            s=initial
                            weight=F(1)
                            for a,y,nxt in zip(probe,word,path):
                                weight*=emission[s][y]*kernel[a][s][nxt]
                                s=nxt
                            expected+=weight
                        self.assertEqual(actual,expected)

    def test_stable_memory_exhaustive_minimality(self):
        for w in exact_worlds().values():
            actual=stable_memory(w)
            self.assertTrue(lumpable(w,actual))
            feasible=[p for p in partitions(w.n) if lumpable(w,p)]
            self.assertEqual(len(actual),min(map(len,feasible)))
            fine=stable_memory(w, tuple((i,) for i in range(w.n)))
            self.assertEqual(len(fine),w.n)

    def test_relabeling_and_horizon_refinement(self):
        w=exact_worlds()['delayed_witness']
        pi=(3,1,0,2)
        v=FiniteWorld(tuple(tuple(tuple(a[pi[i]][pi[j]] for j in range(w.n)) for i in range(w.n)) for a in w.kernel),tuple(w.emission[i] for i in pi))
        p=w.profiles(3);q=v.profiles(3)
        self.assertEqual(q,tuple(p[i] for i in pi))
        previous=set()
        for h in range(1,4):
            partition=exact_quotient(w.profiles(h))
            separated={(i,j) for i in range(w.n) for j in range(w.n) if not any(i in b and j in b for b in partition)}
            self.assertTrue(previous <= separated)
            previous=separated

    def test_roles_pseudometric_and_symmetry(self):
        r=(((0,1,0),(1,0,1),(0,1,0)),)
        d=semantic_role_distances(r)
        self.assertEqual(d[0][2],0)
        self.assertEqual(d[0][1],1)
        for i,j,k in product(range(3),repeat=3):
            self.assertEqual(d[i][j],d[j][i])
            self.assertLessEqual(d[i][j],d[i][k]+d[k][j])

    def test_audit_returns_delayed_witness(self):
        w=exact_worlds()['delayed_witness']
        result=audit_partition(w.profiles(3),exact_quotient(w.profiles(1)))
        self.assertFalse(result['passed'])
        self.assertTrue(any(len(w['probe'])>1 for w in result['witnesses']))
        self.assertTrue(audit_partition(w.profiles(3),stable_memory(w))['passed'])

    def test_all_three_state_deterministic_sources(self):
        for successor in product(range(3),repeat=3):
            for outputs in product(range(2),repeat=3):
                w=deterministic((successor,),outputs)
                expected=min(len(p) for p in partitions(3) if lumpable(w,p))
                self.assertEqual(len(stable_memory(w)),expected)

    def test_invalid_and_capped_inputs(self):
        with self.assertRaises(TypeError):
            deterministic(((0,),),(0.1,))
        with self.assertRaises(ValueError):
            list(partitions(9))
        with self.assertRaises(ValueError):
            distortion(((0,0),),((F(0),),))
        with self.assertRaises(ValueError):
            exact_worlds()['delayed_witness'].profiles(6)


class LearnerTests(unittest.TestCase):
    def make(self, **kw):
        return PredictiveMemory(policy=PredictiveMemoryPolicy(confidence=False,min_samples=1,rebuild_every=10000,**kw))

    def test_chain_veto(self):
        for complete,expected in ((True,2),(False,1)):
            m=self.make(epsilon=.25,complete_link=complete)
            for c,k in enumerate((1,3,5)):
                for t in range(10):m.observe(c,int(t<k))
            m.rebuild()
            self.assertEqual(len(m.cells),expected)

    def test_fold_reopen_and_retained_evidence(self):
        m=self.make(window=20,epsilon=.2)
        for c in (0,1):
            for _ in range(20):m.observe(c,0)
        m.rebuild()
        self.assertEqual(len(m.cells),1)
        before=m.snapshot['retained_samples']
        for _ in range(20):m.observe(1,1)
        m.rebuild()
        self.assertEqual(len(m.cells),2)
        self.assertEqual(m.reopenings,1)
        self.assertEqual(m.snapshot['retained_samples'],before)
        self.assertLess(m.predict(0),.1)
        self.assertGreater(m.predict(1),.9)

    def test_budget_and_unknown_abstention_forecast(self):
        m=self.make(max_contexts=3,window=5)
        for t in range(1000):m.observe(t%17,t%2)
        self.assertLessEqual(m.snapshot['retained_samples'],15)
        self.assertEqual(m.predict('unseen'),.5)
        self.assertEqual({x for c in m.cells for x in c},set(m.evidence))

    def test_probe_domain_and_conservative_uncertainty(self):
        m=PredictiveMemory(probes=('a','b'))
        for c in (0,1):
            for _ in range(20):m.observe(c,0,'a')
        m.rebuild()
        self.assertEqual(len(m.cells),2)
        with self.assertRaises(ValueError):m.predict(0,'unknown')
        with self.assertRaises(ValueError):m.observe(0,2,'a')

    def test_adapter_late_and_bounded_frontiers(self):
        a=PredictiveEvidenceAdapter(('down','up'),PredictiveMemoryPolicy(max_contexts=2))
        self.assertEqual(a.predict(('x',)),.5)
        a.observe(EvidenceAtom(('x',),'up',10,10))
        self.assertFalse(a.observe(EvidenceAtom(('x',),'down',9,11)))
        for t in range(20):a.observe(EvidenceAtom((str(t),),'up',t,t))
        self.assertEqual(len(a.frontiers),2)
        self.assertEqual(a.late,1)


if __name__=='__main__':unittest.main()
