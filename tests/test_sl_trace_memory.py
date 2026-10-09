import unittest
from fractions import Fraction as F
from itertools import product
from cortex.trace_memory import TracePredictiveMemory
from research.sl_bench.trace_worlds import controlled_world,mixture_profiles
from research.sl_bench.oracle import exact_quotient,distances,audit_partition


class TraceTests(unittest.TestCase):
    def test_correlation_invisible_to_one_step(self):
        w,m=controlled_world();p=mixture_profiles(w,m)
        one=tuple({q:v for q,v in x.items() if len(q)==1} for x in p)
        self.assertEqual(exact_quotient(one),((0,1,2,3),))
        self.assertEqual(exact_quotient(p),((0,2),(1,3)))
        self.assertEqual(distances(p)[0][1],1)
        self.assertFalse(audit_partition(p,exact_quotient(one))['passed'])

    def test_delay_requires_three_steps(self):
        w,m=controlled_world(delay=True);p=mixture_profiles(w,m)
        two=tuple({q:v for q,v in x.items() if len(q)<=2} for x in p)
        self.assertEqual(len(exact_quotient(two)),1)
        self.assertEqual(len(exact_quotient(p)),4)

    def test_joint_normalization_prefix_and_window(self):
        m=TracePredictiveMemory(window=3)
        for _ in range(10):m.observe('c',(1,0,1),(0,1,1))
        for p in ((1,),(1,0),(1,0,1)):
            self.assertAlmostEqual(sum(m.predict('c',p)),1)
        self.assertEqual(m.snapshot['retained_traces'],9)
        self.assertEqual(m.snapshot['count_entries'],3)
        self.assertGreater(m.predict('c',(1,0,1))[3],.49)

    def test_validation_is_atomic(self):
        m=TracePredictiveMemory()
        before=m.snapshot
        for p,y in [((0,1),(0,)),((9,),(0,)),((0,),(9,))]:
            with self.assertRaises(ValueError):m.observe('c',p,y)
        self.assertEqual(before,m.snapshot)

    def test_unseen_domain_and_budget(self):
        m=TracePredictiveMemory(max_contexts=2,window=2)
        for i in range(20):m.observe(i,(0,),(1,))
        self.assertEqual(m.snapshot['contexts'],2)
        self.assertEqual(m.snapshot['evictions'],18)
        self.assertEqual(m.predict('new',(0,1)),(.25,)*4)
        cells,unknown=m.empirical_partition()
        self.assertEqual(len(cells),2);self.assertEqual(len(unknown),2)

    def test_joint_distinguishes_same_marginals(self):
        m=TracePredictiveMemory(actions=(0,),horizon=2,window=100)
        for i in range(100):
            m.observe('same',(0,0),(i%2,i%2))
            m.observe('opposite',(0,0),(i%2,1-i%2))
        self.assertEqual(m.predict('same',(0,)),m.predict('opposite',(0,)))
        cells,unknown=m.empirical_partition(epsilon=.1)
        self.assertEqual(len(cells),2);self.assertFalse(unknown)

if __name__=='__main__':unittest.main()
