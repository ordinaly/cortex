import unittest
from cortex.accordion_memory import AccordionMemory,AccordionPolicy,retained_size


class AccordionTests(unittest.TestCase):
    def test_fold_actually_discards_full_windows(self):
        m=AccordionMemory(AccordionPolicy(window=1024,witness=8,fold_every=10000))
        for i in range(1024):
            for c in range(8):m.observe(c,i%2)
        before=m.snapshot['payload_bytes'];size=retained_size(m);m.fold()
        self.assertLess(retained_size(m),size*.95)
        self.assertFalse(m.active)
        self.assertEqual(m.snapshot['payload_bytes'],1088)
        self.assertLess(m.snapshot['payload_bytes'],before)
        self.assertEqual(set(m.witnesses),set(range(8)))

    def test_local_reopen_has_only_retained_samples(self):
        m=AccordionMemory(AccordionPolicy(window=1024,witness=8,fold_every=10000))
        for i in range(1024):
            for c in range(4):m.observe(c,0)
        m.fold()
        for _ in range(8):m.observe(0,1)
        self.assertIn(0,m.active)
        self.assertLessEqual(len(m.active[0]),16)
        self.assertEqual(len(m.membership),3)
        self.assertGreater(m.predict(0),.5)
        self.assertLess(m.predict(1),.1)

    def test_pressure_bound_and_no_dangling_members(self):
        m=AccordionMemory(AccordionPolicy(window=16,witness=4,max_contexts=4,fold_every=16))
        for i in range(1000):m.observe((i//64)%8,0)
        self.assertLessEqual(m.snapshot['contexts'],4)
        self.assertLessEqual(m.snapshot['payload_bytes'],4*16)
        self.assertEqual(set(m.order),set(m.active)|set(m.membership))
        self.assertEqual(set(m.witnesses),set(m.membership))
        self.assertTrue(all(c in m.order for members,_ in m.cells.values() for c in members))

    def test_veto_chain_and_invalid_input(self):
        m=AccordionMemory(AccordionPolicy(window=10000,witness=10,fold_tolerance=.21,fold_every=100000))
        for c,p in enumerate((1000,3000,5000)):
            for t in range(10000):m.observe(c,int(t<p))
        m.fold()
        self.assertEqual(max(map(lambda x:len(x[0]),m.cells.values())),2)
        before=m.snapshot
        with self.assertRaises(ValueError):m.observe('bad',2)
        self.assertEqual(before,m.snapshot)

    def test_unprofitable_fold_rolls_back_exactly(self):
        m=AccordionMemory(AccordionPolicy(window=64,witness=8,fold_every=10000))
        for i in range(64):
            for c in range(8):m.observe(c,i%2)
        indices=(m.active,m.witnesses,m.membership,m.cells)
        predictions=[m.predict(c) for c in range(8)]
        size=retained_size(m)
        m.fold()
        self.assertEqual(m.folds,0)
        self.assertEqual(m.next_id,0)
        self.assertTrue(all(a is b for a,b in zip(indices,(m.active,m.witnesses,m.membership,m.cells))))
        self.assertEqual(predictions,[m.predict(c) for c in range(8)])
        self.assertEqual(size,retained_size(m))

    def test_eviction_dissolves_two_member_cell(self):
        m=AccordionMemory(AccordionPolicy(window=2048,witness=8,max_contexts=2,fold_every=10000))
        for i in range(2048):
            for c in (0,1):m.observe(c,0)
        m.fold()
        self.assertEqual(len(m.membership),2)
        m.observe(2,1)
        self.assertFalse(m.cells)
        self.assertFalse(m.membership)
        self.assertFalse(m.witnesses)
        self.assertEqual(set(m.active),{1,2})
        self.assertEqual(len(m.active[1]),8)
        self.assertEqual(m.evictions,1)

if __name__=='__main__':unittest.main()
