"""Bounded empirical joint-trace predictor for declared finite experiments.

Learner-visible inputs are context, open-loop action word, and observed output
word. Reset-state labels, transitions, and source probabilities are not inputs.
Each completed trial updates its prefixes only after its prediction was scored.
"""
from collections import Counter, OrderedDict, deque
from itertools import product
from math import isfinite


class TracePredictiveMemory:
    def __init__(self, actions=(0,1), symbols=(0,1), horizon=3,
                 max_contexts=32, window=256, action_conditioned=True):
        self.actions, self.symbols = tuple(actions), tuple(symbols)
        if not self.actions or not self.symbols or len(set(self.actions))!=len(self.actions) or len(set(self.symbols))!=len(self.symbols):
            raise ValueError('Nonempty unique alphabets required')
        if any(type(v) is not int or v<1 for v in (horizon,max_contexts,window)):
            raise ValueError('Positive integer caps required')
        if horizon>5 or sum((len(self.actions)*len(self.symbols))**h for h in range(1,horizon+1))>20000:
            raise ValueError('Declared trace domain exceeds resource cap')
        self.horizon,self.max_contexts,self.window = horizon,max_contexts,window
        self.action_conditioned=action_conditioned
        self.probes=tuple(p for h in range(1,horizon+1) for p in product(self.actions,repeat=h))
        self.words={h:tuple(product(self.symbols,repeat=h)) for h in range(1,horizon+1)}
        self.rows=OrderedDict()
        self.evictions=self.trials=0

    def _probe(self, probe):
        probe=tuple(probe)
        if probe not in self.probes:raise ValueError('Outside declared action/horizon domain')
        return probe

    def _key(self, probe):
        return probe if self.action_conditioned else len(probe)

    def predict(self, context, probe):
        probe=self._probe(probe)
        row=self.rows.get(context,{}).get(self._key(probe))
        counts,n=(row[1],len(row[0])) if row else ({},0)
        words=self.words[len(probe)]
        return tuple((counts.get(w,0)+.5)/(n+.5*len(words)) for w in words)

    def observe(self, context, probe, outcomes):
        probe=self._probe(probe);outcomes=tuple(outcomes)
        if len(outcomes)!=len(probe) or any(y not in self.symbols for y in outcomes):
            raise ValueError('Output word must match probe length and alphabet')
        hash(context)
        if context not in self.rows:
            if len(self.rows)>=self.max_contexts:
                self.rows.popitem(last=False);self.evictions+=1
            self.rows[context]={}
        self.rows.move_to_end(context)
        for h in range(1,len(probe)+1):
            key=self._key(probe[:h]);word=outcomes[:h]
            queue,counts=self.rows[context].setdefault(key,(deque(),Counter()))
            if len(queue)==self.window:
                old=queue.popleft();counts[old]-=1
                if counts[old]==0:del counts[old]
            queue.append(word);counts[word]+=1
        self.trials+=1

    def empirical_partition(self, epsilon=.2, min_samples=32):
        """Complete-link proposal using all declared trace distributions.

        Returns a partition and unresolved contexts. No population certificate:
        evaluator must audit with true profiles. Unseen probes block folding.
        Prediction itself remains unpooled regardless of this proposal.
        """
        if not isfinite(epsilon) or not 0<=epsilon<=1 or type(min_samples) is not int or min_samples<1:
            raise ValueError('Invalid proposal tolerance/support')
        profiles={};unresolved=[]
        for c,rows in self.rows.items():
            if any(len(rows.get(self._key(p),((),{}))[0])<min_samples for p in self.probes):
                unresolved.append(c);continue
            profiles[c]={}
            for p in self.probes:
                queue,counts=rows[self._key(p)]
                profiles[c][p]=tuple(counts.get(w,0)/len(queue) for w in self.words[len(p)])
        cells=[]
        for c in self.rows:
            for cell in cells:
                if c not in profiles or any(x not in profiles for x in cell):continue
                if all(max(sum(abs(a-b) for a,b in zip(profiles[c][p],profiles[x][p]))/2 for p in self.probes)<=epsilon for x in cell):
                    cell.append(c);break
            else:cells.append([c])
        return tuple(tuple(c) for c in cells),tuple(unresolved)

    @property
    def snapshot(self):
        return dict(contexts=len(self.rows),retained_traces=sum(len(q) for rows in self.rows.values() for q,_ in rows.values()),
                    count_entries=sum(len(c) for rows in self.rows.values() for _,c in rows.values()),trials=self.trials,evictions=self.evictions)
