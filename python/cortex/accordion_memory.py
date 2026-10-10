"""Experimental binary accordion: physical folding with bounded witnesses.

Folds discard per-context full windows. Reopening recovers only retained witness
samples, never the discarded history. No population sufficiency certificate.
"""
from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite
import sys


@dataclass(frozen=True)
class AccordionPolicy:
    window: int = 256
    witness: int = 32
    max_contexts: int = 64
    fold_every: int = 256
    fold_tolerance: float = .10
    reopen_threshold: float = .30
    folding: bool = True
    reopening: bool = True
    min_retained_saving: float = .05

    def __post_init__(self):
        if any(type(x) is not int or x<1 for x in (self.window,self.witness,self.max_contexts,self.fold_every)):
            raise ValueError('Positive integer capacities required')
        if self.witness>=self.window:raise ValueError('Witness must be smaller than full window')
        if not all(isfinite(x) and 0<=x<=1 for x in (self.fold_tolerance,self.reopen_threshold,self.min_retained_saving)):
            raise ValueError('Thresholds must lie in [0,1]')


def retained_size(value,seen=None):
    """CPython object graph estimate, not peak allocation or process RSS."""
    seen=set() if seen is None else seen
    if id(value) in seen:return 0
    seen.add(id(value));size=sys.getsizeof(value)
    if isinstance(value,dict):size+=sum(retained_size(k,seen)+retained_size(v,seen) for k,v in value.items())
    elif isinstance(value,(tuple,list,set)):size+=sum(retained_size(v,seen) for v in value)
    elif hasattr(value,'__dict__'):size+=retained_size(vars(value),seen)
    return size


def append_bounded(row,y,cap):
    row.append(y)
    if len(row)>cap:del row[0]


def mean(row):return sum(row)/len(row) if row else .5


class AccordionMemory:
    def __init__(self,policy=None):
        self.policy=policy or AccordionPolicy()
        self.order=OrderedDict()
        self.active={}
        self.witnesses={}
        self.membership={}
        self.cells={}
        self.cooldown={}
        self.steps=self.folds=self.reopens=self.evictions=self.next_id=0

    def predict(self,context):
        if context in self.membership:row=self.cells[self.membership[context]][1]
        else:row=self.active.get(context,())
        return (sum(row)+.5)/(len(row)+1)

    def _remove_member(self,context):
        cell_id=self.membership.pop(context)
        members,_=self.cells[cell_id]
        members.remove(context)
        if not members:del self.cells[cell_id]
        elif len(members)==1:
            other=next(iter(members));self.membership.pop(other)
            self.active[other]=bytearray(self.witnesses.pop(other))
            del self.cells[cell_id]
        else:
            # Old pooled samples have lost ownership; rebuild from surviving
            # witnesses rather than letting the departed member contaminate it.
            self.cells[cell_id]=(members,self._pool(members,self.witnesses))

    def _pool(self,members,rows):
        # Interleave equal-depth suffixes; never fabricate or duplicate samples.
        ordered=sorted(members,key=lambda x:self.order[x])
        depth=max(len(rows[x]) for x in ordered)
        values=bytearray()
        for i in range(depth,0,-1):
            for x in ordered:
                if len(rows[x])>=i:append_bounded(values,rows[x][-i],self.policy.window)
        return values

    def fold(self):
        if not self.policy.folding:return
        eligible=[c for c,row in self.active.items() if len(row)==self.policy.window and self.cooldown.get(c,0)==0]
        groups=[]
        for c in eligible:
            for group in groups:
                if all(abs(mean(self.active[c])-mean(self.active[x]))<=self.policy.fold_tolerance for x in group):
                    group.append(c);break
            else:groups.append([c])
        for group in groups:
            # Cheap payload veto before measuring the complete retained object graph.
            if len(group)<2 or self.policy.window+len(group)*self.policy.witness>=len(group)*self.policy.window:continue
            before=retained_size(self)
            previous=(self.active,self.witnesses,self.membership,self.cells,self.next_id,self.folds)
            # Copy indices only; full evidence buffers are not copied.
            self.active=dict(self.active);self.witnesses=dict(self.witnesses)
            self.membership=dict(self.membership);self.cells=dict(self.cells)
            pool=self._pool(group,self.active)
            cell_id=self.next_id
            for c in group:
                self.witnesses[c]=self.active.pop(c)[-self.policy.witness:]
                self.membership[c]=cell_id
            self.cells[cell_id]=(list(group),pool)
            self.active=dict(self.active)  # release deleted dict capacity
            self.next_id+=1;self.folds+=1
            # Include committed counters, all indices, witnesses and shared pool.
            # Temporary rollback buffers are peak costs, not retained state.
            if retained_size(self)>=before*(1-self.policy.min_retained_saving):
                (self.active,self.witnesses,self.membership,self.cells,
                 self.next_id,self.folds)=previous
            del previous

    def observe(self,context,outcome):
        if type(outcome) is not int or outcome not in (0,1):raise ValueError('Binary integer required')
        hash(context)
        if context not in self.order:
            if len(self.order)==self.policy.max_contexts:
                victim=next(iter(self.order))
                if victim in self.membership:self._remove_member(victim);self.witnesses.pop(victim)
                else:self.active.pop(victim)
                self.order.pop(victim);self.cooldown.pop(victim,None);self.evictions+=1
            self.active[context]=bytearray()
        self.order[context]=self.steps;self.order.move_to_end(context)
        self.steps+=1
        if context in self.membership:
            cell_id=self.membership[context]
            witness=self.witnesses[context]
            append_bounded(witness,outcome,self.policy.witness)
            pool=self.cells[cell_id][1]
            # Compare against pool BEFORE training this observation.
            reopen=(self.policy.reopening and len(witness)==self.policy.witness and abs(mean(witness)-mean(pool))>self.policy.reopen_threshold)
            if reopen:
                retained=bytearray(witness)
                self._remove_member(context);self.witnesses.pop(context)
                self.active[context]=retained
                self.cooldown[context]=self.policy.window
                self.reopens+=1
            else:append_bounded(pool,outcome,self.policy.window)
        else:
            append_bounded(self.active[context],outcome,self.policy.window)
            if context in self.cooldown:
                self.cooldown[context]-=1
                if self.cooldown[context]<=0:del self.cooldown[context]
        if self.steps%self.policy.fold_every==0:self.fold()

    @property
    def snapshot(self):
        return dict(contexts=len(self.order),active_contexts=len(self.active),folded_contexts=len(self.membership),cells=len(self.cells),
                    payload_bytes=sum(map(len,self.active.values()))+sum(map(len,self.witnesses.values()))+sum(len(row) for _,row in self.cells.values()),
                    folds=self.folds,reopens=self.reopens,evictions=self.evictions)
