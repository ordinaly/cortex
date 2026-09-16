"""Cortex v0.7 — Integrated Articulation Runtime.

Contract-faithful reimplementation of the validated Core XI/XII learning ideas
inside the v0.6 dependency runtime. This file does NOT claim byte-for-byte
identity with the frozen Consolidated Implementation v0.3 source archive,
which is not mounted in this runtime. It preserves the frozen conceptual
contracts available here:

- persistent entity prototypes under transformation nuisance,
- soft transformation evidence and selective subgroup inference,
- injective simultaneous binding,
- endogenous Beta/Bernoulli residual/noise revision,
- predictive pairwise relation/event induction,
- intervention-sensitive Bayesian causal contrast,
- dependency-directed recomputation of derived articulated state.

The benchmark compares an eager derivation path with a dependency path over
identical online sufficient-statistic updates. The point is execution/locality,
not a new cognitive algorithm claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import exp, log
from typing import Iterable, Sequence
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment

EPS = 1e-12


class EntityCapacityError(RuntimeError):
    """Raised when injective binding cannot be preserved within the declared entity budget."""



def softmax_neg(costs: np.ndarray, beta: float) -> np.ndarray:
    z = -beta * np.asarray(costs, dtype=float)
    z -= z.max()
    e = np.exp(z)
    return e / max(EPS, e.sum())


def shift_vec(x: np.ndarray, g: int) -> np.ndarray:
    return np.roll(x, int(g))


def inv_shift_vec(x: np.ndarray, g: int) -> np.ndarray:
    return np.roll(x, -int(g))


def weighted_mse(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float(np.sum(w * (x-y)**2) / max(EPS, np.sum(w)))


@dataclass(frozen=True)
class Detection:
    x: np.ndarray
    true_entity: int = -1  # benchmark only; learner never reads it


@dataclass(frozen=True)
class Frame:
    detections: tuple[Detection, ...]
    relation_obs: tuple[tuple[int,int,int], ...]  # detection-index pair, observed relation bit
    intervention_src_det: int | None
    outcomes: tuple[tuple[int,int], ...]  # (target detection index, binary outcome)


@dataclass
class EntityStats:
    sum_x: np.ndarray
    count: int
    noise_a: np.ndarray
    noise_b: np.ndarray
    last_seen: int
    last_x: np.ndarray


@dataclass(frozen=True)
class ArticulationSnapshot:
    active_entities: int
    subgroup: tuple[int, ...]
    subgroup_margin: float
    prototypes: tuple[tuple[float, ...], ...]
    noise_state: tuple[tuple[int, ...], ...]
    relation_state: tuple[int, ...]
    causal_state: tuple[int, ...]


@dataclass(frozen=True)
class StepOutput:
    bindings: tuple[int, ...]
    subgroup: tuple[int, ...]
    subgroup_margin: float
    relation_changed: tuple[int, ...]
    causal_changed: tuple[int, ...]
    entity_changed: tuple[int, ...]
    noise_changed: tuple[int, ...]
    reopened_cells: int
    total_cells: int

    @property
    def reopened_fraction(self) -> float:
        return self.reopened_cells / max(1, self.total_cells)


class PairIndex:
    def __init__(self, max_entities: int):
        self.max_entities = int(max_entities)
        self.pairs = tuple(combinations(range(max_entities), 2))
        self.of = {p:i for i,p in enumerate(self.pairs)}
        self.ordered = tuple((i,j) for i in range(max_entities) for j in range(max_entities) if i != j)
        self.ordered_of = {p:i for i,p in enumerate(self.ordered)}

    def pair_id(self, a:int,b:int)->int:
        if a>b: a,b=b,a
        return self.of[(a,b)]

    def ordered_id(self,a:int,b:int)->int:
        return self.ordered_of[(a,b)]


class IntegratedState:
    """Online sufficient statistics plus cached articulated state."""
    def __init__(self, feature_dim=16, max_entities=96, beta=9.0,
                 spawn_cost=0.7, bootstrap_spawn_cost=0.18, residual_threshold=0.38,
                 group_candidates: Sequence[Sequence[int]] | None=None):
        self.d=int(feature_dim); self.max_entities=int(max_entities); self.beta=float(beta)
        self.spawn_cost=float(spawn_cost); self.bootstrap_spawn_cost=float(bootstrap_spawn_cost); self.residual_threshold=float(residual_threshold)
        if group_candidates is None:
            if self.d % 4 != 0:
                group_candidates=[(0,), tuple(range(self.d))]
            else:
                group_candidates=[(0,), (0,self.d//2), tuple(range(0,self.d,self.d//4)), tuple(range(self.d))]
        self.group_candidates=tuple(tuple(map(int,g)) for g in group_candidates)
        self.all_shifts=tuple(range(self.d))
        self.entities:list[EntityStats]=[]
        self.prototypes:list[np.ndarray]=[]
        self.reliability:list[np.ndarray]=[]
        self.noise_state:list[np.ndarray]=[]
        self.group_evidence=np.zeros(len(self.group_candidates),dtype=float)
        self.group_obs=0
        self.group_evidence_by_entity=np.zeros((self.max_entities,len(self.group_candidates)),dtype=float)
        self.group_obs_by_entity=np.zeros(self.max_entities,dtype=np.int32)
        self.subgroup=self.all_shifts
        self.subgroup_margin=0.0
        self.pidx=PairIndex(self.max_entities)
        m=len(self.pidx.pairs); mo=len(self.pidx.ordered)
        # Relation: Beta posterior on observed relation bit conditional on co-visibility.
        self.rel_a=np.ones(m,dtype=float); self.rel_b=np.ones(m,dtype=float)
        self.rel_exposure=np.zeros(m,dtype=np.int32); self.relation_state=np.zeros(m,dtype=np.int8) # -1 absent,0 unresolved,+1 active
        # Causal: Beta outcome under intervention on source vs matched non-intervention baseline.
        self.c_do_a=np.ones(mo); self.c_do_b=np.ones(mo); self.c_ctrl_a=np.ones(mo); self.c_ctrl_b=np.ones(mo)
        self.c_do_n=np.zeros(mo,dtype=np.int32); self.c_ctrl_n=np.zeros(mo,dtype=np.int32)
        self.causal_state=np.zeros(mo,dtype=np.int8) # -1 null,0 unresolved,+1 causal
        self.t=0

    def clone(self):
        return copy.deepcopy(self)

    # ---------- derived cells ----------
    def derive_prototype(self,e:int)->np.ndarray:
        st=self.entities[e]
        return st.sum_x/max(1,st.count)

    def derive_reliability(self,e:int)->np.ndarray:
        st=self.entities[e]
        p=st.noise_a/(st.noise_a+st.noise_b)
        return np.clip(1.0-p,0.18,1.0)

    def derive_noise_state(self,e:int)->np.ndarray:
        st=self.entities[e]
        p=st.noise_a/(st.noise_a+st.noise_b)
        n=st.noise_a+st.noise_b-2.0
        out=np.zeros(self.d,dtype=np.int8)
        out[(n>=6)&(p>=0.30)] = 2       # noise-supported
        out[(n>=6)&(p<=0.10)] = -1      # structure-supported
        return out

    def derive_relation_state(self,pid:int)->int:
        n=int(self.rel_exposure[pid]); p=float(self.rel_a[pid]/(self.rel_a[pid]+self.rel_b[pid]))
        if n<8: return 0
        if p>=0.68: return 1
        if p<=0.32: return -1
        return 0

    def derive_causal_state(self,oid:int)->int:
        nd=int(self.c_do_n[oid]); nc=int(self.c_ctrl_n[oid])
        if nd<8 or nc<8: return 0
        pd=float(self.c_do_a[oid]/(self.c_do_a[oid]+self.c_do_b[oid]))
        pc=float(self.c_ctrl_a[oid]/(self.c_ctrl_a[oid]+self.c_ctrl_b[oid]))
        diff=pd-pc
        if diff>=0.24: return 1
        if abs(diff)<=0.10: return -1
        return 0

    def derive_subgroup(self)->tuple[tuple[int,...],float]:
        # Robust synchronization consensus: each established entity contributes
        # one normalized vote, so a fragmented or temporarily misbound track
        # cannot dominate merely by accumulating many observations.
        eligible=np.where(self.group_obs_by_entity>=4)[0]
        if len(eligible)<2 or int(self.group_obs_by_entity[eligible].sum()) < 12:
            return self.all_shifts, 0.0
        per=self.group_evidence_by_entity[eligible] / self.group_obs_by_entity[eligible,None]
        scores=np.median(per,axis=0)
        order=np.argsort(scores)[::-1]
        best=int(order[0]); second=int(order[1]) if len(order)>1 else best
        margin=float(scores[best]-scores[second])
        return self.group_candidates[best], margin

    # ---------- matching / binding ----------
    def _entity_cost_and_q(self,x:np.ndarray,e:int,match_group:Sequence[int]|None=None):
        proto=self.prototypes[e]; w=self.reliability[e]
        shifts=self.all_shifts if match_group is None else tuple(match_group)
        costs=np.array([weighted_mse(x,shift_vec(proto,g),w) for g in shifts])
        q=softmax_neg(costs,self.beta)
        j=int(np.argmin(costs)); return float(costs[j]), int(shifts[j]), shifts, q

    def bind(self,detections:Sequence[Detection])->tuple[list[int],list[int],list[tuple[tuple[int,...],np.ndarray]]]:
        k=len(detections); ne=len(self.entities)
        if ne==0:
            binds=[]; shifts=[]; qs=[]
            for det in detections:
                e=self._spawn(det.x); binds.append(e); shifts.append(0); qs.append((self.all_shifts,np.ones(self.d)/self.d))
            return binds,shifts,qs
        # During subgroup bootstrap, arbitrary transformations can make distinct
        # entities look spuriously similar. Use a conservative spawn gate until
        # synchronization becomes confident; then relax to the normal gate under
        # the selected nuisance subgroup. This is the entity/invariance
        # co-refinement guard.
        confident = self.subgroup_margin > 0.01
        mg=self.subgroup if confident else self.all_shifts
        spawn_gate = self.spawn_cost if confident else min(self.spawn_cost, self.bootstrap_spawn_cost)
        C=np.full((k, ne+k), spawn_gate, dtype=float)
        best_shift=np.zeros((k,ne),dtype=int)
        qcache={}
        for i,det in enumerate(detections):
            for e in range(ne):
                c,g,sh,q=self._entity_cost_and_q(det.x,e,mg)
                C[i,e]=c; best_shift[i,e]=g; qcache[(i,e)]=(sh,q)
            # distinct dummy columns; slight deterministic preference by row
            for j in range(k): C[i,ne+j]=spawn_gate + 1e-8*abs(i-j)
        rows,cols=linear_sum_assignment(C)
        binds=[-1]*k; shifts=[0]*k; qs=[None]*k
        for i,c in zip(rows,cols):
            if c<ne and C[i,c] <= spawn_gate:
                binds[i]=int(c); shifts[i]=int(best_shift[i,c]); qs[i]=qcache[(i,c)]
            else:
                e=self._spawn(detections[i].x); binds[i]=e; shifts[i]=0; qs[i]=(self.all_shifts,np.ones(self.d)/self.d)
        return binds,shifts,qs

    def _spawn(self,x:np.ndarray)->int:
        if len(self.entities)>=self.max_entities:
            # Never silently violate simultaneous-binding injectivity. Capacity
            # exhaustion is a resource failure and must remain explicit.
            raise EntityCapacityError(
                f"entity capacity exhausted ({self.max_entities}); cannot spawn a distinct identity"
            )
        x=np.asarray(x,dtype=float).copy()
        st=EntityStats(sum_x=x.copy(),count=1,
                       noise_a=np.ones(self.d),noise_b=np.ones(self.d),last_seen=self.t,last_x=x.copy())
        self.entities.append(st); self.prototypes.append(x.copy()); self.reliability.append(np.full(self.d,0.5)); self.noise_state.append(np.zeros(self.d,dtype=np.int8))
        return len(self.entities)-1

    def snapshot(self)->ArticulationSnapshot:
        ne=len(self.entities)
        prot=tuple(tuple(np.round(self.prototypes[e],10)) for e in range(ne))
        ns=tuple(tuple(map(int,self.noise_state[e])) for e in range(ne))
        return ArticulationSnapshot(ne,tuple(self.subgroup),float(self.subgroup_margin),prot,ns,
                                    tuple(map(int,self.relation_state)),tuple(map(int,self.causal_state)))

    # ---------- shared sufficient-stat update ----------
    def update_statistics(self, frame:Frame, eager:bool=False)->StepOutput:
        """Process one frame. `eager` controls only derivation scope, not statistics."""
        self.t += 1
        binds,shifts,qs=self.bind(frame.detections)
        if len(set(binds)) != len(binds):
            raise AssertionError("simultaneous binding must be injective within a frame")
        touched_entities=set(); dirty_noise=set(); relation_touched=set(); causal_touched=set()
        old_ne=len(self.entities)
        # entity/prototype/noise stats
        for i,(det,e,g,qinfo) in enumerate(zip(frame.detections,binds,shifts,qs)):
            touched_entities.add(e)
            aligned=inv_shift_vec(det.x,g)
            st=self.entities[e]
            proto_before=self.prototypes[e]
            resid=np.abs(aligned-proto_before)
            anomalies=resid>self.residual_threshold
            st.sum_x += aligned; st.count += 1; st.last_seen=self.t
            st.noise_a += anomalies.astype(float); st.noise_b += (~anomalies).astype(float)
            dirty_noise.add(e)
            # Gauge-invariant subgroup evidence from relative transforms between
            # repeated observations of the same bound entity. If x_t=g_t p and
            # x_{t-1}=g_{t-1}p, then the relative transform lies in H whenever
            # g_t,g_{t-1} lie in subgroup H, independent of the entity gauge.
            # A freshly spawned entity has no relative-transform witness yet: its
            # stored last_x is the current observation itself. Only established
            # identities may contribute synchronization evidence. This prevents
            # a self-comparison from masquerading as a zero-transform witness.
            bind_cost=weighted_mse(det.x,shift_vec(proto_before,g),self.reliability[e])
            if st.count > 2 and bind_cost <= 0.35:
                prev_raw=st.last_x
                costs=np.array([weighted_mse(det.x,shift_vec(prev_raw,h),np.ones(self.d)) for h in self.all_shifts])
                qall=softmax_neg(costs,self.beta)
                # Reliability gate: ambiguous relative-transform evidence is
                # almost neutral and should not steer subgroup selection.
                concentration=float(np.max(qall))
                if concentration >= 0.18:
                    for hi,H in enumerate(self.group_candidates):
                        support=float(np.sum(qall[list(H)]))
                        ev=log(max(EPS,support/len(H)))
                        self.group_evidence[hi]+=ev
                        self.group_evidence_by_entity[e,hi]+=ev
                    self.group_obs+=1
                    self.group_obs_by_entity[e]+=1
            st.last_x=np.asarray(det.x,dtype=float).copy()
        # relation stats; relation_obs uses detection indices
        rel_lookup={(min(i,j),max(i,j)):int(v) for i,j,v in frame.relation_obs}
        for i,j in combinations(range(len(binds)),2):
            a,b=binds[i],binds[j]
            if a==b: continue
            pid=self.pidx.pair_id(a,b); y=rel_lookup.get((i,j),0)
            self.rel_a[pid]+=y; self.rel_b[pid]+=1-y; self.rel_exposure[pid]+=1; relation_touched.add(pid)
        # causal statistics for visible ordered pairs. Outcome is attached to target detection.
        outmap={int(i):int(y) for i,y in frame.outcomes}
        src_det=frame.intervention_src_det
        for i,a in enumerate(binds):
            for j,b in enumerate(binds):
                if i==j or a==b or j not in outmap: continue
                oid=self.pidx.ordered_id(a,b); y=outmap[j]
                if src_det is not None and i==src_det:
                    self.c_do_a[oid]+=y; self.c_do_b[oid]+=1-y; self.c_do_n[oid]+=1
                else:
                    self.c_ctrl_a[oid]+=y; self.c_ctrl_b[oid]+=1-y; self.c_ctrl_n[oid]+=1
                causal_touched.add(oid)

        # derive articulated states
        entity_changed=[]; noise_changed=[]; relation_changed=[]; causal_changed=[]
        reopened=0
        # Prototypes/reliability: eager derives all active entities, dependency only touched.
        entity_scope=range(len(self.entities)) if eager else sorted(touched_entities)
        for e in entity_scope:
            reopened += 2  # prototype + reliability cell
            p=self.derive_prototype(e); r=self.derive_reliability(e)
            if not np.allclose(p,self.prototypes[e],atol=0,rtol=0): self.prototypes[e]=p; entity_changed.append(e)
            self.reliability[e]=r
        noise_scope=range(len(self.entities)) if eager else sorted(dirty_noise)
        for e in noise_scope:
            reopened += self.d
            ns=self.derive_noise_state(e)
            if not np.array_equal(ns,self.noise_state[e]): self.noise_state[e]=ns; noise_changed.append(e)
        # subgroup is one global cell; evidence changes every frame, so both reopen exactly once.
        reopened+=1
        sg,margin=self.derive_subgroup(); subgroup_changed=(tuple(sg)!=tuple(self.subgroup))
        self.subgroup=tuple(sg); self.subgroup_margin=float(margin)
        if eager:
            rel_scope=[self.pidx.pair_id(a,b) for a,b in combinations(range(len(self.entities)),2)]
        else:
            rel_scope=sorted(relation_touched)
        for pid in rel_scope:
            reopened += 1
            v=self.derive_relation_state(pid)
            if v!=int(self.relation_state[pid]): self.relation_state[pid]=v; relation_changed.append(pid)
        if eager:
            ca_scope=[self.pidx.ordered_id(a,b) for a in range(len(self.entities)) for b in range(len(self.entities)) if a!=b]
        else:
            ca_scope=sorted(causal_touched)
        for oid in ca_scope:
            reopened += 1
            v=self.derive_causal_state(oid)
            if v!=int(self.causal_state[oid]): self.causal_state[oid]=v; causal_changed.append(oid)
        # If subgroup changed, conceptually invalidate current matching policy; no historical rebuild required.
        ne=len(self.entities); active_rel=ne*(ne-1)//2; active_causal=ne*(ne-1)
        total_cells=max(1, ne*(2+self.d)+1+active_rel+active_causal)
        return StepOutput(tuple(binds),tuple(self.subgroup),self.subgroup_margin,tuple(relation_changed),tuple(causal_changed),
                          tuple(sorted(set(entity_changed))),tuple(noise_changed),int(reopened),int(total_cells))


class EagerIntegratedRuntime:
    def __init__(self,**kw): self.state=IntegratedState(**kw)
    def step(self,frame:Frame): return self.state.update_statistics(frame,eager=True)

class DependencyIntegratedRuntime:
    def __init__(self,**kw): self.state=IntegratedState(**kw)
    def step(self,frame:Frame): return self.state.update_statistics(frame,eager=False)


class SyntheticIntegratedWorld:
    """Controlled multi-entity world for integration/locality testing."""
    def __init__(self,n_entities=16,feature_dim=16,visible=4,seed=0,noise_sigma=0.06,corrupt_p=0.035,intervention_p=0.35):
        self.rng=np.random.default_rng(seed); self.n=int(n_entities); self.d=int(feature_dim); self.visible=int(visible)
        self.noise_sigma=float(noise_sigma); self.corrupt_p=float(corrupt_p); self.intervention_p=float(intervention_p); self.t=0
        # distinct structured prototypes on a cycle
        raw=self.rng.normal(size=(self.n,self.d))
        # smooth and normalize to create nontrivial but separated cyclic shapes
        sm=(raw+0.55*np.roll(raw,1,axis=1)+0.35*np.roll(raw,-1,axis=1))
        sm=(sm-sm.mean(1,keepdims=True))/(sm.std(1,keepdims=True)+1e-6)
        self.proto=sm
        self.true_group=(0,self.d//4,self.d//2,3*self.d//4)
        # sparse symmetric relation graph
        A=self.rng.random((self.n,self.n))<min(0.22,4/max(4,self.n)); A=np.triu(A,1); self.rel=A|A.T
        # sparse directed causal graph, avoiding self loops
        C=self.rng.random((self.n,self.n))<min(0.10,2/max(4,self.n)); np.fill_diagonal(C,False); self.causal=C
        self.local_shift_time=None; self.shift_entity=None

    def schedule_local_drift(self,t:int,entity:int=0): self.local_shift_time=int(t); self.shift_entity=int(entity)

    def frame(self)->Frame:
        self.t+=1
        if self.local_shift_time is not None and self.t==self.local_shift_time:
            e=self.shift_entity; self.proto[e]=0.70*self.proto[e]+0.30*self.rng.normal(size=self.d)
            self.proto[e]=(self.proto[e]-self.proto[e].mean())/(self.proto[e].std()+1e-6)
            # also toggle one adjacent relation to make a local structural change
            j=(e+1)%self.n; self.rel[e,j]=~self.rel[e,j]; self.rel[j,e]=self.rel[e,j]
        ids=self.rng.choice(self.n,size=min(self.visible,self.n),replace=False)
        dets=[]
        for e in ids:
            g=int(self.rng.choice(self.true_group)); x=shift_vec(self.proto[e],g)+self.rng.normal(scale=self.noise_sigma,size=self.d)
            bad=self.rng.random(self.d)<self.corrupt_p
            x[bad]+=self.rng.normal(scale=1.25,size=int(bad.sum()))
            dets.append(Detection(x.astype(float),int(e)))
        relobs=[]
        for i,j in combinations(range(len(ids)),2):
            truth=bool(self.rel[ids[i],ids[j]]); p=0.90 if truth else 0.10
            y=int(self.rng.random()<p); relobs.append((i,j,y))
        # interventions occur in 35% frames; target outcomes cover every other visible detection
        src_det=int(self.rng.integers(len(ids))) if len(ids)>1 and self.rng.random()<self.intervention_p else None
        outcomes=[]
        for j,target in enumerate(ids):
            if src_det is not None and j!=src_det:
                src=ids[src_det]; p=0.86 if self.causal[src,target] else 0.20
            else:
                p=0.20
            outcomes.append((j,int(self.rng.random()<p)))
        return Frame(tuple(dets),tuple(relobs),src_det,tuple(outcomes))
