"""Cortex v0.9 — Asymptotic Continual Reasoning.

A bounded-regime controller layered downstream of Cortex v0.8.
The frozen v0.8 active articulation remains unchanged. v0.9 adds:
- bounded recurrent regime memory,
- evidence-gated switching / novelty discovery,
- reactivation of previously learned regimes,
- explicit unresolved/budget-pressure states,
- bounded per-step prototype search at fixed budget B.

This file is an executable research prototype, not a general convergence proof.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np

from cortex_v08 import CortexV08Runtime, EPS


def rms(a: np.ndarray, b: np.ndarray) -> float:
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(np.sqrt(np.mean((a-b)**2)))


def bernoulli_kl(p: float, q: float) -> float:
    p=float(np.clip(p,1e-9,1-1e-9)); q=float(np.clip(q,1e-9,1-1e-9))
    return float(p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q)))


@dataclass
class RegimePrototype:
    centroid: np.ndarray
    count: float = 1.0
    successes: float = 1.0
    failures: float = 1.0
    utility: float = 1.0
    last_used: int = 0
    created_t: int = 0

    @property
    def mean(self) -> float:
        return float(self.successes / max(EPS, self.successes+self.failures))


@dataclass(frozen=True)
class ContinualRead:
    prediction: float
    memberships: np.ndarray
    active_ids: tuple[int,...]
    current_id: int | None
    nearest_id: int | None
    nearest_dist: float
    revision: bool
    reactivated: bool
    discovered: bool
    unresolved: bool
    budget_pressure: bool
    stored: int
    comparisons: int


class AsymptoticContinualReasoner:
    """Bounded fuzzy recurrent-regime reasoner.

    The controller does not alter Cortex v0.8's active articulation. It decides
    which historical regime prototype should be treated as currently relevant.

    A change is never accepted from one contradictory observation. Evidence must
    persist for ``switch_evidence`` steps. Known regimes are reactivated; novel
    regimes may be spawned only within the fixed budget or after a separately
    certified redundant pair is merged.
    """
    def __init__(self, dim: int, budget: int = 24, tau: float = 0.20,
                 alpha_cut: float = 0.10,
                 stay_tolerance: float = 0.28,
                 recurrence_tolerance: float = 0.38,
                 update_tolerance: float = 0.22,
                 switch_evidence: int = 4,
                 redundancy_tolerance: float = 0.14,
                 centroid_lr_cap: float = 0.08,
                 decay: float = 0.9995,
                 prior_strength: float = 2.0, recompress_interval: int = 32):
        self.dim=int(dim); self.budget=int(budget); self.tau=float(tau)
        self.alpha_cut=float(alpha_cut); self.stay_tolerance=float(stay_tolerance)
        self.recurrence_tolerance=float(recurrence_tolerance)
        self.update_tolerance=float(update_tolerance)
        self.switch_evidence=max(1,int(switch_evidence))
        self.redundancy_tolerance=float(redundancy_tolerance)
        self.centroid_lr_cap=float(centroid_lr_cap); self.decay=float(decay)
        self.prior_strength=float(prior_strength)
        self.recompress_interval=max(1,int(recompress_interval)); self._last_recompress_check=-10**9
        self.prototypes: list[RegimePrototype]=[]
        self.current_id: int|None=None
        self.t=0
        self.pending_kind: str|None=None
        self.pending_id: int|None=None
        self.pending_count=0
        self.pending_sum=np.zeros(self.dim,float)
        self.revision_count=0; self.reactivation_count=0; self.discovery_count=0
        self.unresolved_count=0; self.budget_pressure_count=0; self.recompression_count=0
        self.comparison_count=0

    def _matrix(self) -> np.ndarray:
        if not self.prototypes: return np.empty((0,self.dim),float)
        return np.stack([p.centroid for p in self.prototypes],axis=0)

    def distances(self, x: np.ndarray) -> np.ndarray:
        P=self._matrix()
        if len(P)==0: return np.empty(0,float)
        self.comparison_count += len(P)
        return np.sqrt(np.mean((P-x[None,:])**2,axis=1))

    def _memberships(self,d:np.ndarray)->np.ndarray:
        if len(d)==0:return np.empty(0,float)
        s=np.exp(-d/max(EPS,self.tau)); z=float(s.sum())
        if z<=EPS:
            m=np.zeros_like(s); m[int(np.argmin(d))]=1.; return m
        return s/z

    def _append(self,x:np.ndarray,t:int)->int:
        self.prototypes.append(RegimePrototype(x.copy(),1.0,self.prior_strength/2,self.prior_strength/2,1.0,t,t))
        self.discovery_count += 1
        return len(self.prototypes)-1

    def _reset_pending(self):
        self.pending_kind=None; self.pending_id=None; self.pending_count=0; self.pending_sum.fill(0.0)

    def _push_pending(self,kind:str,pid:int|None,x:np.ndarray)->bool:
        if self.pending_kind!=kind or self.pending_id!=pid:
            self.pending_kind=kind; self.pending_id=pid; self.pending_count=0; self.pending_sum.fill(0.0)
        self.pending_count += 1; self.pending_sum += x
        return self.pending_count>=self.switch_evidence

    def _certified_recompress(self)->bool:
        n=len(self.prototypes)
        if n<2:return False
        if self.t-self._last_recompress_check<self.recompress_interval:return False
        self._last_recompress_check=self.t
        P=self._matrix()
        D=np.sqrt(np.mean((P[:,None,:]-P[None,:,:])**2,axis=2))
        D[np.tril_indices(n)]=np.inf
        k=int(np.argmin(D)); i,j=np.unravel_index(k,D.shape); d=float(D[i,j])
        if not np.isfinite(d) or d>self.redundancy_tolerance:return False
        a,b=self.prototypes[i],self.prototypes[j]
        wa,wb=a.count,b.count
        c=(wa*a.centroid+wb*b.centroid)/max(EPS,wa+wb)
        # Redundancy certificate: merged centroid remains close to both sources.
        if max(rms(c,a.centroid),rms(c,b.centroid))>self.redundancy_tolerance:return False
        a.centroid=c; a.count=wa+wb; a.successes+=b.successes; a.failures+=b.failures
        a.utility+=b.utility; a.last_used=max(a.last_used,b.last_used)
        del self.prototypes[j]
        if self.current_id is not None:
            if self.current_id==j:self.current_id=i
            elif self.current_id>j:self.current_id-=1
        self.recompression_count += 1
        return True

    def _prediction(self,m:np.ndarray)->float:
        if not self.prototypes:return 0.5
        if len(m)==0:return self.prototypes[self.current_id or 0].mean
        active=np.where(m>=self.alpha_cut)[0]
        if len(active)==0: active=np.array([int(np.argmax(m))])
        w=m[active]; w=w/max(EPS,float(w.sum()))
        return float(sum(float(wi)*self.prototypes[int(i)].mean for wi,i in zip(w,active)))

    def step(self, x: Sequence[float], y: int|float|None=None) -> ContinualRead:
        self.t+=1
        x=np.asarray(x,float)
        if x.shape!=(self.dim,):raise ValueError((x.shape,self.dim))
        for p in self.prototypes:p.utility*=self.decay
        revision=reactivated=discovered=unresolved=budget_pressure=False

        if not self.prototypes:
            self.current_id=self._append(x,self.t); discovered=True; revision=True

        d=self.distances(x); m=self._memberships(d)
        nearest=int(np.argmin(d)) if len(d) else None
        nd=float(d[nearest]) if nearest is not None else float('inf')
        pred=self._prediction(m)  # pre-outcome prediction

        cur=self.current_id
        curd=float(d[cur]) if cur is not None and cur<len(d) else float('inf')
        if cur is not None and curd<=self.stay_tolerance:
            self._reset_pending()
            # Conservative centroid adaptation while current regime is supported.
            p=self.prototypes[cur]
            if curd<=self.update_tolerance:
                eta=min(self.centroid_lr_cap,1.0/max(2.0,p.count+1.0))
                p.centroid=(1-eta)*p.centroid+eta*x; p.count+=1.0
        else:
            # Prefer recurrence to an already-known prototype.
            cand=None
            if nearest is not None and nearest!=cur and nd<=self.recurrence_tolerance:
                cand=nearest
            if cand is not None:
                if self._push_pending('reactivate',cand,x):
                    self.current_id=cand; revision=True; reactivated=True
                    self.revision_count+=1; self.reactivation_count+=1
                    self._reset_pending()
            else:
                if self._push_pending('novel',None,x):
                    meanx=self.pending_sum/max(1,self.pending_count)
                    if len(self.prototypes)<self.budget or self._certified_recompress():
                        self.current_id=self._append(meanx,self.t); revision=True; discovered=True
                        self.revision_count+=1; self._reset_pending()
                    else:
                        unresolved=True; budget_pressure=True
                        self.unresolved_count+=1; self.budget_pressure_count+=1
                        # Keep accumulating evidence, but do not falsely merge.

        # Outcome updates only the articulation active after possible revision.
        if y is not None and self.current_id is not None:
            p=self.prototypes[self.current_id]
            yy=float(y)
            p.successes += yy; p.failures += 1.0-yy
            p.utility += 1.0; p.last_used=self.t

        # Recompute readout memberships only if a structural edit occurred.
        if revision or discovered:
            d=self.distances(x); m=self._memberships(d)
            nearest=int(np.argmin(d)) if len(d) else None
            nd=float(d[nearest]) if nearest is not None else float('inf')
        active=tuple(int(i) for i in np.where(m>=self.alpha_cut)[0]) if len(m) else tuple()
        if not active and nearest is not None:active=(nearest,)
        return ContinualRead(pred,m,active,self.current_id,nearest,nd,revision,reactivated,discovered,
                             unresolved,budget_pressure,len(self.prototypes),len(d))


class CortexV09Runtime:
    """Frozen v0.8 runtime plus downstream continual-regime controller."""
    def __init__(self, continual_dim:int=12, continual_budget:int=24,
                 continual_kwargs:dict|None=None, **v08_kwargs):
        self.core=CortexV08Runtime(memory_dim=continual_dim,memory_budget=continual_budget,**v08_kwargs)
        self.continual=AsymptoticContinualReasoner(continual_dim,budget=continual_budget,**(continual_kwargs or {}))

    @property
    def state(self):return self.core.state

    def step(self, frame, outcome: int|float|None=None):
        core_out,mem_out=self.core.step(frame)
        cont=self.continual.step(self.core.articulation_vector(),outcome)
        return core_out,mem_out,cont
