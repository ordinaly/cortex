"""Cortex v0.9.2-alpha — Tensor-Directed Plasticity.

Experimental Cortex-only extension. It preserves v0.9.1 structural plasticity and
adds bounded low-rank residual tensors for parametric plasticity inside known regimes.
"""
import sys, math
from dataclasses import dataclass
from typing import Sequence
import numpy as np
sys.path[:0]=['/mnt/data/cortex_v09','/mnt/data/cortex_v08','/mnt/data/cortex_v07_final']
from cortex_v09 import AsymptoticContinualReasoner, ContinualRead, EPS


def logit(p):
    p=float(np.clip(p,1e-4,1-1e-4)); return math.log(p/(1-p))

def sigmoid(z):
    z=float(np.clip(z,-25,25)); return 1/(1+math.exp(-z))

class AdaptiveEvidenceReasoner(AsymptoticContinualReasoner):
    def __init__(self, *args, evidence_decay=.65, noise_lr=.02, hazard_lr=.25,
                 recurrence_threshold=2.0, novelty_threshold=3.5,
                 hazard_discount_rec=.55, hazard_discount_nov=.35,
                 min_sigma=.08, **kwargs):
        kwargs['switch_evidence']=999999
        super().__init__(*args, **kwargs)
        self.evidence_decay=float(evidence_decay); self.noise_lr=float(noise_lr); self.hazard_lr=float(hazard_lr)
        self.recurrence_threshold=float(recurrence_threshold); self.novelty_threshold=float(novelty_threshold)
        self.hazard_discount_rec=float(hazard_discount_rec); self.hazard_discount_nov=float(hazard_discount_nov)
        self.min_sigma=float(min_sigma)
        self.sigma2=max(self.min_sigma**2,(self.stay_tolerance/2)**2)
        self.hazard=0.0; self.pending_score=0.0
    def _reset_adaptive_pending(self):
        super()._reset_pending(); self.pending_score=0.0
    def _set_candidate(self,kind,pid):
        if self.pending_kind!=kind or self.pending_id!=pid:
            self.pending_kind=kind; self.pending_id=pid; self.pending_count=0; self.pending_sum.fill(0.0); self.pending_score=0.0
    def step(self,x,y=None):
        self.t+=1; x=np.asarray(x,float)
        if x.shape!=(self.dim,): raise ValueError((x.shape,self.dim))
        for p in self.prototypes: p.utility*=self.decay
        revision=reactivated=discovered=unresolved=budget_pressure=False
        if not self.prototypes:
            self.current_id=self._append(x,self.t); discovered=True; revision=True
        d=self.distances(x); m=self._memberships(d)
        nearest=int(np.argmin(d)) if len(d) else None
        nd=float(d[nearest]) if nearest is not None else float('inf')
        pred=self._prediction(m)
        cur=self.current_id; curd=float(d[cur]) if cur is not None and cur<len(d) else float('inf')
        sigma=max(self.min_sigma,math.sqrt(self.sigma2))
        denom=max(1e-6,self.recurrence_tolerance-self.stay_tolerance)
        surprise=float(np.clip((curd-self.stay_tolerance)/denom,0.0,1.0))
        self.hazard=(1-self.hazard_lr)*self.hazard+self.hazard_lr*surprise
        if cur is not None and curd<=self.stay_tolerance:
            self._reset_adaptive_pending()
            capped=min(curd,self.stay_tolerance)
            self.sigma2=(1-self.noise_lr)*self.sigma2+self.noise_lr*capped*capped
            p=self.prototypes[cur]
            if curd<=self.update_tolerance:
                conf=max(0.0,1.0-curd/max(self.update_tolerance,1e-6))
                eta=min(self.centroid_lr_cap,1.0/max(2.0,p.count+1.0))*(0.5+0.5*conf)
                p.centroid=(1-eta)*p.centroid+eta*x; p.count+=1.0
        else:
            cand=None
            if nearest is not None and nearest!=cur and nd<=self.recurrence_tolerance: cand=nearest
            if cand is not None:
                self._set_candidate('reactivate',cand); self.pending_count+=1; self.pending_sum+=x
                llr=max(0.0,(curd*curd-nd*nd)/(2*sigma*sigma)); llr=min(llr,6.0)
                self.pending_score=self.evidence_decay*self.pending_score+llr
                theta=max(0.55,self.recurrence_threshold*(1-self.hazard_discount_rec*self.hazard))
                if self.pending_score>=theta:
                    self.current_id=cand; revision=True; reactivated=True
                    self.revision_count+=1; self.reactivation_count+=1; self._reset_adaptive_pending()
            else:
                self._set_candidate('novel',None); self.pending_count+=1; self.pending_sum+=x
                all_far=max(0.0,nd-self.recurrence_tolerance); cur_bad=max(0.0,curd-self.stay_tolerance)
                novelty_llr=min(max(0.0,(all_far+0.5*cur_bad)/sigma),4.0)
                self.pending_score=self.evidence_decay*self.pending_score+novelty_llr
                theta=max(1.75,self.novelty_threshold*(1-self.hazard_discount_nov*self.hazard))
                if self.pending_score>=theta:
                    meanx=self.pending_sum/max(1,self.pending_count)
                    if len(self.prototypes)<self.budget or self._certified_recompress():
                        self.current_id=self._append(meanx,self.t); revision=True; discovered=True
                        self.revision_count+=1; self._reset_adaptive_pending()
                    else:
                        unresolved=True; budget_pressure=True; self.unresolved_count+=1; self.budget_pressure_count+=1
        if y is not None and self.current_id is not None:
            p=self.prototypes[self.current_id]; yy=float(y); p.successes+=yy; p.failures+=1.0-yy; p.utility+=1.0; p.last_used=self.t
        if revision or discovered:
            d=self.distances(x); m=self._memberships(d); nearest=int(np.argmin(d)) if len(d) else None; nd=float(d[nearest]) if nearest is not None else float('inf')
        active=tuple(int(i) for i in np.where(m>=self.alpha_cut)[0]) if len(m) else tuple()
        if not active and nearest is not None: active=(nearest,)
        return ContinualRead(pred,m,active,self.current_id,nearest,nd,revision,reactivated,discovered,unresolved,budget_pressure,len(self.prototypes),len(d))


@dataclass
class TensorCell:
    T: np.ndarray
    u: np.ndarray
    theta: float = 0.0
    proj2: float = 0.01
    obs: int = 0
    coherence: float = 0.0
    strength: float = 0.0
    ready: bool = False

class TensorPlasticityReasoner(AdaptiveEvidenceReasoner):
    """v0.9.2 experimental tensor-directed parametric plasticity.

    A bounded 3-way residual tensor (feature x timescale x moment-channel) is
    maintained per regime prototype.  A rank-1 direction extracted from its
    mode-1 unfolding controls a one-dimensional local predictive correction.
    No tensor state is allowed to allocate a new regime by itself.
    """
    def __init__(self,*args,tensor_rates=(0.22,0.06,0.015),tensor_min_obs=20,
                 tensor_coherence=0.48,tensor_strength=0.006,
                 tensor_lr=0.055,tensor_l2=0.002,theta_cap=3.0,
                 tensor_warmup=12,**kwargs):
        self.tensor_rates=np.asarray(tensor_rates,float)
        self.tensor_min_obs=int(tensor_min_obs)
        self.tensor_coherence=float(tensor_coherence)
        self.tensor_strength=float(tensor_strength)
        self.tensor_lr=float(tensor_lr); self.tensor_l2=float(tensor_l2); self.theta_cap=float(theta_cap)
        self.tensor_warmup=int(tensor_warmup)
        self.tensor_cells=[]
        self.tensor_updates=0; self.tensor_ready_events=0
        super().__init__(*args,**kwargs)

    def _append(self,x,t):
        idx=super()._append(x,t)
        d=self.dim; s=len(self.tensor_rates)
        u=np.zeros(d,float); u[0]=1.0
        self.tensor_cells.append(TensorCell(np.zeros((d,s,2),float),u))
        return idx

    def _certified_recompress(self):
        # Mirror parent merge, but merge tensor state conservatively if it occurs.
        n0=len(self.prototypes)
        if n0<2:return False
        # reproduce parent's logic so we know which index merged
        if self.t-self._last_recompress_check<self.recompress_interval:return False
        self._last_recompress_check=self.t
        P=self._matrix(); D=np.sqrt(np.mean((P[:,None,:]-P[None,:,:])**2,axis=2)); D[np.tril_indices(n0)]=np.inf
        k=int(np.argmin(D)); i,j=np.unravel_index(k,D.shape); dist=float(D[i,j])
        if not np.isfinite(dist) or dist>self.redundancy_tolerance:return False
        a,b=self.prototypes[i],self.prototypes[j]; wa,wb=a.count,b.count
        c=(wa*a.centroid+wb*b.centroid)/max(EPS,wa+wb)
        if max(float(np.sqrt(np.mean((c-a.centroid)**2))),float(np.sqrt(np.mean((c-b.centroid)**2))))>self.redundancy_tolerance:return False
        a.centroid=c; a.count=wa+wb; a.successes+=b.successes; a.failures+=b.failures; a.utility+=b.utility; a.last_used=max(a.last_used,b.last_used)
        # count-weighted tensor merge; theta reset to avoid unjustified mixed correction
        ca,cb=self.tensor_cells[i],self.tensor_cells[j]
        w=max(EPS,wa+wb); ca.T=(wa*ca.T+wb*cb.T)/w; ca.obs += cb.obs; ca.theta=0.0; ca.ready=False
        del self.prototypes[j]; del self.tensor_cells[j]
        if self.current_id is not None:
            if self.current_id==j:self.current_id=i
            elif self.current_id>j:self.current_id-=1
        self.recompression_count+=1
        return True

    def _refresh_direction(self,cell):
        M=cell.T.reshape(self.dim,-1)
        fro=float(np.linalg.norm(M))
        if fro<1e-12:
            cell.coherence=0.0; cell.strength=0.0; cell.ready=False; return
        # Two warm-started power iterations on M M^T approximate the leading
        # feature-mode singular vector in O(d * scales * channels), avoiding SVD.
        u=cell.u.copy()
        if np.linalg.norm(u)<1e-12:
            u=np.ones(self.dim,float)/math.sqrt(self.dim)
        for _ in range(2):
            v=M.T@u; un=M@v; nrm=float(np.linalg.norm(un))
            if nrm<1e-12: break
            u=un/nrm
        if np.dot(u,cell.u)<0:u=-u
        cell.u=u
        s1=float(np.linalg.norm(M.T@u))
        cell.strength=s1
        cell.coherence=float((s1*s1)/(fro*fro+1e-12))
        was=cell.ready
        cell.ready=(cell.obs>=self.tensor_min_obs and cell.coherence>=self.tensor_coherence and cell.strength>=self.tensor_strength)
        if cell.ready and not was:self.tensor_ready_events+=1

    def _prototype_tensor_prediction(self,i,x):
        p=self.prototypes[i]; base=p.mean; c=self.tensor_cells[i]
        if not c.ready:return base,0.0
        z=np.asarray(x,float)-p.centroid
        q=float(np.dot(c.u,z)); scale=max(0.05,math.sqrt(c.proj2)); qn=float(np.clip(q/scale,-3,3))
        return sigmoid(logit(base)+c.theta*qn),qn

    def _prediction_tensor(self,m,x):
        if not self.prototypes:return 0.5,{}
        if len(m)==0:
            i=self.current_id or 0; p,q=self._prototype_tensor_prediction(i,x); return p,{i:(p,q)}
        active=np.where(m>=self.alpha_cut)[0]
        if len(active)==0:active=np.array([int(np.argmax(m))])
        w=m[active];w=w/max(EPS,float(w.sum()))
        cache={}; out=0.0
        for wi,i in zip(w,active):
            i=int(i); pi,qi=self._prototype_tensor_prediction(i,x); cache[i]=(pi,qi); out+=float(wi)*pi
        return float(out),cache

    def step(self,x,y=None):
        # We copy AdaptiveEvidenceReasoner's structural control but use tensor-corrected pre-outcome prediction.
        self.t+=1; x=np.asarray(x,float)
        if x.shape!=(self.dim,):raise ValueError((x.shape,self.dim))
        for p in self.prototypes:p.utility*=self.decay
        revision=reactivated=discovered=unresolved=budget_pressure=False
        if not self.prototypes:
            self.current_id=self._append(x,self.t); discovered=True; revision=True
        d=self.distances(x); m=self._memberships(d); nearest=int(np.argmin(d)) if len(d) else None; nd=float(d[nearest]) if nearest is not None else float('inf')
        pred,pcache=self._prediction_tensor(m,x)
        cur=self.current_id; curd=float(d[cur]) if cur is not None and cur<len(d) else float('inf')
        sigma=max(self.min_sigma,math.sqrt(self.sigma2)); denom=max(1e-6,self.recurrence_tolerance-self.stay_tolerance)
        surprise=float(np.clip((curd-self.stay_tolerance)/denom,0.0,1.0)); self.hazard=(1-self.hazard_lr)*self.hazard+self.hazard_lr*surprise
        if cur is not None and curd<=self.stay_tolerance:
            self._reset_adaptive_pending(); capped=min(curd,self.stay_tolerance); self.sigma2=(1-self.noise_lr)*self.sigma2+self.noise_lr*capped*capped
            p=self.prototypes[cur]
            if curd<=self.update_tolerance:
                conf=max(0.0,1.0-curd/max(self.update_tolerance,1e-6)); eta=min(self.centroid_lr_cap,1.0/max(2.0,p.count+1.0))*(0.5+0.5*conf)
                p.centroid=(1-eta)*p.centroid+eta*x; p.count+=1.0
        else:
            cand=None
            if nearest is not None and nearest!=cur and nd<=self.recurrence_tolerance:cand=nearest
            if cand is not None:
                self._set_candidate('reactivate',cand); self.pending_count+=1; self.pending_sum+=x
                llr=max(0.0,(curd*curd-nd*nd)/(2*sigma*sigma));llr=min(llr,6.0);self.pending_score=self.evidence_decay*self.pending_score+llr
                theta=max(0.55,self.recurrence_threshold*(1-self.hazard_discount_rec*self.hazard))
                if self.pending_score>=theta:
                    self.current_id=cand;revision=True;reactivated=True;self.revision_count+=1;self.reactivation_count+=1;self._reset_adaptive_pending()
            else:
                self._set_candidate('novel',None);self.pending_count+=1;self.pending_sum+=x
                all_far=max(0.0,nd-self.recurrence_tolerance);cur_bad=max(0.0,curd-self.stay_tolerance)
                novelty_llr=min(max(0.0,(all_far+0.5*cur_bad)/sigma),4.0);self.pending_score=self.evidence_decay*self.pending_score+novelty_llr
                theta=max(1.75,self.novelty_threshold*(1-self.hazard_discount_nov*self.hazard))
                if self.pending_score>=theta:
                    meanx=self.pending_sum/max(1,self.pending_count)
                    if len(self.prototypes)<self.budget or self._certified_recompress():
                        self.current_id=self._append(meanx,self.t);revision=True;discovered=True;self.revision_count+=1;self._reset_adaptive_pending()
                    else:
                        unresolved=True;budget_pressure=True;self.unresolved_count+=1;self.budget_pressure_count+=1
        # Outcome update: structural prototype + its bounded tensor state.
        if y is not None and self.current_id is not None:
            i=self.current_id; p=self.prototypes[i]; c=self.tensor_cells[i]; yy=float(y)
            # tensor residual is against uncorrected prototype mean; this detects internal misspecification rather than head's own residual.
            base=p.mean; e=yy-base; z=x-p.centroid
            for si,rate in enumerate(self.tensor_rates):
                c.T[:,si,0]=(1-rate)*c.T[:,si,0]+rate*(e*z)
                c.T[:,si,1]=(1-rate)*c.T[:,si,1]+rate*(e*z*np.abs(z))
            c.obs+=1; self.tensor_updates+=1
            if c.obs>=self.tensor_warmup and (c.obs%12==0 or not c.ready):self._refresh_direction(c)
            # update projection scale and local scalar correction only when structured tensor evidence is present
            q=float(np.dot(c.u,z)); c.proj2=0.96*c.proj2+0.04*q*q
            if c.ready:
                scale=max(0.05,math.sqrt(c.proj2)); qn=float(np.clip(q/scale,-3,3)); ph=sigmoid(logit(base)+c.theta*qn)
                grad=(ph-yy)*qn + self.tensor_l2*c.theta
                c.theta=float(np.clip(c.theta-self.tensor_lr*grad,-self.theta_cap,self.theta_cap))
            p.successes+=yy;p.failures+=1-yy;p.utility+=1.0;p.last_used=self.t
        if revision or discovered:
            d=self.distances(x);m=self._memberships(d);nearest=int(np.argmin(d)) if len(d) else None;nd=float(d[nearest]) if nearest is not None else float('inf')
        active=tuple(int(i) for i in np.where(m>=self.alpha_cut)[0]) if len(m) else tuple()
        if not active and nearest is not None:active=(nearest,)
        return ContinualRead(pred,m,active,self.current_id,nearest,nd,revision,reactivated,discovered,unresolved,budget_pressure,len(self.prototypes),len(d))

