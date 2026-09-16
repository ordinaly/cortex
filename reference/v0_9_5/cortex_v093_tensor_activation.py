"""Cortex v0.9.3-alpha — Burst-Gated Tensor Plasticity.

A compute-optimized variant of v0.9.2.  Tensor directions are learned in short
bursts and frozen between bursts.  Only scalar loss EMAs and the 1-D tensor head
remain active every observation.
"""
import math
from dataclasses import dataclass
import numpy as np
from cortex_v092_tensor_plasticity import (
    AdaptiveEvidenceReasoner, TensorPlasticityReasoner, TensorCell,
    EPS, ContinualRead
)

def _clip(v, lo, hi):
    v=float(v)
    return lo if v < lo else hi if v > hi else v

def _logit(p):
    p=_clip(p,1e-4,1-1e-4)
    return math.log(p/(1-p))

def _sigmoid(z):
    z=_clip(z,-25.0,25.0)
    return 1.0/(1.0+math.exp(-z))

@dataclass
class BurstTensorCell(TensorCell):
    base_loss: float = 0.25
    corr_loss: float = 0.25
    burst_left: int = 0
    since_burst: int = 0
    tensor_steps: int = 0
    refreshes: int = 0
    wakes: int = 0
    sleeps: int = 0

class BurstTensorPlasticityReasoner(TensorPlasticityReasoner):
    def __init__(self,*args,
                 loss_lr=0.05,
                 initial_burst=36,
                 post_ready_burst=18,
                 refresh_interval=72,
                 refresh_burst=8,
                 degradation_ratio=0.98,
                 degradation_patience=4,
                 refresh_every_active=12,
                 **kwargs):
        self.loss_lr=float(loss_lr)
        self.initial_burst=int(initial_burst)
        self.post_ready_burst=int(post_ready_burst)
        self.refresh_interval=int(refresh_interval)
        self.refresh_burst=int(refresh_burst)
        self.degradation_ratio=float(degradation_ratio)
        self.degradation_patience=int(degradation_patience)
        self.refresh_every_active=int(refresh_every_active)
        self.tensor_awake_steps=0; self.tensor_sleep_steps=0
        self.tensor_refreshes=0; self.tensor_wakes=0; self.tensor_sleeps=0
        self._bad_counts=[]
        super().__init__(*args,**kwargs)

    def _append(self,x,t):
        idx=AdaptiveEvidenceReasoner._append(self,x,t)
        d=self.dim;s=len(self.tensor_rates)
        u=np.zeros(d,float);u[0]=1.0
        cell=BurstTensorCell(np.zeros((d,s,2),float),u,burst_left=self.initial_burst)
        self.tensor_cells.append(cell);self._bad_counts.append(0)
        return idx

    def _prototype_tensor_prediction(self,i,x):
        p=self.prototypes[i]; base=p.mean; c=self.tensor_cells[i]
        if not c.ready:return base,0.0
        z=np.asarray(x,float)-p.centroid
        q=float(np.dot(c.u,z)); scale=max(0.05,math.sqrt(c.proj2)); qn=_clip(q/scale,-3.0,3.0)
        return _sigmoid(_logit(base)+c.theta*qn),qn

    def _certified_recompress(self):
        n0=len(self.prototypes)
        if n0<2:return False
        if self.t-self._last_recompress_check<self.recompress_interval:return False
        self._last_recompress_check=self.t
        P=self._matrix();D=np.sqrt(np.mean((P[:,None,:]-P[None,:,:])**2,axis=2));D[np.tril_indices(n0)]=np.inf
        k=int(np.argmin(D));i,j=np.unravel_index(k,D.shape);dist=float(D[i,j])
        if not np.isfinite(dist) or dist>self.redundancy_tolerance:return False
        a,b=self.prototypes[i],self.prototypes[j];wa,wb=a.count,b.count
        c=(wa*a.centroid+wb*b.centroid)/max(EPS,wa+wb)
        if max(float(np.sqrt(np.mean((c-a.centroid)**2))),float(np.sqrt(np.mean((c-b.centroid)**2))))>self.redundancy_tolerance:return False
        a.centroid=c;a.count=wa+wb;a.successes+=b.successes;a.failures+=b.failures;a.utility+=b.utility;a.last_used=max(a.last_used,b.last_used)
        ca,cb=self.tensor_cells[i],self.tensor_cells[j];w=max(EPS,wa+wb)
        ca.T=(wa*ca.T+wb*cb.T)/w;ca.obs+=cb.obs;ca.theta=0.0;ca.ready=False;ca.burst_left=self.initial_burst;ca.since_burst=0
        ca.base_loss=(wa*ca.base_loss+wb*cb.base_loss)/w;ca.corr_loss=(wa*ca.corr_loss+wb*cb.corr_loss)/w
        del self.prototypes[j];del self.tensor_cells[j];del self._bad_counts[j]
        if self.current_id is not None:
            if self.current_id==j:self.current_id=i
            elif self.current_id>j:self.current_id-=1
        self.recompression_count+=1
        return True

    def _schedule_gate(self,i,c):
        # While no direction exists, allow a finite initial training burst. If it
        # was insufficient, sparse refresh bursts continue until ready.
        if c.burst_left>0:return
        c.since_burst+=1
        if not c.ready:
            if c.since_burst>=max(12,self.refresh_interval//2):
                c.burst_left=self.refresh_burst;c.since_burst=0;c.wakes+=1;self.tensor_wakes+=1
            return
        ratio=c.corr_loss/max(1e-9,c.base_loss)
        bad=(ratio>=self.degradation_ratio)
        self._bad_counts[i]=self._bad_counts[i]+1 if bad else 0
        if self._bad_counts[i]>=self.degradation_patience or c.since_burst>=self.refresh_interval:
            c.burst_left=self.refresh_burst;c.since_burst=0;c.wakes+=1;self.tensor_wakes+=1;self._bad_counts[i]=0

    def step(self,x,y=None):
        self.t+=1;x=np.asarray(x,float)
        if x.shape!=(self.dim,):raise ValueError((x.shape,self.dim))
        for p in self.prototypes:p.utility*=self.decay
        revision=reactivated=discovered=unresolved=budget_pressure=False
        if not self.prototypes:
            self.current_id=self._append(x,self.t);discovered=True;revision=True
        d=self.distances(x);m=self._memberships(d);nearest=int(np.argmin(d)) if len(d) else None;nd=float(d[nearest]) if nearest is not None else float('inf')
        pred,pcache=self._prediction_tensor(m,x)
        cur=self.current_id;curd=float(d[cur]) if cur is not None and cur<len(d) else float('inf')
        sigma=max(self.min_sigma,math.sqrt(self.sigma2));denom=max(1e-6,self.recurrence_tolerance-self.stay_tolerance)
        surprise=_clip((curd-self.stay_tolerance)/denom,0.0,1.0);self.hazard=(1-self.hazard_lr)*self.hazard+self.hazard_lr*surprise
        if cur is not None and curd<=self.stay_tolerance:
            self._reset_adaptive_pending();capped=min(curd,self.stay_tolerance);self.sigma2=(1-self.noise_lr)*self.sigma2+self.noise_lr*capped*capped
            p=self.prototypes[cur]
            if curd<=self.update_tolerance:
                conf=max(0.0,1.0-curd/max(self.update_tolerance,1e-6));eta=min(self.centroid_lr_cap,1.0/max(2.0,p.count+1.0))*(0.5+0.5*conf)
                p.centroid=(1-eta)*p.centroid+eta*x;p.count+=1.0
        else:
            cand=None
            if nearest is not None and nearest!=cur and nd<=self.recurrence_tolerance:cand=nearest
            if cand is not None:
                self._set_candidate('reactivate',cand);self.pending_count+=1;self.pending_sum+=x
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

        if y is not None and self.current_id is not None:
            i=self.current_id;p=self.prototypes[i];c=self.tensor_cells[i];yy=float(y)
            base=p.mean;e=yy-base;z=x-p.centroid
            # Evaluate current correction cheaply for gating statistics.
            q=float(np.dot(c.u,z));scale=max(0.05,math.sqrt(c.proj2));qn=_clip(q/scale,-3.0,3.0)
            ph=_sigmoid(_logit(base)+c.theta*qn) if c.ready else base
            r=self.loss_lr;c.base_loss=(1-r)*c.base_loss+r*(yy-base)**2;c.corr_loss=(1-r)*c.corr_loss+r*(yy-ph)**2
            c.obs+=1;self._schedule_gate(i,c)

            if c.burst_left>0:
                was_ready=c.ready
                for si,rate in enumerate(self.tensor_rates):
                    c.T[:,si,0]=(1-rate)*c.T[:,si,0]+rate*(e*z)
                    c.T[:,si,1]=(1-rate)*c.T[:,si,1]+rate*(e*z*np.abs(z))
                c.tensor_steps+=1;self.tensor_updates+=1;self.tensor_awake_steps+=1;c.burst_left-=1
                if c.tensor_steps>=self.tensor_warmup and (c.tensor_steps%self.refresh_every_active==0 or not c.ready):
                    self._refresh_direction(c);c.refreshes+=1;self.tensor_refreshes+=1
                    if c.ready and not was_ready:
                        # Keep learning briefly after first valid direction.
                        c.burst_left=max(c.burst_left,self.post_ready_burst)
                if c.burst_left==0:c.sleeps+=1;self.tensor_sleeps+=1
            else:
                self.tensor_sleep_steps+=1

            # Cheap scalar head remains plastic at every observation.
            q=float(np.dot(c.u,z));c.proj2=0.96*c.proj2+0.04*q*q
            if c.ready:
                scale=max(0.05,math.sqrt(c.proj2));qn=_clip(q/scale,-3.0,3.0);ph=_sigmoid(_logit(base)+c.theta*qn)
                grad=(ph-yy)*qn+self.tensor_l2*c.theta
                c.theta=_clip(c.theta-self.tensor_lr*grad,-self.theta_cap,self.theta_cap)
            p.successes+=yy;p.failures+=1-yy;p.utility+=1.0;p.last_used=self.t

        if revision or discovered:
            d=self.distances(x);m=self._memberships(d);nearest=int(np.argmin(d)) if len(d) else None;nd=float(d[nearest]) if nearest is not None else float('inf')
        active=tuple(int(i) for i in np.where(m>=self.alpha_cut)[0]) if len(m) else tuple()
        if not active and nearest is not None:active=(nearest,)
        return ContinualRead(pred,m,active,self.current_id,nearest,nd,revision,reactivated,discovered,unresolved,budget_pressure,len(self.prototypes),len(d))
