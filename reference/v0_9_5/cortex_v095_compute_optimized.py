"""Cortex v0.9.5-alpha — Compute-Optimized Refinement Runtime.

Preserves v0.9.4-alpha's rank-triggered split/merge semantics while attacking
avoidable runtime work:
  1. cached centroid carrier for repeated distance queries;
  2. strided residual-curvature surveillance with decay-corrected EMA;
  3. event-directed split/merge bookkeeping instead of unconditional checks.

The active reasoning semantics remain inherited from v0.9.4-alpha.
"""
from __future__ import annotations

import numpy as np

from cortex_v094_rank_refinement import RankRefinementReasoner
from cortex_v09 import ContinualRead


class ComputeOptimizedRefinementReasoner(RankRefinementReasoner):
    def __init__(self, *args, curvature_stride: int = 2, **kwargs):
        self.curvature_stride = max(1, int(curvature_stride))
        self._centroid_cache = None
        self._lineage_active = False
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Carrier cache. In this architecture only the current prototype centroid
    # can drift without a structural edit. Appends / splits / merges change
    # length and invalidate the cache explicitly.
    # ------------------------------------------------------------------
    def _invalidate_carrier(self):
        self._centroid_cache = None

    def _matrix(self):
        n = len(self.prototypes)
        if n == 0:
            return np.zeros((0, self.dim), float)
        if self._centroid_cache is None or self._centroid_cache.shape != (n, self.dim):
            self._centroid_cache = np.stack([p.centroid for p in self.prototypes], axis=0).astype(float, copy=True)
        else:
            i = self.current_id
            if i is not None and 0 <= i < n:
                self._centroid_cache[i, :] = self.prototypes[i].centroid
        return self._centroid_cache

    def _append(self, x, t):
        idx = super()._append(x, t)
        self._invalidate_carrier()
        return idx

    def _certified_recompress(self):
        ok = super()._certified_recompress()
        if ok:
            self._invalidate_carrier()
        return ok

    def _promote_split(self, i, cand, x_now):
        ok = super()._promote_split(i, cand, x_now)
        if ok:
            self._invalidate_carrier()
            self._lineage_active = True
        return ok

    def _merge_pair(self, i, j):
        super()._merge_pair(i, j)
        self._invalidate_carrier()
        self._lineage_active = any(getattr(p, "lineage_id", None) is not None for p in self.prototypes)

    # ------------------------------------------------------------------
    # Curvature subsampling. The EMA coefficient is adjusted so one sampled
    # update represents `stride` ordinary decay steps. This preserves the
    # monitor's approximate time constant while avoiding most outer products.
    # ------------------------------------------------------------------
    def _update_curvature(self, i, x, y, prediction, centroid_before):
        c = self.tensor_cells[i]
        z = np.asarray(x, float) - centroid_before
        e = float(y) - float(prediction)
        s = self.curvature_stride
        r = 1.0 - (1.0 - self.curvature_lr) ** s
        c.curvature = (1.0 - r) * c.curvature + r * e * np.outer(z, z)
        # Count represented awake observations, not only physical outer products,
        # so the existing minimum-support contract retains its meaning.
        c.curvature_obs += s
        self.curvature_updates += 1

    def step(self, x, y=None):
        x = np.asarray(x, float)
        pre_current = self.current_id
        pre_centroid = None
        pre_tensor_steps = None
        if pre_current is not None and 0 <= pre_current < len(self.prototypes):
            pre_centroid = self.prototypes[pre_current].centroid.copy()
            pre_tensor_steps = getattr(self.tensor_cells[pre_current], "tensor_steps", 0)

        read = super(RankRefinementReasoner, self).step(x, y)  # call frozen v0.9.3 directly
        structural_edit = False

        if y is not None and self.current_id is not None:
            i = self.current_id
            pcur = self.prototypes[i]
            if getattr(pcur, "lineage_id", None) is not None:
                rr = self.merge_recent_lr
                pcur.recent_mean = (1.0 - rr) * float(getattr(pcur, "recent_mean", pcur.mean)) + rr * float(y)
                pcur.recent_obs = int(getattr(pcur, "recent_obs", 0)) + 1
                self._lineage_active = True

            c = self.tensor_cells[i]
            tensor_advanced = getattr(c, "tensor_steps", 0) > (pre_tensor_steps or 0)
            curvature_updated = False
            if (
                pre_current == i
                and pre_centroid is not None
                and tensor_advanced
                and not read.revision
                and (getattr(c, "tensor_steps", 0) % self.curvature_stride == 0)
            ):
                self._update_curvature(i, x, float(y), float(read.prediction), pre_centroid)
                curvature_updated = True

            # Launch checks can only become informative when the curvature monitor
            # has just advanced. Candidate validation, once launched, remains fully
            # prospective and is updated on every applicable observation.
            if curvature_updated:
                self._maybe_launch_split(i)

            pkey = id(self.prototypes[i]) if i < len(self.prototypes) else None
            action = None
            if pkey in self.split_candidates:
                action = self._shadow_update_candidate(i, x, float(y), float(read.prediction))
            if action == "promote" and pkey in self.split_candidates:
                cand = self.split_candidates[pkey]
                structural_edit = self._promote_split(i, cand, x)
            elif action == "reject" and pkey in self.split_candidates:
                self.split_rejections += 1
                self.split_candidates.pop(pkey, None)
                self.split_cooldown_until[pkey] = self.t + self.split_cooldown

        if self._lineage_active and self._maybe_merge_lineage():
            structural_edit = True

        # Keep carrier row synchronized after ordinary centroid learning.
        if self._centroid_cache is not None and self.current_id is not None and self.current_id < len(self.prototypes):
            self._centroid_cache[self.current_id, :] = self.prototypes[self.current_id].centroid

        if not structural_edit:
            return read

        d = self.distances(x)
        m = self._memberships(d)
        nearest = int(np.argmin(d)) if len(d) else None
        nd = float(d[nearest]) if nearest is not None else float("inf")
        active = tuple(int(k) for k in np.where(m >= self.alpha_cut)[0]) if len(m) else tuple()
        if not active and nearest is not None:
            active = (nearest,)
        return ContinualRead(
            float(read.prediction), m, active, self.current_id, nearest, nd,
            True, read.reactivated, read.discovered, read.unresolved,
            read.budget_pressure, len(self.prototypes), len(d),
        )
