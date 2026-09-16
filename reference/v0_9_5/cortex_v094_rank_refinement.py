"""Cortex v0.9.4-alpha — Rank-Triggered Concept Refinement.

Experimental Cortex-only extension of v0.9.3-alpha.

The v0.9.3 structural controller and burst-gated rank-1 tensor head are preserved.
v0.9.4 adds a *gated residual-curvature tensor* per active prototype.  Persistent
multi-directional residual structure may launch a shadow split hypothesis.  A split
is promoted only after a two-stage prospective (prequential) validation period and
only if the predictive gain pays a complexity penalty.  Split children may later be
reconciled by a conservative merge certificate.

This is an executable research prototype, not a convergence proof.
"""
from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from cortex_v093_tensor_activation import (
    BurstTensorPlasticityReasoner,
    BurstTensorCell,
    _clip,
    _logit,
    _sigmoid,
)
from cortex_v09 import RegimePrototype, ContinualRead, EPS, rms


@dataclass
class DirectionShadow:
    direction: np.ndarray
    bias_neg: float = 0.0
    bias_pos: float = 0.0
    parent_loss: float = 0.0
    child_loss: float = 0.0
    n: int = 0

    def gain(self) -> float:
        return self.parent_loss - self.child_loss

    def mean_gain(self) -> float:
        return self.gain() / max(1, self.n)


@dataclass
class SplitCandidate:
    parent_obj_id: int
    started_t: int
    directions: List[DirectionShadow]
    selection_target: int = 72
    validation_target: int = 128
    max_validation: int = 224
    phase: str = "select"
    selected: int | None = None
    validation_n: int = 0
    validation_parent_loss: float = 0.0
    validation_child_loss: float = 0.0
    bias_neg: float = 0.0
    bias_pos: float = 0.0
    neg_n: int = 0
    pos_n: int = 0
    neg_success: float = 0.0
    pos_success: float = 0.0
    neg_sum: np.ndarray | None = None
    pos_sum: np.ndarray | None = None

    def validation_gain(self) -> float:
        return self.validation_parent_loss - self.validation_child_loss

    def validation_mean_gain(self) -> float:
        return self.validation_gain() / max(1, self.validation_n)


class RankRefinementReasoner(BurstTensorPlasticityReasoner):
    """v0.9.4-alpha: conservative split/merge plasticity.

    The residual-curvature matrix Q_i = EMA[(y-p_hat) z z^T] is updated only on
    timesteps where v0.9.3 already has the tensor awake.  This makes split
    surveillance conditional on an already-triggered expensive-computation window.

    A rank certificate may *propose* a split.  It cannot directly edit structure.
    Candidate split directions are selected on one prospective segment and validated
    on a disjoint later segment.  Promotion requires both branches to receive support
    and future Brier improvement to exceed a complexity penalty.
    """

    def __init__(
        self,
        *args,
        curvature_lr: float = 0.055,
        rank_ratio_threshold: float = 0.25,
        rank_strength_threshold: float = 0.0012,
        rank_patience: int = 2,
        split_min_parent_obs: int = 48,
        split_selection_obs: int = 72,
        split_validation_obs: int = 128,
        split_max_validation: int = 224,
        split_min_branch: int = 24,
        split_mean_gain: float = 0.0100,
        split_complexity_penalty: float = 0.080,
        split_accept_margin: float = 0.25,
        shadow_bias_lr: float = 0.045,
        shadow_bias_cap: float = 2.5,
        split_cooldown: int = 288,
        recursive_split_penalty: float = 1.6,
        max_refinement_depth: int = 1,
        merge_interval: int = 64,
        merge_min_age: int = 192,
        merge_pred_tolerance: float = 0.090,
        merge_dist_tolerance: float = 0.16,
        merge_min_count: float = 24.0,
        merge_recent_lr: float = 0.035,
        merge_recent_min_obs: int = 28,
        merge_patience: int = 4,
        **kwargs,
    ):
        self.curvature_lr = float(curvature_lr)
        self.rank_ratio_threshold = float(rank_ratio_threshold)
        self.rank_strength_threshold = float(rank_strength_threshold)
        self.rank_patience = int(rank_patience)
        self.split_min_parent_obs = int(split_min_parent_obs)
        self.split_selection_obs = int(split_selection_obs)
        self.split_validation_obs = int(split_validation_obs)
        self.split_max_validation = int(split_max_validation)
        self.split_min_branch = int(split_min_branch)
        self.split_mean_gain = float(split_mean_gain)
        self.split_complexity_penalty = float(split_complexity_penalty)
        self.split_accept_margin = float(split_accept_margin)
        self.shadow_bias_lr = float(shadow_bias_lr)
        self.shadow_bias_cap = float(shadow_bias_cap)
        self.split_cooldown = int(split_cooldown)
        self.recursive_split_penalty = float(recursive_split_penalty)
        self.max_refinement_depth = int(max_refinement_depth)
        self.merge_interval = int(merge_interval)
        self.merge_min_age = int(merge_min_age)
        self.merge_pred_tolerance = float(merge_pred_tolerance)
        self.merge_dist_tolerance = float(merge_dist_tolerance)
        self.merge_min_count = float(merge_min_count)
        self.merge_recent_lr = float(merge_recent_lr)
        self.merge_recent_min_obs = int(merge_recent_min_obs)
        self.merge_patience = int(merge_patience)

        self.split_candidates: Dict[int, SplitCandidate] = {}
        self.split_cooldown_until: Dict[int, int] = {}
        self.split_proposals = 0
        self.split_rejections = 0
        self.split_promotions = 0
        self.split_budget_blocks = 0
        self.merge_promotions = 0
        self.rank_checks = 0
        self.curvature_updates = 0
        self._next_lineage = 1
        self._last_merge_check = -10**9
        self._merge_evidence: Dict[int, int] = {}
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Per-prototype auxiliary state is attached to TensorCell / Prototype
    # dynamically so index changes from ordinary recurrence do not require a
    # parallel index bookkeeping structure.
    # ------------------------------------------------------------------
    def _append(self, x, t):
        idx = super()._append(x, t)
        self._ensure_refinement_state(idx)
        return idx

    def _ensure_refinement_state(self, i: int):
        if i < 0 or i >= len(self.tensor_cells):
            return
        c = self.tensor_cells[i]
        if not hasattr(c, "curvature"):
            c.curvature = np.zeros((self.dim, self.dim), float)
            c.curvature_obs = 0
            c.rank_ratio2 = 0.0
            c.rank_strength1 = 0.0
            c.rank_strength2 = 0.0
            c.rank_persist = 0
            c.last_rank_refresh_seen = -1
            c.rank_dirs = []
        p = self.prototypes[i]
        if not hasattr(p, "lineage_id"):
            p.lineage_id = None
            p.split_created_t = None
            p.recent_mean = p.mean
            p.recent_obs = 0
            p.refinement_depth = 0

    def _purge_dead_candidates(self):
        alive = {id(p) for p in self.prototypes}
        for k in list(self.split_candidates):
            if k not in alive:
                del self.split_candidates[k]
        for k in list(self.split_cooldown_until):
            if k not in alive:
                del self.split_cooldown_until[k]

    def _certified_recompress(self):
        ok = super()._certified_recompress()
        if ok:
            self._purge_dead_candidates()
            for i in range(len(self.prototypes)):
                self._ensure_refinement_state(i)
        return ok

    # ------------------------------------------------------------------
    # Residual-curvature tensor and rank certificate
    # ------------------------------------------------------------------
    def _update_curvature(self, i: int, x: np.ndarray, y: float, prediction: float, centroid_before: np.ndarray):
        self._ensure_refinement_state(i)
        c = self.tensor_cells[i]
        z = np.asarray(x, float) - centroid_before
        e = float(y) - float(prediction)
        r = self.curvature_lr
        # Symmetric signed residual second moment.  The sign matters: a purely
        # high-variance direction without systematic predictive miss does not
        # accumulate a coherent curvature tensor.
        c.curvature = (1.0 - r) * c.curvature + r * e * np.outer(z, z)
        c.curvature_obs += 1
        self.curvature_updates += 1

    @staticmethod
    def _canonicalize_direction(u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, float).copy()
        n = float(np.linalg.norm(u))
        if n <= 1e-12:
            return u
        u /= n
        j = int(np.argmax(np.abs(u)))
        if u[j] < 0:
            u = -u
        return u

    def _rank_certificate(self, i: int):
        self._ensure_refinement_state(i)
        c = self.tensor_cells[i]
        if c.curvature_obs < self.split_min_parent_obs:
            return False
        Q = 0.5 * (c.curvature + c.curvature.T)
        vals, vecs = np.linalg.eigh(Q)
        order = np.argsort(np.abs(vals))[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        s1 = abs(float(vals[0])) if len(vals) else 0.0
        s2 = abs(float(vals[1])) if len(vals) > 1 else 0.0
        ratio = s2 / max(1e-12, s1)
        c.rank_strength1 = s1
        c.rank_strength2 = s2
        c.rank_ratio2 = ratio
        self.rank_checks += 1

        if s1 >= self.rank_strength_threshold and ratio >= self.rank_ratio_threshold:
            c.rank_persist += 1
        else:
            c.rank_persist = max(0, c.rank_persist - 1)

        # Candidate directions: top two curvature eigenvectors plus the two
        # canonical feature axes carrying most mass in their joint subspace.
        dirs: List[np.ndarray] = []
        for j in range(min(2, vecs.shape[1])):
            dirs.append(self._canonicalize_direction(vecs[:, j]))
        if vecs.shape[1] >= 2:
            participation = np.sqrt(vecs[:, 0] ** 2 + vecs[:, 1] ** 2)
            for axis in np.argsort(participation)[::-1][:2]:
                e = np.zeros(self.dim, float)
                e[int(axis)] = 1.0
                dirs.append(e)
        # Deduplicate nearly parallel choices.
        unique: List[np.ndarray] = []
        for u in dirs:
            if np.linalg.norm(u) <= 1e-12:
                continue
            if all(abs(float(np.dot(u, v))) < 0.96 for v in unique):
                unique.append(u)
        c.rank_dirs = unique
        return c.rank_persist >= self.rank_patience and len(unique) > 0

    def _maybe_launch_split(self, i: int):
        self._ensure_refinement_state(i)
        p = self.prototypes[i]
        key = id(p)
        # v0.9.4-alpha deliberately permits one structural refinement level.
        # Recursive concept trees are deferred until a stronger complexity
        # controller is validated; this prevents split cascades in the alpha.
        if int(getattr(p, "refinement_depth", 0)) >= self.max_refinement_depth:
            return
        if key in self.split_candidates:
            return
        if self.t < self.split_cooldown_until.get(key, -1):
            return
        c = self.tensor_cells[i]
        # Rank checks are tied to v0.9.3 tensor refreshes, not every observation.
        if c.refreshes == c.last_rank_refresh_seen:
            return
        c.last_rank_refresh_seen = c.refreshes
        if not self._rank_certificate(i):
            return
        if c.obs < self.split_min_parent_obs:
            return
        shadows = [DirectionShadow(u.copy()) for u in c.rank_dirs]
        self.split_candidates[key] = SplitCandidate(
            parent_obj_id=key,
            started_t=self.t,
            directions=shadows,
            selection_target=self.split_selection_obs,
            validation_target=self.split_validation_obs,
            max_validation=self.split_max_validation,
        )
        self.split_proposals += 1

    # ------------------------------------------------------------------
    # Prospective shadow validation
    # ------------------------------------------------------------------
    def _shadow_update_one(self, sh: DirectionShadow, x: np.ndarray, centroid: np.ndarray, y: float, parent_pred: float):
        pos = float(np.dot(sh.direction, x - centroid)) >= 0.0
        b = sh.bias_pos if pos else sh.bias_neg
        child_pred = _sigmoid(_logit(parent_pred) + b)
        sh.parent_loss += (y - parent_pred) ** 2
        sh.child_loss += (y - child_pred) ** 2
        sh.n += 1
        grad = child_pred - y
        if pos:
            sh.bias_pos = _clip(sh.bias_pos - self.shadow_bias_lr * grad, -self.shadow_bias_cap, self.shadow_bias_cap)
        else:
            sh.bias_neg = _clip(sh.bias_neg - self.shadow_bias_lr * grad, -self.shadow_bias_cap, self.shadow_bias_cap)

    def _shadow_update_candidate(self, i: int, x: np.ndarray, y: float, parent_pred: float):
        p = self.prototypes[i]
        key = id(p)
        cand = self.split_candidates.get(key)
        if cand is None:
            return None
        centroid = p.centroid.copy()

        if cand.phase == "select":
            for sh in cand.directions:
                self._shadow_update_one(sh, x, centroid, y, parent_pred)
            if min(sh.n for sh in cand.directions) >= cand.selection_target:
                # Direction selection uses only the selection segment.  Validation
                # losses and branch parameters start fresh afterwards.
                scores = [sh.mean_gain() for sh in cand.directions]
                cand.selected = int(np.argmax(scores))
                cand.phase = "validate"
                cand.validation_n = 0
                cand.validation_parent_loss = 0.0
                cand.validation_child_loss = 0.0
                cand.bias_neg = 0.0
                cand.bias_pos = 0.0
                cand.neg_n = cand.pos_n = 0
                cand.neg_success = cand.pos_success = 0.0
                cand.neg_sum = np.zeros(self.dim, float)
                cand.pos_sum = np.zeros(self.dim, float)
            return None

        sh = cand.directions[cand.selected]
        pos = float(np.dot(sh.direction, x - centroid)) >= 0.0
        b = cand.bias_pos if pos else cand.bias_neg
        child_pred = _sigmoid(_logit(parent_pred) + b)
        cand.validation_parent_loss += (y - parent_pred) ** 2
        cand.validation_child_loss += (y - child_pred) ** 2
        cand.validation_n += 1
        grad = child_pred - y
        if pos:
            cand.bias_pos = _clip(cand.bias_pos - self.shadow_bias_lr * grad, -self.shadow_bias_cap, self.shadow_bias_cap)
            cand.pos_n += 1
            cand.pos_success += y
            cand.pos_sum += x
        else:
            cand.bias_neg = _clip(cand.bias_neg - self.shadow_bias_lr * grad, -self.shadow_bias_cap, self.shadow_bias_cap)
            cand.neg_n += 1
            cand.neg_success += y
            cand.neg_sum += x

        if cand.validation_n < cand.validation_target:
            return None

        gain = cand.validation_gain()
        mean_gain = cand.validation_mean_gain()
        depth = int(getattr(p, "refinement_depth", 0))
        depth_factor = 1.0 + self.recursive_split_penalty * depth
        complexity_cost = depth_factor * self.split_complexity_penalty * math.log1p(cand.validation_n)
        evidence = gain - complexity_cost
        supported = cand.neg_n >= self.split_min_branch and cand.pos_n >= self.split_min_branch
        if supported and mean_gain >= depth_factor * self.split_mean_gain and evidence >= depth_factor * self.split_accept_margin:
            return "promote"
        if cand.validation_n >= cand.max_validation:
            return "reject"
        return None

    # ------------------------------------------------------------------
    # Structural split / merge operations
    # ------------------------------------------------------------------
    def _fresh_tensor_cell(self):
        d = self.dim
        s = len(self.tensor_rates)
        u = np.zeros(d, float)
        u[0] = 1.0
        c = BurstTensorCell(np.zeros((d, s, 2), float), u, burst_left=self.initial_burst)
        c.curvature = np.zeros((d, d), float)
        c.curvature_obs = 0
        c.rank_ratio2 = 0.0
        c.rank_strength1 = 0.0
        c.rank_strength2 = 0.0
        c.rank_persist = 0
        c.last_rank_refresh_seen = -1
        c.rank_dirs = []
        return c

    def _inherit_tensor_cell(self, parent_cell):
        # Structural refinement should not erase already-learned parametric
        # structure.  Children inherit the rank-1 correction but start with a
        # fresh split-surveillance tensor.
        c = copy.deepcopy(parent_cell)
        c.curvature = np.zeros((self.dim, self.dim), float)
        c.curvature_obs = 0
        c.rank_ratio2 = 0.0
        c.rank_strength1 = 0.0
        c.rank_strength2 = 0.0
        c.rank_persist = 0
        c.last_rank_refresh_seen = -1
        c.rank_dirs = []
        c.burst_left = max(c.burst_left, self.refresh_burst)
        c.since_burst = 0
        return c

    def _promote_split(self, i: int, cand: SplitCandidate, x_now: np.ndarray):
        if len(self.prototypes) >= self.budget:
            self.split_budget_blocks += 1
            self.split_rejections += 1
            key = id(self.prototypes[i])
            self.split_cooldown_until[key] = self.t + self.split_cooldown
            self.split_candidates.pop(key, None)
            return False
        if cand.neg_n < self.split_min_branch or cand.pos_n < self.split_min_branch:
            return False

        parent = self.prototypes[i]
        parent_tensor = self.tensor_cells[i]
        key = id(parent)
        u = cand.directions[cand.selected].direction
        neg_centroid = cand.neg_sum / max(1, cand.neg_n)
        pos_centroid = cand.pos_sum / max(1, cand.pos_n)
        # If shadow branches are geometrically indistinguishable, a structural
        # split would only encode an outcome lookup table and is rejected.
        if rms(neg_centroid, pos_centroid) < 0.5 * self.redundancy_tolerance:
            self.split_rejections += 1
            self.split_cooldown_until[key] = self.t + self.split_cooldown
            self.split_candidates.pop(key, None)
            return False

        prior = self.prior_strength / 2.0
        lineage = self._next_lineage
        self._next_lineage += 1
        neg = RegimePrototype(
            neg_centroid.copy(),
            float(cand.neg_n),
            prior + float(cand.neg_success),
            prior + float(cand.neg_n - cand.neg_success),
            max(1.0, parent.utility * cand.neg_n / max(1, cand.neg_n + cand.pos_n)),
            self.t,
            self.t,
        )
        pos = RegimePrototype(
            pos_centroid.copy(),
            float(cand.pos_n),
            prior + float(cand.pos_success),
            prior + float(cand.pos_n - cand.pos_success),
            max(1.0, parent.utility * cand.pos_n / max(1, cand.neg_n + cand.pos_n)),
            self.t,
            self.t,
        )
        neg.lineage_id = pos.lineage_id = lineage
        neg.split_created_t = pos.split_created_t = self.t
        depth = int(getattr(parent, "refinement_depth", 0)) + 1
        neg.refinement_depth = pos.refinement_depth = depth
        neg.recent_mean = neg.mean; pos.recent_mean = pos.mean
        neg.recent_obs = 0; pos.recent_obs = 0

        # Replace parent in-place by negative child; append positive child.  This
        # preserves all other prototype indices.
        self.prototypes[i] = neg
        self.tensor_cells[i] = self._inherit_tensor_cell(parent_tensor)
        self._bad_counts[i] = 0
        j = len(self.prototypes)
        self.prototypes.append(pos)
        self.tensor_cells.append(self._inherit_tensor_cell(parent_tensor))
        self._bad_counts.append(0)

        # Current child is determined by the validated split direction.
        self.current_id = j if float(np.dot(u, x_now - parent.centroid)) >= 0.0 else i
        self.revision_count += 1
        self.split_promotions += 1
        self.split_candidates.pop(key, None)
        self.split_cooldown_until.pop(key, None)
        return True

    def _merge_pair(self, i: int, j: int):
        a, b = self.prototypes[i], self.prototypes[j]
        wa, wb = max(1.0, a.count), max(1.0, b.count)
        w = wa + wb
        centroid = (wa * a.centroid + wb * b.centroid) / w
        a.centroid = centroid
        a.count = w
        a.successes += b.successes
        a.failures += b.failures
        a.utility += b.utility
        a.last_used = max(a.last_used, b.last_used)
        # Once reconciled, the distinction is no longer treated as a special
        # split lineage; future evidence must earn a new split from scratch.
        depth = max(int(getattr(a, "refinement_depth", 1)), int(getattr(b, "refinement_depth", 1)))
        a.lineage_id = None
        a.split_created_t = None
        a.refinement_depth = max(0, depth - 1)
        a.recent_mean = a.mean
        a.recent_obs = 0
        self.tensor_cells[i] = self._fresh_tensor_cell()
        self._bad_counts[i] = 0
        del self.prototypes[j]
        del self.tensor_cells[j]
        del self._bad_counts[j]
        if self.current_id is not None:
            if self.current_id == j:
                self.current_id = i
            elif self.current_id > j:
                self.current_id -= 1
        self.merge_promotions += 1
        self.recompression_count += 1
        self._purge_dead_candidates()

    def _maybe_merge_lineage(self):
        if self.t - self._last_merge_check < self.merge_interval:
            return False
        self._last_merge_check = self.t
        groups: Dict[int, List[int]] = {}
        for i, p in enumerate(self.prototypes):
            lid = getattr(p, "lineage_id", None)
            if lid is not None:
                groups.setdefault(int(lid), []).append(i)
        live_lids=set(groups)
        for lid in list(self._merge_evidence):
            if lid not in live_lids:
                del self._merge_evidence[lid]
        for lid, ids in groups.items():
            if len(ids) != 2:
                self._merge_evidence[lid]=0
                continue
            i, j = ids
            a, b = self.prototypes[i], self.prototypes[j]
            age = self.t - max(getattr(a, "split_created_t", self.t), getattr(b, "split_created_t", self.t))
            cert=True
            if age < self.merge_min_age: cert=False
            if min(a.count, b.count) < self.merge_min_count: cert=False
            if min(getattr(a, "recent_obs", 0), getattr(b, "recent_obs", 0)) < self.merge_recent_min_obs: cert=False
            if abs(float(getattr(a, "recent_mean", a.mean)) - float(getattr(b, "recent_mean", b.mean))) > self.merge_pred_tolerance: cert=False
            c = (a.count * a.centroid + b.count * b.centroid) / max(EPS, a.count + b.count)
            if max(rms(c, a.centroid), rms(c, b.centroid)) > self.merge_dist_tolerance: cert=False
            if cert:
                self._merge_evidence[lid]=self._merge_evidence.get(lid,0)+1
            else:
                self._merge_evidence[lid]=0
            if self._merge_evidence[lid] >= self.merge_patience:
                self._merge_pair(i, j)
                self._merge_evidence.pop(lid,None)
                return True
        return False

    # ------------------------------------------------------------------
    # Main step: frozen v0.9.3 first, refinement second.
    # ------------------------------------------------------------------
    def step(self, x, y=None):
        x = np.asarray(x, float)
        pre_current = self.current_id
        pre_centroid = None
        pre_tensor_steps = None
        if pre_current is not None and 0 <= pre_current < len(self.prototypes):
            pre_centroid = self.prototypes[pre_current].centroid.copy()
            pre_tensor_steps = getattr(self.tensor_cells[pre_current], "tensor_steps", 0)

        read = super().step(x, y)
        structural_edit = False

        if y is not None and self.current_id is not None:
            i = self.current_id
            self._ensure_refinement_state(i)
            pcur = self.prototypes[i]
            if getattr(pcur, "lineage_id", None) is not None:
                rr = self.merge_recent_lr
                pcur.recent_mean = (1.0 - rr) * float(getattr(pcur, "recent_mean", pcur.mean)) + rr * float(y)
                pcur.recent_obs = int(getattr(pcur, "recent_obs", 0)) + 1
            # Curvature state is updated only if the same prototype remained active
            # and v0.9.3 actually spent a tensor-awake update on this observation.
            if (
                pre_current == i
                and pre_centroid is not None
                and getattr(self.tensor_cells[i], "tensor_steps", 0) > (pre_tensor_steps or 0)
                and not read.revision
            ):
                self._update_curvature(i, x, float(y), float(read.prediction), pre_centroid)

            self._maybe_launch_split(i)
            action = self._shadow_update_candidate(i, x, float(y), float(read.prediction))
            pkey = id(self.prototypes[i]) if i < len(self.prototypes) else None
            if action == "promote" and pkey in self.split_candidates:
                cand = self.split_candidates[pkey]
                structural_edit = self._promote_split(i, cand, x)
            elif action == "reject" and pkey in self.split_candidates:
                self.split_rejections += 1
                self.split_candidates.pop(pkey, None)
                self.split_cooldown_until[pkey] = self.t + self.split_cooldown

        if self._maybe_merge_lineage():
            structural_edit = True

        self._purge_dead_candidates()

        if not structural_edit:
            return read

        # Prediction remains the pre-outcome/pre-refinement prediction; only the
        # structural fields of the readout are refreshed after the edit.
        d = self.distances(x)
        m = self._memberships(d)
        nearest = int(np.argmin(d)) if len(d) else None
        nd = float(d[nearest]) if nearest is not None else float("inf")
        active = tuple(int(k) for k in np.where(m >= self.alpha_cut)[0]) if len(m) else tuple()
        if not active and nearest is not None:
            active = (nearest,)
        return ContinualRead(
            float(read.prediction),
            m,
            active,
            self.current_id,
            nearest,
            nd,
            True,
            read.reactivated,
            read.discovered,
            read.unresolved,
            read.budget_pressure,
            len(self.prototypes),
            len(d),
        )

    def refinement_diagnostics(self):
        ratios = [float(getattr(c, "rank_ratio2", 0.0)) for c in self.tensor_cells]
        return {
            "split_proposals": self.split_proposals,
            "split_rejections": self.split_rejections,
            "split_promotions": self.split_promotions,
            "split_budget_blocks": self.split_budget_blocks,
            "merge_promotions": self.merge_promotions,
            "active_split_candidates": len(self.split_candidates),
            "rank_checks": self.rank_checks,
            "curvature_updates": self.curvature_updates,
            "max_rank_ratio2": max(ratios, default=0.0),
        }
