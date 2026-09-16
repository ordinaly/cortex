"""Cortex v0.8 — Fuzzy Accordion Memory.

Adds a bounded approximate memory layer to the frozen Cortex v0.7 active
articulation runtime. The active v0.7 learner remains exact with respect to its
v0.7 contract; fuzzy memory is downstream historical/predictive compression.

Design principles
-----------------
1. No fuzzy merge is allowed to alter the active v0.7 articulation.
2. Similarity is graded, but compression is accepted only under an explicit
   predictive-distortion budget.
3. When the prototype budget is full, a new consequential state may displace
   memory only if a redundant prototype pair can first be certified mergeable.
   Otherwise the memory returns an explicit unresolved/budget-pressure signal.
4. Activation is an alpha-cut over fuzzy memberships, so stored complexity and
   active computational complexity are separate quantities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np

from cortex_v07 import DependencyIntegratedRuntime, Frame

EPS = 1e-12


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


@dataclass(frozen=True)
class FuzzyMemoryRead:
    reconstruction: np.ndarray
    memberships: np.ndarray
    active_ids: tuple[int, ...]
    nearest_id: int | None
    nearest_dist: float
    stored: int
    unresolved: bool
    compressed: bool
    structural_change: bool


class FuzzyAccordionMemory:
    """Budgeted fuzzy prototype memory with certified recompression.

    Parameters
    ----------
    budget:
        Maximum number of stored prototypes.
    tau:
        Similarity temperature E(x,c)=exp(-d(x,c)/tau).
    alpha:
        Membership alpha-cut controlling active memory.
    fit_tolerance:
        States within this RMS distance may update the nearest prototype.
    distortion_budget:
        Maximum reconstruction distortion allowed for an accepted approximate
        representation. Above it, the state is consequentially unresolved.
    redundancy_tolerance:
        Two stored prototypes may be coarsened only if their mutual distance is
        below this value. This is the explicit coarsening certificate.
    decay:
        Utility decay used only for diagnostics/activation pressure.
    """
    def __init__(self, dim: int, budget: int = 32, tau: float = 0.08,
                 alpha: float = 0.12, fit_tolerance: float = 0.035,
                 distortion_budget: float = 0.08,
                 redundancy_tolerance: float = 0.045, decay: float = 0.995, recompress_interval: int = 16):
        self.dim = int(dim)
        self.budget = int(budget)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.fit_tolerance = float(fit_tolerance)
        self.distortion_budget = float(distortion_budget)
        self.redundancy_tolerance = float(redundancy_tolerance)
        self.decay = float(decay)
        self.recompress_interval = max(1, int(recompress_interval))
        self._last_recompress_check = -10**9
        self.prototypes: list[np.ndarray] = []
        self.counts: list[float] = []
        self.utility: list[float] = []
        self.last_used: list[int] = []
        self.t = 0
        self.unresolved_count = 0
        self.recompression_count = 0
        self.spawn_count = 0
        self.prototype_updates = 0

    def _distances(self, x: np.ndarray) -> np.ndarray:
        if not self.prototypes:
            return np.empty(0, dtype=float)
        return np.array([_l2(x, c) for c in self.prototypes], dtype=float)

    def _memberships_from_distances(self, d: np.ndarray) -> np.ndarray:
        if len(d) == 0:
            return np.empty(0, dtype=float)
        s = np.exp(-d / max(EPS, self.tau))
        z = float(s.sum())
        if z <= EPS:
            out = np.zeros_like(s); out[int(np.argmin(d))] = 1.0; return out
        return s / z

    def memberships(self, x: Sequence[float]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self._memberships_from_distances(self._distances(x))

    def reconstruct(self, memberships: np.ndarray) -> np.ndarray:
        if len(self.prototypes) == 0:
            return np.zeros(self.dim, dtype=float)
        m = np.asarray(memberships, dtype=float)
        if m.sum() <= EPS:
            return self.prototypes[int(np.argmax(m))].copy()
        P = np.stack(self.prototypes, axis=0)
        return (m[:, None] * P).sum(axis=0) / max(EPS, float(m.sum()))

    def _append(self, x: np.ndarray) -> int:
        self.prototypes.append(x.copy())
        self.counts.append(1.0)
        self.utility.append(1.0)
        self.last_used.append(self.t)
        self.spawn_count += 1
        return len(self.prototypes) - 1

    def _certified_recompress(self) -> bool:
        """Merge the most redundant pair iff redundancy is certified.

        This is deliberately asymmetric with refinement: a novel state is a
        positive witness for refinement, whereas coarsening requires a separate
        redundancy certificate.
        """
        n = len(self.prototypes)
        if n < 2:
            return False
        # Negative recompression results are cached briefly. Prototype movement
        # is slow, so re-testing all O(B^2) pairs on every observation wastes
        # compute while a short delay in optional coarsening is conservative.
        if self.t - self._last_recompress_check < self.recompress_interval:
            return False
        self._last_recompress_check = self.t
        P = np.stack(self.prototypes, axis=0)
        # RMS pairwise distances, vectorized.
        D = np.sqrt(np.mean((P[:,None,:]-P[None,:,:])**2, axis=2))
        D[np.tril_indices(n)] = np.inf
        flat = int(np.argmin(D)); i, j = np.unravel_index(flat, D.shape)
        best_d = float(D[i,j])
        if not np.isfinite(best_d) or best_d > self.redundancy_tolerance:
            return False
        wi, wj = self.counts[i], self.counts[j]
        merged = (wi * self.prototypes[i] + wj * self.prototypes[j]) / max(EPS, wi + wj)
        # Each source prototype must remain within the predictive distortion budget.
        if max(_l2(merged, self.prototypes[i]), _l2(merged, self.prototypes[j])) > self.distortion_budget:
            return False
        self.prototypes[i] = merged
        self.counts[i] = wi + wj
        self.utility[i] = self.utility[i] + self.utility[j]
        self.last_used[i] = max(self.last_used[i], self.last_used[j])
        del self.prototypes[j]; del self.counts[j]; del self.utility[j]; del self.last_used[j]
        self.recompression_count += 1
        return True

    def step(self, x: Sequence[float]) -> FuzzyMemoryRead:
        self.t += 1
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,):
            raise ValueError(f"expected vector of shape {(self.dim,)}, got {x.shape}")
        self.utility = [u * self.decay for u in self.utility]
        structural_change = False
        compressed = False
        unresolved = False

        if not self.prototypes:
            k = self._append(x); structural_change = True
        else:
            d0 = self._distances(x)
            nearest = int(np.argmin(d0)); dn = float(d0[nearest])
            if dn <= self.fit_tolerance:
                # Local online prototype update; this is approximate memory only.
                c = self.counts[nearest]
                self.prototypes[nearest] = (c * self.prototypes[nearest] + x) / (c + 1.0)
                self.counts[nearest] = c + 1.0
                self.prototype_updates += 1
                compressed = True
            elif len(self.prototypes) < self.budget:
                k = self._append(x); structural_change = True
            else:
                # Budget full. A consequentially novel state can enter only if
                # redundancy elsewhere is independently certified.
                if self._certified_recompress():
                    self._append(x); structural_change = True
                else:
                    # We can still *read* an approximate reconstruction when it
                    # satisfies the distortion budget, but we never pretend a
                    # larger error is a valid merge.
                    m0 = self._memberships_from_distances(d0)
                    r0 = self.reconstruct(m0)
                    if _l2(x, r0) <= self.distortion_budget:
                        compressed = True
                    else:
                        unresolved = True
                        self.unresolved_count += 1

        d = self._distances(x)
        m = self._memberships_from_distances(d)
        r = self.reconstruct(m)
        if len(m):
            nearest = int(np.argmin(d)); nearest_dist = float(d[nearest])
            active = tuple(int(i) for i in np.where(m >= self.alpha)[0])
            if not active:
                active = (nearest,)
            for i in active:
                self.utility[i] += float(m[i])
                self.last_used[i] = self.t
        else:
            nearest = None; nearest_dist = float("inf"); active = tuple()
        return FuzzyMemoryRead(r, m, active, nearest, nearest_dist, len(self.prototypes),
                               unresolved, compressed, structural_change)


class CortexV08Runtime:
    """Frozen v0.7 dependency runtime + bounded fuzzy historical memory."""
    def __init__(self, memory_dim: int = 12, memory_budget: int = 32,
                 memory_kwargs: dict | None = None, **v07_kwargs):
        self.core = DependencyIntegratedRuntime(**v07_kwargs)
        mk = dict(memory_kwargs or {})
        self.memory = FuzzyAccordionMemory(memory_dim, budget=memory_budget, **mk)
        self.memory_dim = memory_dim

    @property
    def state(self):
        return self.core.state

    def articulation_vector(self) -> np.ndarray:
        """Public summary of current learned articulation for historical memory.

        The vector deliberately summarizes rather than redefines v0.7 state.
        """
        st = self.core.state
        ne = len(st.entities)
        # subgroup candidate one-hot (or all-zero if not exact candidate)
        gvec = np.zeros(4, dtype=float)
        for i, g in enumerate(st.group_candidates[:4]):
            if tuple(st.subgroup) == tuple(g): gvec[i] = 1.0
        # residual-state fractions over active entities/coordinates
        if ne:
            ns = np.concatenate([st.noise_state[e] for e in range(ne)])
            noise = np.array([(ns == -1).mean(), (ns == 0).mean(), (ns == 2).mean()])
        else:
            noise = np.array([0.0, 1.0, 0.0])
        # relation and causal state fractions restricted to active entity cells
        if ne >= 2:
            rids = [st.pidx.pair_id(a,b) for a in range(ne) for b in range(a+1,ne)]
            rs = st.relation_state[rids]
            rel_active = float((rs == 1).mean()) if len(rs) else 0.0
            rel_resolved = float((rs != 0).mean()) if len(rs) else 0.0
            oids = [st.pidx.ordered_id(a,b) for a in range(ne) for b in range(ne) if a != b]
            cs = st.causal_state[oids]
            causal_active = float((cs == 1).mean()) if len(cs) else 0.0
        else:
            rel_active = rel_resolved = causal_active = 0.0
        base = np.concatenate([
            np.array([min(1.0, ne / max(1, st.max_entities)),
                      min(1.0, max(0.0, st.subgroup_margin) / 0.25)]),
            gvec,
            noise,
            np.array([rel_active, rel_resolved, causal_active])
        ])
        if len(base) != self.memory_dim:
            raise RuntimeError(f"articulation vector dim {len(base)} != memory_dim {self.memory_dim}")
        return base

    def step(self, frame: Frame):
        out = self.core.step(frame)
        mem = self.memory.step(self.articulation_vector())
        return out, mem
