//! Native Cortex articulation layer.
//!
//! This crate ports the entity/invariance/binding portion of the frozen v0.9.5
//! Python reference.  It intentionally stops before relation and causal graph
//! updates; those belong to `cortex-graph` and will be integrated through the
//! top-level runtime after differential parity is established here.

mod profiling;
pub use profiling::{ArticulationStageTimings, ArticulationWork, ProfiledArticulationRead};

use serde::{Deserialize, Serialize};

const EPS: f64 = 1.0e-12;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArticulationConfig {
    pub feature_dim: usize,
    pub max_entities: usize,
    pub beta: f64,
    pub spawn_cost: f64,
    pub bootstrap_spawn_cost: f64,
    pub residual_threshold: f64,
    pub group_candidates: Option<Vec<Vec<usize>>>,
    /// Optional fixed nuisance group. When set, matching uses exactly these
    /// shifts from the first observation and subgroup learning is disabled.
    #[serde(default)]
    pub fixed_group: Option<Vec<usize>>,
}

impl Default for ArticulationConfig {
    fn default() -> Self {
        Self {
            feature_dim: 16,
            max_entities: 96,
            beta: 9.0,
            spawn_cost: 0.7,
            bootstrap_spawn_cost: 0.18,
            residual_threshold: 0.38,
            group_candidates: None,
            fixed_group: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransformEvidence {
    pub shifts: Vec<usize>,
    pub weights: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BindingResult {
    pub bindings: Vec<usize>,
    pub shifts: Vec<usize>,
    pub evidence: Vec<TransformEvidence>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArticulationRead {
    pub bindings: Vec<usize>,
    pub shifts: Vec<usize>,
    pub subgroup: Vec<usize>,
    pub subgroup_margin: f64,
    pub entity_changed: Vec<usize>,
    pub noise_changed: Vec<usize>,
    pub active_entities: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArticulationSnapshot {
    pub active_entities: usize,
    pub subgroup: Vec<usize>,
    pub subgroup_margin: f64,
    pub prototypes: Vec<Vec<f64>>,
    pub reliability: Vec<Vec<f64>>,
    pub noise_state: Vec<Vec<i8>>,
    pub time: usize,
}

#[derive(Clone, Debug)]
struct EntityStats {
    sum_x: Vec<f64>,
    count: usize,
    noise_a: Vec<f64>,
    noise_b: Vec<f64>,
    last_seen: usize,
    last_x: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct ArticulationState {
    pub cfg: ArticulationConfig,
    entities: Vec<EntityStats>,
    prototypes: Vec<Vec<f64>>,
    reliability: Vec<Vec<f64>>,
    noise_state: Vec<Vec<i8>>,
    all_shifts: Vec<usize>,
    group_candidates: Vec<Vec<usize>>,
    group_evidence: Vec<f64>,
    group_obs: usize,
    group_evidence_by_entity: Vec<Vec<f64>>,
    group_obs_by_entity: Vec<usize>,
    subgroup: Vec<usize>,
    subgroup_margin: f64,
    time: usize,
}

impl ArticulationState {
    pub fn new(cfg: ArticulationConfig) -> Result<Self, String> {
        if cfg.feature_dim == 0 {
            return Err("feature_dim must be positive".into());
        }
        if cfg.max_entities == 0 {
            return Err("max_entities must be positive".into());
        }
        if !cfg.beta.is_finite() || cfg.beta <= 0.0 {
            return Err("beta must be finite and positive".into());
        }
        let d = cfg.feature_dim;
        let all_shifts: Vec<usize> = (0..d).collect();
        if let Some(group) = &cfg.fixed_group {
            if group.is_empty() || group.iter().any(|&x| x >= d) {
                return Err("fixed_group must contain valid shifts".into());
            }
            let mut unique = group.clone();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() != group.len() {
                return Err("fixed_group cannot contain duplicate shifts".into());
            }
        }
        let group_candidates = if let Some(groups) = cfg.group_candidates.clone() {
            if groups.is_empty() {
                return Err("group_candidates cannot be empty".into());
            }
            for g in &groups {
                if g.is_empty() || g.iter().any(|&x| x >= d) {
                    return Err("each group candidate must contain valid shifts".into());
                }
            }
            groups
        } else if d % 4 != 0 {
            vec![vec![0], all_shifts.clone()]
        } else {
            vec![
                vec![0],
                vec![0, d / 2],
                vec![0, d / 4, d / 2, 3 * d / 4],
                all_shifts.clone(),
            ]
        };
        let g = group_candidates.len();
        let initial_subgroup = cfg
            .fixed_group
            .clone()
            .unwrap_or_else(|| all_shifts.clone());
        Ok(Self {
            group_evidence: vec![0.0; g],
            group_obs: 0,
            group_evidence_by_entity: vec![vec![0.0; g]; cfg.max_entities],
            group_obs_by_entity: vec![0; cfg.max_entities],
            subgroup: initial_subgroup,
            subgroup_margin: 0.0,
            entities: Vec::new(),
            prototypes: Vec::new(),
            reliability: Vec::new(),
            noise_state: Vec::new(),
            all_shifts,
            group_candidates,
            time: 0,
            cfg,
        })
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn snapshot(&self) -> ArticulationSnapshot {
        ArticulationSnapshot {
            active_entities: self.entities.len(),
            subgroup: self.subgroup.clone(),
            subgroup_margin: self.subgroup_margin,
            prototypes: self.prototypes.clone(),
            reliability: self.reliability.clone(),
            noise_state: self.noise_state.clone(),
            time: self.time,
        }
    }

    fn active_match_group(&self) -> Vec<usize> {
        if let Some(group) = &self.cfg.fixed_group {
            group.clone()
        } else if self.subgroup_margin > 0.01 {
            self.subgroup.clone()
        } else {
            self.all_shifts.clone()
        }
    }

    fn active_spawn_gate(&self) -> f64 {
        if self.cfg.fixed_group.is_some() || self.subgroup_margin > 0.01 {
            self.cfg.spawn_cost
        } else {
            self.cfg.spawn_cost.min(self.cfg.bootstrap_spawn_cost)
        }
    }

    pub fn bind(&mut self, detections: &[Vec<f64>]) -> Result<BindingResult, String> {
        for x in detections {
            self.validate_vector(x)?;
        }
        let k = detections.len();
        if k == 0 {
            return Ok(BindingResult {
                bindings: Vec::new(),
                shifts: Vec::new(),
                evidence: Vec::new(),
            });
        }
        let ne = self.entities.len();
        if ne == 0 {
            let initial_group = self.active_match_group();
            let initial_weight = 1.0 / initial_group.len() as f64;
            let mut bindings = Vec::with_capacity(k);
            let mut shifts = Vec::with_capacity(k);
            let mut evidence = Vec::with_capacity(k);
            for x in detections {
                let e = self.spawn(x)?;
                bindings.push(e);
                shifts.push(0);
                evidence.push(TransformEvidence {
                    shifts: initial_group.clone(),
                    weights: vec![initial_weight; initial_group.len()],
                });
            }
            return Ok(BindingResult {
                bindings,
                shifts,
                evidence,
            });
        }

        let match_group = self.active_match_group();
        let spawn_gate = self.active_spawn_gate();

        let cols = ne + k;
        let mut cost = vec![vec![spawn_gate; cols]; k];
        let mut best_shift = vec![vec![0usize; ne]; k];
        let mut qcache = vec![vec![None::<TransformEvidence>; ne]; k];

        for (i, x) in detections.iter().enumerate() {
            for e in 0..ne {
                let (c, g, q) = self.entity_cost_and_q(x, e, &match_group);
                cost[i][e] = c;
                best_shift[i][e] = g;
                qcache[i][e] = Some(q);
            }
            for j in 0..k {
                cost[i][ne + j] = spawn_gate + 1.0e-8 * i.abs_diff(j) as f64;
            }
        }

        let assignment = linear_sum_assignment(&cost)?;
        let mut bindings = vec![usize::MAX; k];
        let mut shifts = vec![0usize; k];
        let mut evidence = Vec::with_capacity(k);

        for i in 0..k {
            let col = assignment[i];
            if col < ne && cost[i][col] <= spawn_gate {
                bindings[i] = col;
                shifts[i] = best_shift[i][col];
                evidence.push(qcache[i][col].clone().expect("existing entity evidence"));
            } else {
                let e = self.spawn(&detections[i])?;
                bindings[i] = e;
                shifts[i] = 0;
                let weight = 1.0 / match_group.len() as f64;
                evidence.push(TransformEvidence {
                    shifts: match_group.clone(),
                    weights: vec![weight; match_group.len()],
                });
            }
        }

        // The augmented assignment must preserve simultaneous injectivity.
        let mut sorted = bindings.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err("simultaneous binding must be injective within a frame".into());
        }

        Ok(BindingResult {
            bindings,
            shifts,
            evidence,
        })
    }

    /// Process the entity/invariance/noise portion of one observation frame.
    /// Relation and causal updates are intentionally delegated to cortex-graph.
    pub fn observe(&mut self, detections: &[Vec<f64>]) -> Result<ArticulationRead, String> {
        self.time += 1;
        let binding = self.bind(detections)?;
        let mut touched = Vec::new();
        let mut noise_dirty = Vec::new();

        for (i, x) in detections.iter().enumerate() {
            let e = binding.bindings[i];
            let g = binding.shifts[i];
            let proto_before = self.prototypes[e].clone();
            let reliability_before = self.reliability[e].clone();
            let prev_raw = self.entities[e].last_x.clone();
            let aligned = inv_shift_vec(x, g);
            let residual: Vec<f64> = aligned
                .iter()
                .zip(&proto_before)
                .map(|(a, b)| (a - b).abs())
                .collect();
            let anomalies: Vec<bool> = residual
                .iter()
                .map(|&r| r > self.cfg.residual_threshold)
                .collect();
            let bind_cost = weighted_mse(x, &shift_vec(&proto_before, g), &reliability_before);

            {
                let st = &mut self.entities[e];
                for f in 0..self.cfg.feature_dim {
                    st.sum_x[f] += aligned[f];
                    if anomalies[f] {
                        st.noise_a[f] += 1.0;
                    } else {
                        st.noise_b[f] += 1.0;
                    }
                }
                st.count += 1;
                st.last_seen = self.time;
            }

            let established = self.entities[e].count > 2;
            if self.cfg.fixed_group.is_none() && established && bind_cost <= 0.35 {
                let costs: Vec<f64> = self
                    .all_shifts
                    .iter()
                    .map(|&h| {
                        weighted_mse(
                            x,
                            &shift_vec(&prev_raw, h),
                            &vec![1.0; self.cfg.feature_dim],
                        )
                    })
                    .collect();
                let qall = softmax_neg(&costs, self.cfg.beta);
                let concentration = qall.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if concentration >= 0.18 {
                    for (hi, group) in self.group_candidates.iter().enumerate() {
                        let support: f64 = group.iter().map(|&h| qall[h]).sum();
                        let ev = (support / group.len() as f64).max(EPS).ln();
                        self.group_evidence[hi] += ev;
                        self.group_evidence_by_entity[e][hi] += ev;
                    }
                    self.group_obs += 1;
                    self.group_obs_by_entity[e] += 1;
                }
            }
            self.entities[e].last_x.clone_from(x);
            touched.push(e);
            noise_dirty.push(e);
        }

        touched.sort_unstable();
        touched.dedup();
        noise_dirty.sort_unstable();
        noise_dirty.dedup();

        let mut entity_changed = Vec::new();
        for &e in &touched {
            let p = self.derive_prototype(e);
            if p != self.prototypes[e] {
                self.prototypes[e] = p;
                entity_changed.push(e);
            }
            self.reliability[e] = self.derive_reliability(e);
        }

        let mut noise_changed = Vec::new();
        for &e in &noise_dirty {
            let ns = self.derive_noise_state(e);
            if ns != self.noise_state[e] {
                self.noise_state[e] = ns;
                noise_changed.push(e);
            }
        }

        let (subgroup, margin) = self.derive_subgroup();
        self.subgroup = subgroup;
        self.subgroup_margin = margin;

        Ok(ArticulationRead {
            bindings: binding.bindings,
            shifts: binding.shifts,
            subgroup: self.subgroup.clone(),
            subgroup_margin: self.subgroup_margin,
            entity_changed,
            noise_changed,
            active_entities: self.entities.len(),
        })
    }

    fn validate_vector(&self, x: &[f64]) -> Result<(), String> {
        if x.len() != self.cfg.feature_dim {
            return Err(format!(
                "expected vector of length {}, got {}",
                self.cfg.feature_dim,
                x.len()
            ));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err("observation contains non-finite values".into());
        }
        Ok(())
    }

    fn spawn(&mut self, x: &[f64]) -> Result<usize, String> {
        if self.entities.len() >= self.cfg.max_entities {
            return Err(format!(
                "entity capacity exhausted ({}); cannot spawn a distinct identity",
                self.cfg.max_entities
            ));
        }
        let e = self.entities.len();
        self.entities.push(EntityStats {
            sum_x: x.to_vec(),
            count: 1,
            noise_a: vec![1.0; self.cfg.feature_dim],
            noise_b: vec![1.0; self.cfg.feature_dim],
            last_seen: self.time,
            last_x: x.to_vec(),
        });
        self.prototypes.push(x.to_vec());
        self.reliability.push(vec![0.5; self.cfg.feature_dim]);
        self.noise_state.push(vec![0; self.cfg.feature_dim]);
        Ok(e)
    }

    fn entity_cost_and_q(
        &self,
        x: &[f64],
        entity: usize,
        match_group: &[usize],
    ) -> (f64, usize, TransformEvidence) {
        let proto = &self.prototypes[entity];
        let w = &self.reliability[entity];
        let costs: Vec<f64> = match_group
            .iter()
            .map(|&g| weighted_mse(x, &shift_vec(proto, g), w))
            .collect();
        let q = softmax_neg(&costs, self.cfg.beta);
        let mut best = 0usize;
        let mut best_cost = f64::INFINITY;
        for (j, &c) in costs.iter().enumerate() {
            // np.argmin semantics: preserve the first exact minimum.
            if c < best_cost {
                best_cost = c;
                best = j;
            }
        }
        (
            best_cost,
            match_group[best],
            TransformEvidence {
                shifts: match_group.to_vec(),
                weights: q,
            },
        )
    }

    fn derive_prototype(&self, e: usize) -> Vec<f64> {
        let st = &self.entities[e];
        let denom = st.count.max(1) as f64;
        st.sum_x.iter().map(|v| v / denom).collect()
    }

    fn derive_reliability(&self, e: usize) -> Vec<f64> {
        let st = &self.entities[e];
        (0..self.cfg.feature_dim)
            .map(|f| {
                let p = st.noise_a[f] / (st.noise_a[f] + st.noise_b[f]);
                (1.0 - p).clamp(0.18, 1.0)
            })
            .collect()
    }

    fn derive_noise_state(&self, e: usize) -> Vec<i8> {
        let st = &self.entities[e];
        let mut out = vec![0i8; self.cfg.feature_dim];
        for (f, slot) in out.iter_mut().enumerate() {
            let p = st.noise_a[f] / (st.noise_a[f] + st.noise_b[f]);
            let n = st.noise_a[f] + st.noise_b[f] - 2.0;
            if n >= 6.0 && p >= 0.30 {
                *slot = 2;
            } else if n >= 6.0 && p <= 0.10 {
                *slot = -1;
            }
        }
        out
    }

    fn derive_subgroup(&self) -> (Vec<usize>, f64) {
        if let Some(group) = &self.cfg.fixed_group {
            return (group.clone(), 0.0);
        }
        let eligible: Vec<usize> = self
            .group_obs_by_entity
            .iter()
            .enumerate()
            .filter_map(|(e, &n)| (n >= 4).then_some(e))
            .collect();
        let total_obs: usize = eligible.iter().map(|&e| self.group_obs_by_entity[e]).sum();
        if eligible.len() < 2 || total_obs < 12 {
            return (self.all_shifts.clone(), 0.0);
        }

        let mut scores = vec![0.0; self.group_candidates.len()];
        for (hi, score) in scores.iter_mut().enumerate() {
            let mut vals: Vec<f64> = eligible
                .iter()
                .map(|&e| self.group_evidence_by_entity[e][hi] / self.group_obs_by_entity[e] as f64)
                .collect();
            vals.sort_by(f64::total_cmp);
            *score = median_sorted(&vals);
        }
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
        order.reverse();
        let best = order[0];
        let second = *order.get(1).unwrap_or(&best);
        (
            self.group_candidates[best].clone(),
            scores[best] - scores[second],
        )
    }
}

fn median_sorted(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

fn softmax_neg(costs: &[f64], beta: f64) -> Vec<f64> {
    if costs.is_empty() {
        return Vec::new();
    }
    let mut z: Vec<f64> = costs.iter().map(|&c| -beta * c).collect();
    let max_z = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for v in &mut z {
        *v = (*v - max_z).exp();
        sum += *v;
    }
    let denom = sum.max(EPS);
    for v in &mut z {
        *v /= denom;
    }
    z
}

fn shift_vec(x: &[f64], g: usize) -> Vec<f64> {
    let d = x.len();
    if d == 0 {
        return Vec::new();
    }
    let g = g % d;
    (0..d).map(|i| x[(i + d - g) % d]).collect()
}

fn inv_shift_vec(x: &[f64], g: usize) -> Vec<f64> {
    let d = x.len();
    if d == 0 {
        return Vec::new();
    }
    let g = g % d;
    (0..d).map(|i| x[(i + g) % d]).collect()
}

fn weighted_mse(x: &[f64], y: &[f64], w: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), w.len());
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..x.len() {
        num += w[i] * (x[i] - y[i]).powi(2);
        den += w[i];
    }
    num / den.max(EPS)
}

/// Rectangular Hungarian assignment for row_count <= col_count.
/// Strict `<` comparisons intentionally preserve the first exact minimum.
fn linear_sum_assignment(cost: &[Vec<f64>]) -> Result<Vec<usize>, String> {
    let n = cost.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let m = cost[0].len();
    if n > m {
        return Err("assignment requires at least as many columns as rows".into());
    }
    if m == 0 || cost.iter().any(|row| row.len() != m) {
        return Err("assignment cost matrix must be non-empty and rectangular".into());
    }

    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; m + 1];
    let mut p = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=m {
                if used[j] {
                    continue;
                }
                let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else if j > 0 {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![usize::MAX; n];
    for j in 1..=m {
        if p[j] != 0 {
            assignment[p[j] - 1] = j - 1;
        }
    }
    if assignment.contains(&usize::MAX) {
        return Err("assignment failed to cover every row".into());
    }
    Ok(assignment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cyclic_shift_matches_numpy_roll_convention() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(shift_vec(&x, 1), vec![4.0, 1.0, 2.0, 3.0]);
        assert_eq!(inv_shift_vec(&shift_vec(&x, 1), 1), x);
    }

    #[test]
    fn binding_is_injective_and_reactivates_shifted_entities() {
        let cfg = ArticulationConfig {
            feature_dim: 8,
            max_entities: 8,
            ..ArticulationConfig::default()
        };
        let mut s = ArticulationState::new(cfg).unwrap();
        let a = vec![0.1, 0.7, -0.2, 1.3, 0.4, -0.9, 0.2, 0.5];
        let b = vec![1.1, -0.4, 0.8, 0.2, -1.2, 0.3, 0.6, -0.1];
        let first = s.observe(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(first.bindings, vec![0, 1]);
        let second = s.observe(&[shift_vec(&a, 2), shift_vec(&b, 4)]).unwrap();
        assert_eq!(second.bindings, vec![0, 1]);
        assert_eq!(second.shifts, vec![2, 4]);
        assert_eq!(s.entity_count(), 2);
    }

    #[test]
    fn fixed_group_uses_normal_spawn_gate_immediately() {
        let cfg = ArticulationConfig {
            feature_dim: 4,
            max_entities: 8,
            fixed_group: Some(vec![0]),
            ..ArticulationConfig::default()
        };
        let mut s = ArticulationState::new(cfg).unwrap();
        let first = s.observe(&[vec![0.0; 4]]).unwrap();
        assert_eq!(first.bindings, vec![0]);

        // MSE = 0.30: above bootstrap_spawn_cost (0.18), below normal
        // spawn_cost (0.70). Fixed mode must use the normal gate immediately.
        let delta = 0.3_f64.sqrt();
        let second = s.observe(&[vec![delta; 4]]).unwrap();
        assert_eq!(second.bindings, vec![0]);
        assert_eq!(s.entity_count(), 1);
        assert_eq!(second.subgroup, vec![0]);
    }

    #[test]
    fn fixed_identity_group_cannot_match_by_cyclic_shift() {
        let cfg = ArticulationConfig {
            feature_dim: 8,
            max_entities: 8,
            fixed_group: Some(vec![0]),
            ..ArticulationConfig::default()
        };
        let mut s = ArticulationState::new(cfg).unwrap();
        let a = vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        s.observe(&[a.clone()]).unwrap();

        // This is exactly the same vector under a native cyclic shift, but a
        // fixed identity group must not use that transform to rescue the match.
        let shifted = shift_vec(&a, 2);
        let read = s.observe(&[shifted]).unwrap();
        assert_eq!(read.bindings, vec![1]);
        assert_eq!(read.shifts, vec![0]);
        assert_eq!(s.entity_count(), 2);
    }

    #[test]
    fn invalid_fixed_groups_are_rejected() {
        for group in [vec![], vec![0, 0], vec![4]] {
            let cfg = ArticulationConfig {
                feature_dim: 4,
                fixed_group: Some(group),
                ..ArticulationConfig::default()
            };
            assert!(ArticulationState::new(cfg).is_err());
        }
    }

    #[test]
    fn profiled_fixed_group_preserves_production_semantics() {
        let cfg = ArticulationConfig {
            feature_dim: 4,
            max_entities: 8,
            fixed_group: Some(vec![0]),
            ..ArticulationConfig::default()
        };
        let mut normal = ArticulationState::new(cfg.clone()).unwrap();
        let mut profiled = ArticulationState::new(cfg).unwrap();
        let sequence = [
            vec![vec![0.0, 0.0, 0.0, 0.0]],
            vec![vec![0.5, 0.5, 0.5, 0.5]],
            vec![vec![2.0, 0.0, 0.0, 0.0]],
        ];
        for detections in sequence {
            let expected = normal.observe(&detections).unwrap();
            let measured = profiled.observe_profiled(&detections).unwrap();
            assert_eq!(expected.bindings, measured.read.bindings);
            assert_eq!(expected.shifts, measured.read.shifts);
            assert_eq!(expected.subgroup, measured.read.subgroup);
            assert_eq!(normal.snapshot().active_entities, profiled.snapshot().active_entities);
            if normal.snapshot().active_entities > 0 {
                assert_eq!(measured.work.match_group_size, 1);
            }
        }
        assert_eq!(normal.snapshot().prototypes, profiled.snapshot().prototypes);
        assert_eq!(normal.snapshot().reliability, profiled.snapshot().reliability);
        assert_eq!(normal.snapshot().noise_state, profiled.snapshot().noise_state);
    }

    #[test]
    fn capacity_failure_is_explicit() {
        let cfg = ArticulationConfig {
            feature_dim: 4,
            max_entities: 1,
            bootstrap_spawn_cost: 0.01,
            ..ArticulationConfig::default()
        };
        let mut s = ArticulationState::new(cfg).unwrap();
        s.observe(&[vec![1.0, 0.0, 0.0, 0.0]]).unwrap();
        let err = s.observe(&[vec![10.0, 10.0, 10.0, 10.0]]).unwrap_err();
        assert!(err.contains("capacity exhausted"));
    }
}
