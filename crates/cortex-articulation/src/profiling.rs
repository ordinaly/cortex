use super::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Opt-in timings for the internal articulation pipeline.
///
/// The production `ArticulationState::observe` path is unchanged. This profiler
/// mirrors that transition for research attribution and is guarded by parity
/// tests against the production path.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArticulationStageTimings {
    pub validation_ns: u64,
    pub match_scoring_ns: u64,
    pub assignment_ns: u64,
    pub bind_finalize_ns: u64,
    pub entity_update_ns: u64,
    pub subgroup_evidence_ns: u64,
    pub dedup_ns: u64,
    pub prototype_reliability_ns: u64,
    pub noise_state_ns: u64,
    pub subgroup_select_ns: u64,
    pub total_ns: u64,
}

/// Deterministic work counters accompanying articulation timings.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArticulationWork {
    pub detections: usize,
    pub entities_before: usize,
    pub match_group_size: usize,
    pub pair_scores: usize,
    pub transform_evaluations: usize,
    pub assignment_rows: usize,
    pub assignment_cols: usize,
    pub spawned: usize,
    pub subgroup_evidence_detections: usize,
    pub subgroup_transform_evaluations: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfiledArticulationRead {
    pub read: ArticulationRead,
    pub timings: ArticulationStageTimings,
    pub work: ArticulationWork,
}

impl ArticulationState {
    /// Execute the articulation transition with fine-grained timing and work
    /// counters. This method exists only for research attribution.
    pub fn observe_profiled(
        &mut self,
        detections: &[Vec<f64>],
    ) -> Result<ProfiledArticulationRead, String> {
        let total_start = Instant::now();
        self.time += 1;

        let (binding, mut timings, mut work) = self.bind_profiled(detections)?;
        let mut touched = Vec::new();
        let mut noise_dirty = Vec::new();

        for (i, x) in detections.iter().enumerate() {
            let update_start = Instant::now();
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

            let mut evidence_ns = 0u64;
            let established = self.entities[e].count > 2;
            if established && bind_cost <= 0.35 {
                let evidence_start = Instant::now();
                work.subgroup_evidence_detections += 1;
                work.subgroup_transform_evaluations += self.all_shifts.len();
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
                evidence_ns = elapsed_ns(evidence_start);
                timings.subgroup_evidence_ns =
                    timings.subgroup_evidence_ns.saturating_add(evidence_ns);
            }
            self.entities[e].last_x.clone_from(x);
            touched.push(e);
            noise_dirty.push(e);
            let update_ns = elapsed_ns(update_start);
            timings.entity_update_ns = timings
                .entity_update_ns
                .saturating_add(update_ns.saturating_sub(evidence_ns));
        }

        let stage_start = Instant::now();
        touched.sort_unstable();
        touched.dedup();
        noise_dirty.sort_unstable();
        noise_dirty.dedup();
        timings.dedup_ns = elapsed_ns(stage_start);

        let stage_start = Instant::now();
        let mut entity_changed = Vec::new();
        for &e in &touched {
            let p = self.derive_prototype(e);
            if p != self.prototypes[e] {
                self.prototypes[e] = p;
                entity_changed.push(e);
            }
            self.reliability[e] = self.derive_reliability(e);
        }
        timings.prototype_reliability_ns = elapsed_ns(stage_start);

        let stage_start = Instant::now();
        let mut noise_changed = Vec::new();
        for &e in &noise_dirty {
            let ns = self.derive_noise_state(e);
            if ns != self.noise_state[e] {
                self.noise_state[e] = ns;
                noise_changed.push(e);
            }
        }
        timings.noise_state_ns = elapsed_ns(stage_start);

        let stage_start = Instant::now();
        let (subgroup, margin) = self.derive_subgroup();
        self.subgroup = subgroup;
        self.subgroup_margin = margin;
        timings.subgroup_select_ns = elapsed_ns(stage_start);

        timings.total_ns = elapsed_ns(total_start);
        Ok(ProfiledArticulationRead {
            read: ArticulationRead {
                bindings: binding.bindings,
                shifts: binding.shifts,
                subgroup: self.subgroup.clone(),
                subgroup_margin: self.subgroup_margin,
                entity_changed,
                noise_changed,
                active_entities: self.entities.len(),
            },
            timings,
            work,
        })
    }

    fn bind_profiled(
        &mut self,
        detections: &[Vec<f64>],
    ) -> Result<(BindingResult, ArticulationStageTimings, ArticulationWork), String> {
        let mut timings = ArticulationStageTimings::default();
        let mut work = ArticulationWork {
            detections: detections.len(),
            entities_before: self.entities.len(),
            ..ArticulationWork::default()
        };

        let stage_start = Instant::now();
        for x in detections {
            self.validate_vector(x)?;
        }
        timings.validation_ns = elapsed_ns(stage_start);

        let k = detections.len();
        if k == 0 {
            return Ok((
                BindingResult {
                    bindings: Vec::new(),
                    shifts: Vec::new(),
                    evidence: Vec::new(),
                },
                timings,
                work,
            ));
        }
        let ne = self.entities.len();
        if ne == 0 {
            let stage_start = Instant::now();
            let mut bindings = Vec::with_capacity(k);
            let mut shifts = Vec::with_capacity(k);
            let mut evidence = Vec::with_capacity(k);
            for x in detections {
                let e = self.spawn(x)?;
                work.spawned += 1;
                bindings.push(e);
                shifts.push(0);
                evidence.push(TransformEvidence {
                    shifts: self.all_shifts.clone(),
                    weights: vec![1.0 / self.cfg.feature_dim as f64; self.cfg.feature_dim],
                });
            }
            timings.bind_finalize_ns = elapsed_ns(stage_start);
            return Ok((
                BindingResult {
                    bindings,
                    shifts,
                    evidence,
                },
                timings,
                work,
            ));
        }

        let confident = self.subgroup_margin > 0.01;
        let match_group = if confident {
            self.subgroup.clone()
        } else {
            self.all_shifts.clone()
        };
        work.match_group_size = match_group.len();
        let spawn_gate = if confident {
            self.cfg.spawn_cost
        } else {
            self.cfg.spawn_cost.min(self.cfg.bootstrap_spawn_cost)
        };

        let stage_start = Instant::now();
        let cols = ne + k;
        let mut cost = vec![vec![spawn_gate; cols]; k];
        let mut best_shift = vec![vec![0usize; ne]; k];
        let mut qcache = vec![vec![None::<TransformEvidence>; ne]; k];
        work.pair_scores = k.saturating_mul(ne);
        work.transform_evaluations = work.pair_scores.saturating_mul(match_group.len());

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
        timings.match_scoring_ns = elapsed_ns(stage_start);

        let stage_start = Instant::now();
        let assignment = linear_sum_assignment(&cost)?;
        timings.assignment_ns = elapsed_ns(stage_start);
        work.assignment_rows = k;
        work.assignment_cols = cols;

        let stage_start = Instant::now();
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
                work.spawned += 1;
                bindings[i] = e;
                shifts[i] = 0;
                evidence.push(TransformEvidence {
                    shifts: self.all_shifts.clone(),
                    weights: vec![1.0 / self.cfg.feature_dim as f64; self.cfg.feature_dim],
                });
            }
        }

        let mut sorted = bindings.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err("simultaneous binding must be injective within a frame".into());
        }
        timings.bind_finalize_ns = elapsed_ns(stage_start);

        Ok((
            BindingResult {
                bindings,
                shifts,
                evidence,
            },
            timings,
            work,
        ))
    }
}

fn elapsed_ns(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}
