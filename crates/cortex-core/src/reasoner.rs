use crate::math::{canonicalize_direction, clip, column, dot, jacobi_eigen_symmetric, logit, norm, rms, sigmoid, EPS};
use crate::state::*;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShadowAction { Promote, Reject }

#[derive(Debug, Clone)]
pub struct CortexReasoner {
    pub cfg: Config,
    pub prototypes: Vec<Prototype>,
    pub tensor_cells: Vec<TensorCell>,
    bad_counts: Vec<usize>,
    pub current_id: Option<usize>,
    pub t: usize,
    pending_kind: Option<String>,
    pending_id: Option<usize>,
    pending_count: usize,
    pending_sum: Vec<f64>,
    pending_score: f64,
    sigma2: f64,
    hazard: f64,
    last_recompress_check: isize,
    pub revision_count: usize,
    pub reactivation_count: usize,
    pub discovery_count: usize,
    pub unresolved_count: usize,
    pub budget_pressure_count: usize,
    pub recompression_count: usize,
    pub comparison_count: usize,
    pub tensor_updates: usize,
    pub tensor_ready_events: usize,
    pub tensor_awake_steps: usize,
    pub tensor_sleep_steps: usize,
    pub tensor_refreshes: usize,
    pub tensor_wakes: usize,
    pub tensor_sleeps: usize,
    pub split_candidates: HashMap<u64, SplitCandidate>,
    split_cooldown_until: HashMap<u64, usize>,
    pub split_proposals: usize,
    pub split_rejections: usize,
    pub split_promotions: usize,
    pub split_budget_blocks: usize,
    pub merge_promotions: usize,
    pub rank_checks: usize,
    pub curvature_updates: usize,
    next_lineage: u64,
    next_uid: u64,
    last_merge_check: isize,
    merge_evidence: HashMap<u64, usize>,
    lineage_active: bool,
}

impl CortexReasoner {
    pub fn new(cfg: Config) -> Result<Self, String> {
        if cfg.dim == 0 { return Err("dim must be > 0".into()); }
        if cfg.budget == 0 { return Err("budget must be > 0".into()); }
        if cfg.tensor_rates.is_empty() { return Err("tensor_rates must not be empty".into()); }
        let sigma2 = cfg.min_sigma.powi(2).max((cfg.stay_tolerance / 2.0).powi(2));
        Ok(Self {
            pending_sum: vec![0.0; cfg.dim],
            cfg,
            prototypes: Vec::new(), tensor_cells: Vec::new(), bad_counts: Vec::new(),
            current_id: None, t: 0, pending_kind: None, pending_id: None, pending_count: 0,
            pending_score: 0.0, sigma2, hazard: 0.0, last_recompress_check: -1_000_000_000,
            revision_count: 0, reactivation_count: 0, discovery_count: 0,
            unresolved_count: 0, budget_pressure_count: 0, recompression_count: 0,
            comparison_count: 0, tensor_updates: 0, tensor_ready_events: 0,
            tensor_awake_steps: 0, tensor_sleep_steps: 0, tensor_refreshes: 0,
            tensor_wakes: 0, tensor_sleeps: 0, split_candidates: HashMap::new(),
            split_cooldown_until: HashMap::new(), split_proposals: 0, split_rejections: 0,
            split_promotions: 0, split_budget_blocks: 0, merge_promotions: 0,
            rank_checks: 0, curvature_updates: 0, next_lineage: 1, next_uid: 1,
            last_merge_check: -1_000_000_000, merge_evidence: HashMap::new(), lineage_active: false,
        })
    }

    pub fn from_json(s: &str) -> Result<Self, String> {
        let cfg: Config = serde_json::from_str(s).map_err(|e| e.to_string())?;
        Self::new(cfg)
    }

    fn fresh_tensor_cell(&self) -> TensorCell {
        let d = self.cfg.dim;
        let s = self.cfg.tensor_rates.len();
        let mut u = vec![0.0; d];
        u[0] = 1.0;
        TensorCell {
            tensor: vec![0.0; d * s * 2], u, theta: 0.0, proj2: 0.01, obs: 0,
            coherence: 0.0, strength: 0.0, ready: false,
            base_loss: 0.25, corr_loss: 0.25, burst_left: self.cfg.initial_burst,
            since_burst: 0, tensor_steps: 0, refreshes: 0, wakes: 0, sleeps: 0,
            curvature: vec![0.0; d * d], curvature_obs: 0, rank_ratio2: 0.0,
            rank_strength1: 0.0, rank_strength2: 0.0, rank_persist: 0,
            last_rank_refresh_seen: -1, rank_dirs: Vec::new(),
        }
    }

    fn inherit_tensor_cell(&self, parent: &TensorCell) -> TensorCell {
        let mut c = parent.clone();
        c.curvature.fill(0.0);
        c.curvature_obs = 0;
        c.rank_ratio2 = 0.0; c.rank_strength1 = 0.0; c.rank_strength2 = 0.0;
        c.rank_persist = 0; c.last_rank_refresh_seen = -1; c.rank_dirs.clear();
        c.burst_left = c.burst_left.max(self.cfg.refresh_burst);
        c.since_burst = 0;
        c
    }

    fn append(&mut self, x: &[f64], t: usize) -> usize {
        let uid = self.next_uid; self.next_uid += 1;
        let prior = self.cfg.prior_strength / 2.0;
        self.prototypes.push(Prototype {
            uid, centroid: x.to_vec(), count: 1.0, successes: prior, failures: prior,
            utility: 1.0, last_used: t, created_t: t, lineage_id: None,
            split_created_t: None, recent_mean: 0.5, recent_obs: 0, refinement_depth: 0,
        });
        let i = self.prototypes.len() - 1;
        self.prototypes[i].recent_mean = self.prototypes[i].mean();
        self.tensor_cells.push(self.fresh_tensor_cell());
        self.bad_counts.push(0);
        self.discovery_count += 1;
        i
    }

    fn distances(&mut self, x: &[f64]) -> Vec<f64> {
        self.comparison_count += self.prototypes.len();
        self.prototypes.iter().map(|p| rms(&p.centroid, x)).collect()
    }

    fn memberships(&self, d: &[f64]) -> Vec<f64> {
        if d.is_empty() { return Vec::new(); }
        let mut s: Vec<f64> = d.iter().map(|v| (-v / self.cfg.tau.max(EPS)).exp()).collect();
        let z: f64 = s.iter().sum();
        if z <= EPS {
            let mut m = vec![0.0; d.len()];
            if let Some((i, _)) = d.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)) { m[i] = 1.0; }
            return m;
        }
        for v in &mut s { *v /= z; }
        s
    }

    fn active_ids(&self, m: &[f64], nearest: Option<usize>) -> Vec<usize> {
        let mut out: Vec<usize> = m.iter().enumerate().filter_map(|(i, &v)| if v >= self.cfg.alpha_cut { Some(i) } else { None }).collect();
        if out.is_empty() { if let Some(i) = nearest { out.push(i); } }
        out
    }

    fn prototype_tensor_prediction(&self, i: usize, x: &[f64]) -> (f64, f64) {
        let p = &self.prototypes[i];
        let base = p.mean();
        let c = &self.tensor_cells[i];
        if !c.ready { return (base, 0.0); }
        let z: Vec<f64> = x.iter().zip(&p.centroid).map(|(a, b)| a - b).collect();
        let q = dot(&c.u, &z);
        let scale = c.proj2.sqrt().max(0.05);
        let qn = clip(q / scale, -3.0, 3.0);
        (sigmoid(logit(base) + c.theta * qn), qn)
    }

    fn prediction_tensor(&self, m: &[f64], x: &[f64]) -> f64 {
        if self.prototypes.is_empty() { return 0.5; }
        if m.is_empty() {
            return self.prototype_tensor_prediction(self.current_id.unwrap_or(0), x).0;
        }
        let mut active: Vec<usize> = m.iter().enumerate().filter_map(|(i, &v)| if v >= self.cfg.alpha_cut { Some(i) } else { None }).collect();
        if active.is_empty() {
            active.push(m.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)).map(|x| x.0).unwrap_or(0));
        }
        let z: f64 = active.iter().map(|&i| m[i]).sum::<f64>().max(EPS);
        active.into_iter().map(|i| (m[i] / z) * self.prototype_tensor_prediction(i, x).0).sum()
    }

    fn reset_pending(&mut self) {
        self.pending_kind = None; self.pending_id = None; self.pending_count = 0; self.pending_score = 0.0;
        self.pending_sum.fill(0.0);
    }

    fn set_candidate(&mut self, kind: &str, pid: Option<usize>) {
        if self.pending_kind.as_deref() != Some(kind) || self.pending_id != pid {
            self.pending_kind = Some(kind.to_owned()); self.pending_id = pid; self.pending_count = 0;
            self.pending_score = 0.0; self.pending_sum.fill(0.0);
        }
    }

    fn refresh_direction(&mut self, i: usize) {
        let d = self.cfg.dim;
        let s = self.cfg.tensor_rates.len();
        let cell = &mut self.tensor_cells[i];
        let fro2: f64 = cell.tensor.iter().map(|v| v * v).sum();
        if fro2.sqrt() < 1.0e-12 {
            cell.coherence = 0.0; cell.strength = 0.0; cell.ready = false; return;
        }
        let old_ready = cell.ready;
        let old_u = cell.u.clone();
        let mut u = old_u.clone();
        if norm(&u) < 1.0e-12 { u.fill(1.0 / (d as f64).sqrt()); }
        for _ in 0..2 {
            // v = M^T u ; M is d x (s*2)
            let cols = s * 2;
            let mut vv = vec![0.0; cols];
            for f in 0..d { for col in 0..cols { vv[col] += cell.tensor[f * cols + col] * u[f]; } }
            let mut un = vec![0.0; d];
            for f in 0..d { for col in 0..cols { un[f] += cell.tensor[f * cols + col] * vv[col]; } }
            let n = norm(&un); if n < 1.0e-12 { break; }
            for f in 0..d { u[f] = un[f] / n; }
        }
        if dot(&u, &old_u) < 0.0 { for v in &mut u { *v = -*v; } }
        let cols = s * 2;
        let mut mtu = vec![0.0; cols];
        for f in 0..d { for col in 0..cols { mtu[col] += cell.tensor[f * cols + col] * u[f]; } }
        let s1 = norm(&mtu);
        cell.u = u; cell.strength = s1; cell.coherence = s1 * s1 / (fro2 + 1.0e-12);
        cell.ready = cell.obs >= self.cfg.tensor_min_obs && cell.coherence >= self.cfg.tensor_coherence && cell.strength >= self.cfg.tensor_strength;
        if cell.ready && !old_ready { self.tensor_ready_events += 1; }
    }

    fn schedule_gate(&mut self, i: usize) {
        let c = &mut self.tensor_cells[i];
        if c.burst_left > 0 { return; }
        c.since_burst += 1;
        if !c.ready {
            if c.since_burst >= (self.cfg.refresh_interval / 2).max(12) {
                c.burst_left = self.cfg.refresh_burst; c.since_burst = 0; c.wakes += 1; self.tensor_wakes += 1;
            }
            return;
        }
        let ratio = c.corr_loss / c.base_loss.max(1.0e-9);
        self.bad_counts[i] = if ratio >= self.cfg.degradation_ratio { self.bad_counts[i] + 1 } else { 0 };
        if self.bad_counts[i] >= self.cfg.degradation_patience || c.since_burst >= self.cfg.refresh_interval {
            c.burst_left = self.cfg.refresh_burst; c.since_burst = 0; c.wakes += 1; self.tensor_wakes += 1; self.bad_counts[i] = 0;
        }
    }

    fn certified_recompress(&mut self) -> bool {
        let n = self.prototypes.len();
        if n < 2 || (self.t as isize - self.last_recompress_check) < self.cfg.recompress_interval as isize { return false; }
        self.last_recompress_check = self.t as isize;
        let mut best = (usize::MAX, usize::MAX, f64::INFINITY);
        for i in 0..n { for j in (i + 1)..n {
            let d = rms(&self.prototypes[i].centroid, &self.prototypes[j].centroid);
            if d < best.2 { best = (i, j, d); }
        }}
        let (i, j, dist) = best;
        if i == usize::MAX || !dist.is_finite() || dist > self.cfg.redundancy_tolerance { return false; }
        let (wa, wb) = (self.prototypes[i].count, self.prototypes[j].count);
        let w = (wa + wb).max(EPS);
        let centroid: Vec<f64> = self.prototypes[i].centroid.iter().zip(&self.prototypes[j].centroid).map(|(a, b)| (wa * a + wb * b) / w).collect();
        if rms(&centroid, &self.prototypes[i].centroid).max(rms(&centroid, &self.prototypes[j].centroid)) > self.cfg.redundancy_tolerance { return false; }
        let b = self.prototypes[j].clone();
        {
            let a = &mut self.prototypes[i];
            a.centroid = centroid; a.count = wa + wb; a.successes += b.successes; a.failures += b.failures;
            a.utility += b.utility; a.last_used = a.last_used.max(b.last_used);
        }
        let cb = self.tensor_cells[j].clone();
        {
            let ca = &mut self.tensor_cells[i];
            for k in 0..ca.tensor.len() { ca.tensor[k] = (wa * ca.tensor[k] + wb * cb.tensor[k]) / w; }
            ca.obs += cb.obs; ca.theta = 0.0; ca.ready = false; ca.burst_left = self.cfg.initial_burst; ca.since_burst = 0;
            ca.curvature.fill(0.0); ca.curvature_obs = 0; ca.rank_persist = 0; ca.rank_dirs.clear();
        }
        self.prototypes.remove(j); self.tensor_cells.remove(j); self.bad_counts.remove(j);
        if let Some(cur) = self.current_id {
            self.current_id = Some(if cur == j { i } else if cur > j { cur - 1 } else { cur });
        }
        self.recompression_count += 1; self.purge_dead_candidates();
        true
    }

    fn update_tensor_outcome(&mut self, i: usize, x: &[f64], yy: f64) {
        let base = self.prototypes[i].mean();
        let z: Vec<f64> = x.iter().zip(&self.prototypes[i].centroid).map(|(a, b)| a - b).collect();
        let e = yy - base;
        let (q0, proj20, ready0, theta0) = {
            let c = &self.tensor_cells[i]; (dot(&c.u, &z), c.proj2, c.ready, c.theta)
        };
        let scale0 = proj20.sqrt().max(0.05);
        let qn0 = clip(q0 / scale0, -3.0, 3.0);
        let ph0 = if ready0 { sigmoid(logit(base) + theta0 * qn0) } else { base };
        let r = self.cfg.loss_lr;
        {
            let c = &mut self.tensor_cells[i];
            c.base_loss = (1.0 - r) * c.base_loss + r * (yy - base).powi(2);
            c.corr_loss = (1.0 - r) * c.corr_loss + r * (yy - ph0).powi(2);
            c.obs += 1;
        }
        self.schedule_gate(i);

        let mut needs_refresh = false;
        let mut became_sleep = false;
        {
            let c = &mut self.tensor_cells[i];
            if c.burst_left > 0 {
                let s = self.cfg.tensor_rates.len();
                for (si, &rate) in self.cfg.tensor_rates.iter().enumerate() {
                    for f in 0..self.cfg.dim {
                        let idx0 = (f * s + si) * 2;
                        c.tensor[idx0] = (1.0 - rate) * c.tensor[idx0] + rate * (e * z[f]);
                        c.tensor[idx0 + 1] = (1.0 - rate) * c.tensor[idx0 + 1] + rate * (e * z[f] * z[f].abs());
                    }
                }
                c.tensor_steps += 1; self.tensor_updates += 1; self.tensor_awake_steps += 1; c.burst_left -= 1;
                needs_refresh = c.tensor_steps >= self.cfg.tensor_warmup && (c.tensor_steps % self.cfg.refresh_every_active == 0 || !c.ready);
                if c.burst_left == 0 { c.sleeps += 1; self.tensor_sleeps += 1; became_sleep = true; }
            } else {
                self.tensor_sleep_steps += 1;
            }
        }
        if needs_refresh {
            let was_ready = self.tensor_cells[i].ready;
            self.refresh_direction(i);
            let c = &mut self.tensor_cells[i]; c.refreshes += 1; self.tensor_refreshes += 1;
            if c.ready && !was_ready { c.burst_left = c.burst_left.max(self.cfg.post_ready_burst); }
        }
        let _ = became_sleep;

        // Cheap scalar head always stays plastic.
        let q = dot(&self.tensor_cells[i].u, &z);
        {
            let c = &mut self.tensor_cells[i];
            c.proj2 = 0.96 * c.proj2 + 0.04 * q * q;
            if c.ready {
                let scale = c.proj2.sqrt().max(0.05);
                let qn = clip(q / scale, -3.0, 3.0);
                let ph = sigmoid(logit(base) + c.theta * qn);
                let grad = (ph - yy) * qn + self.cfg.tensor_l2 * c.theta;
                c.theta = clip(c.theta - self.cfg.tensor_lr * grad, -self.cfg.theta_cap, self.cfg.theta_cap);
            }
        }
        let p = &mut self.prototypes[i]; p.successes += yy; p.failures += 1.0 - yy; p.utility += 1.0; p.last_used = self.t;
    }

    /// Frozen v0.9.3 structural + tensor step. v0.9.5 refinement runs afterwards.
    fn step_base(&mut self, x: &[f64], y: Option<f64>) -> Read {
        self.t += 1;
        for p in &mut self.prototypes { p.utility *= self.cfg.decay; }
        let mut revision = false; let mut reactivated = false; let mut discovered = false;
        let mut unresolved = false; let mut budget_pressure = false;
        if self.prototypes.is_empty() {
            self.current_id = Some(self.append(x, self.t)); discovered = true; revision = true;
        }
        let mut d = self.distances(x);
        let mut m = self.memberships(&d);
        let mut nearest = d.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)).map(|x| x.0);
        let mut nd = nearest.map(|i| d[i]).unwrap_or(f64::INFINITY);
        let pred = self.prediction_tensor(&m, x);
        let cur = self.current_id;
        let curd = cur.and_then(|i| d.get(i).copied()).unwrap_or(f64::INFINITY);
        let sigma = self.sigma2.sqrt().max(self.cfg.min_sigma);
        let denom = (self.cfg.recurrence_tolerance - self.cfg.stay_tolerance).max(1.0e-6);
        let surprise = clip((curd - self.cfg.stay_tolerance) / denom, 0.0, 1.0);
        self.hazard = (1.0 - self.cfg.hazard_lr) * self.hazard + self.cfg.hazard_lr * surprise;

        if let Some(ci) = cur.filter(|_| curd <= self.cfg.stay_tolerance) {
            self.reset_pending();
            let capped = curd.min(self.cfg.stay_tolerance);
            self.sigma2 = (1.0 - self.cfg.noise_lr) * self.sigma2 + self.cfg.noise_lr * capped * capped;
            if curd <= self.cfg.update_tolerance {
                let conf = (1.0 - curd / self.cfg.update_tolerance.max(1.0e-6)).max(0.0);
                let eta = self.cfg.centroid_lr_cap.min(1.0 / (self.prototypes[ci].count + 1.0).max(2.0)) * (0.5 + 0.5 * conf);
                for f in 0..self.cfg.dim { self.prototypes[ci].centroid[f] = (1.0 - eta) * self.prototypes[ci].centroid[f] + eta * x[f]; }
                self.prototypes[ci].count += 1.0;
            }
        } else {
            let cand = nearest.filter(|&i| Some(i) != cur && nd <= self.cfg.recurrence_tolerance);
            if let Some(cand) = cand {
                self.set_candidate("reactivate", Some(cand)); self.pending_count += 1;
                for f in 0..self.cfg.dim { self.pending_sum[f] += x[f]; }
                let llr = clip((curd * curd - nd * nd) / (2.0 * sigma * sigma), 0.0, 6.0);
                self.pending_score = self.cfg.evidence_decay * self.pending_score + llr;
                let theta = 0.55f64.max(self.cfg.recurrence_threshold * (1.0 - self.cfg.hazard_discount_rec * self.hazard));
                if self.pending_score >= theta {
                    self.current_id = Some(cand); revision = true; reactivated = true;
                    self.revision_count += 1; self.reactivation_count += 1; self.reset_pending();
                }
            } else {
                self.set_candidate("novel", None); self.pending_count += 1;
                for f in 0..self.cfg.dim { self.pending_sum[f] += x[f]; }
                let all_far = (nd - self.cfg.recurrence_tolerance).max(0.0);
                let cur_bad = (curd - self.cfg.stay_tolerance).max(0.0);
                let novelty_llr = clip((all_far + 0.5 * cur_bad) / sigma, 0.0, 4.0);
                self.pending_score = self.cfg.evidence_decay * self.pending_score + novelty_llr;
                let theta = 1.75f64.max(self.cfg.novelty_threshold * (1.0 - self.cfg.hazard_discount_nov * self.hazard));
                if self.pending_score >= theta {
                    let denom = self.pending_count.max(1) as f64;
                    let meanx: Vec<f64> = self.pending_sum.iter().map(|v| v / denom).collect();
                    if self.prototypes.len() < self.cfg.budget || self.certified_recompress() {
                        self.current_id = Some(self.append(&meanx, self.t)); revision = true; discovered = true;
                        self.revision_count += 1; self.reset_pending();
                    } else {
                        unresolved = true; budget_pressure = true;
                        self.unresolved_count += 1; self.budget_pressure_count += 1;
                    }
                }
            }
        }

        if let (Some(yy), Some(i)) = (y, self.current_id) { self.update_tensor_outcome(i, x, yy); }

        if revision || discovered {
            d = self.distances(x); m = self.memberships(&d);
            nearest = d.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)).map(|x| x.0);
            nd = nearest.map(|i| d[i]).unwrap_or(f64::INFINITY);
        }
        let active = self.active_ids(&m, nearest);
        Read { prediction: pred, memberships: m, active_ids: active, current_id: self.current_id,
            nearest_id: nearest, nearest_dist: nd, revision, reactivated, discovered,
            unresolved, budget_pressure, stored: self.prototypes.len(), comparisons: d.len() }
    }

    fn update_curvature(&mut self, i: usize, x: &[f64], y: f64, prediction: f64, centroid_before: &[f64]) {
        let z: Vec<f64> = x.iter().zip(centroid_before).map(|(a, b)| a - b).collect();
        let e = y - prediction;
        let stride = self.cfg.curvature_stride.max(1);
        let r = 1.0 - (1.0 - self.cfg.curvature_lr).powi(stride as i32);
        let c = &mut self.tensor_cells[i];
        for row in 0..self.cfg.dim { for col in 0..self.cfg.dim {
            let k = row * self.cfg.dim + col;
            c.curvature[k] = (1.0 - r) * c.curvature[k] + r * e * z[row] * z[col];
        }}
        c.curvature_obs += stride; self.curvature_updates += 1;
    }

    fn rank_certificate(&mut self, i: usize) -> bool {
        if self.tensor_cells[i].curvature_obs < self.cfg.split_min_parent_obs { return false; }
        let d = self.cfg.dim;
        let q = self.tensor_cells[i].curvature.clone();
        let (vals, vecs) = jacobi_eigen_symmetric(&q, d, 12 * d.max(1), 1.0e-12);
        let mut order: Vec<usize> = (0..d).collect();
        order.sort_by(|&a, &b| vals[b].abs().partial_cmp(&vals[a].abs()).unwrap_or(Ordering::Equal));
        let s1 = order.first().map(|&j| vals[j].abs()).unwrap_or(0.0);
        let s2 = order.get(1).map(|&j| vals[j].abs()).unwrap_or(0.0);
        let ratio = s2 / s1.max(1.0e-12);
        self.rank_checks += 1;
        let mut dirs = Vec::new();
        for &j in order.iter().take(2) { dirs.push(canonicalize_direction(column(&vecs, d, d, j))); }
        if order.len() >= 2 {
            let u1 = column(&vecs, d, d, order[0]); let u2 = column(&vecs, d, d, order[1]);
            let mut axes: Vec<(usize, f64)> = (0..d).map(|k| (k, (u1[k] * u1[k] + u2[k] * u2[k]).sqrt())).collect();
            axes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            for &(axis, _) in axes.iter().take(2) { let mut e = vec![0.0; d]; e[axis] = 1.0; dirs.push(e); }
        }
        let mut unique: Vec<Vec<f64>> = Vec::new();
        'outer: for u in dirs {
            if norm(&u) <= 1.0e-12 { continue; }
            for v in &unique { if dot(&u, v).abs() >= 0.96 { continue 'outer; } }
            unique.push(u);
        }
        let c = &mut self.tensor_cells[i];
        c.rank_strength1 = s1; c.rank_strength2 = s2; c.rank_ratio2 = ratio;
        if s1 >= self.cfg.rank_strength_threshold && ratio >= self.cfg.rank_ratio_threshold { c.rank_persist += 1; }
        else { c.rank_persist = c.rank_persist.saturating_sub(1); }
        c.rank_dirs = unique;
        c.rank_persist >= self.cfg.rank_patience && !c.rank_dirs.is_empty()
    }

    fn maybe_launch_split(&mut self, i: usize) {
        if i >= self.prototypes.len() { return; }
        let p = &self.prototypes[i]; let uid = p.uid;
        if p.refinement_depth >= self.cfg.max_refinement_depth || self.split_candidates.contains_key(&uid) { return; }
        if self.t < *self.split_cooldown_until.get(&uid).unwrap_or(&0) { return; }
        let refreshes = self.tensor_cells[i].refreshes as isize;
        if refreshes == self.tensor_cells[i].last_rank_refresh_seen { return; }
        self.tensor_cells[i].last_rank_refresh_seen = refreshes;
        if !self.rank_certificate(i) || self.tensor_cells[i].obs < self.cfg.split_min_parent_obs { return; }
        let directions = self.tensor_cells[i].rank_dirs.iter().cloned().map(|direction| DirectionShadow {
            direction, bias_neg: 0.0, bias_pos: 0.0, parent_loss: 0.0, child_loss: 0.0, n: 0
        }).collect();
        self.split_candidates.insert(uid, SplitCandidate {
            parent_uid: uid, started_t: self.t, directions,
            selection_target: self.cfg.split_selection_obs, validation_target: self.cfg.split_validation_obs,
            max_validation: self.cfg.split_max_validation, phase: CandidatePhase::Select, selected: None,
            validation_n: 0, validation_parent_loss: 0.0, validation_child_loss: 0.0,
            bias_neg: 0.0, bias_pos: 0.0, neg_n: 0, pos_n: 0, neg_success: 0.0, pos_success: 0.0,
            neg_sum: vec![0.0; self.cfg.dim], pos_sum: vec![0.0; self.cfg.dim],
        });
        self.split_proposals += 1;
    }

    fn shadow_update_candidate(&mut self, i: usize, x: &[f64], y: f64, parent_pred: f64) -> Option<ShadowAction> {
        if i >= self.prototypes.len() { return None; }
        let uid = self.prototypes[i].uid;
        let centroid = self.prototypes[i].centroid.clone();
        let depth = self.prototypes[i].refinement_depth;
        let cand = self.split_candidates.get_mut(&uid)?;
        match cand.phase {
            CandidatePhase::Select => {
                for sh in &mut cand.directions {
                    let pos = dot(&sh.direction, &x.iter().zip(&centroid).map(|(a,b)| a-b).collect::<Vec<_>>()) >= 0.0;
                    let bias = if pos { sh.bias_pos } else { sh.bias_neg };
                    let child = sigmoid(logit(parent_pred) + bias);
                    sh.parent_loss += (y - parent_pred).powi(2); sh.child_loss += (y - child).powi(2); sh.n += 1;
                    let grad = child - y;
                    if pos { sh.bias_pos = clip(sh.bias_pos - self.cfg.shadow_bias_lr * grad, -self.cfg.shadow_bias_cap, self.cfg.shadow_bias_cap); }
                    else { sh.bias_neg = clip(sh.bias_neg - self.cfg.shadow_bias_lr * grad, -self.cfg.shadow_bias_cap, self.cfg.shadow_bias_cap); }
                }
                if !cand.directions.is_empty() && cand.directions.iter().map(|s| s.n).min().unwrap_or(0) >= cand.selection_target {
                    cand.selected = cand.directions.iter().enumerate().max_by(|(_,a),(_,b)| a.mean_gain().partial_cmp(&b.mean_gain()).unwrap_or(Ordering::Equal)).map(|x| x.0);
                    cand.phase = CandidatePhase::Validate; cand.validation_n = 0; cand.validation_parent_loss = 0.0; cand.validation_child_loss = 0.0;
                    cand.bias_neg = 0.0; cand.bias_pos = 0.0; cand.neg_n = 0; cand.pos_n = 0; cand.neg_success = 0.0; cand.pos_success = 0.0;
                    cand.neg_sum.fill(0.0); cand.pos_sum.fill(0.0);
                }
                None
            }
            CandidatePhase::Validate => {
                let sel = cand.selected?;
                let direction = &cand.directions[sel].direction;
                let delta: Vec<f64> = x.iter().zip(&centroid).map(|(a,b)| a-b).collect();
                let pos = dot(direction, &delta) >= 0.0;
                let bias = if pos { cand.bias_pos } else { cand.bias_neg };
                let child = sigmoid(logit(parent_pred) + bias);
                cand.validation_parent_loss += (y - parent_pred).powi(2); cand.validation_child_loss += (y - child).powi(2); cand.validation_n += 1;
                let grad = child - y;
                if pos {
                    cand.bias_pos = clip(cand.bias_pos - self.cfg.shadow_bias_lr * grad, -self.cfg.shadow_bias_cap, self.cfg.shadow_bias_cap);
                    cand.pos_n += 1; cand.pos_success += y; for f in 0..self.cfg.dim { cand.pos_sum[f] += x[f]; }
                } else {
                    cand.bias_neg = clip(cand.bias_neg - self.cfg.shadow_bias_lr * grad, -self.cfg.shadow_bias_cap, self.cfg.shadow_bias_cap);
                    cand.neg_n += 1; cand.neg_success += y; for f in 0..self.cfg.dim { cand.neg_sum[f] += x[f]; }
                }
                if cand.validation_n < cand.validation_target { return None; }
                let depth_factor = 1.0 + self.cfg.recursive_split_penalty * depth as f64;
                let complexity = depth_factor * self.cfg.split_complexity_penalty * ((cand.validation_n as f64) + 1.0).ln();
                let evidence = cand.validation_gain() - complexity;
                let supported = cand.neg_n >= self.cfg.split_min_branch && cand.pos_n >= self.cfg.split_min_branch;
                if supported && cand.validation_mean_gain() >= depth_factor * self.cfg.split_mean_gain && evidence >= depth_factor * self.cfg.split_accept_margin {
                    Some(ShadowAction::Promote)
                } else if cand.validation_n >= cand.max_validation { Some(ShadowAction::Reject) } else { None }
            }
        }
    }

    fn promote_split(&mut self, i: usize, cand: SplitCandidate, x_now: &[f64]) -> bool {
        if i >= self.prototypes.len() { return false; }
        let uid = self.prototypes[i].uid;
        if self.prototypes.len() >= self.cfg.budget {
            self.split_budget_blocks += 1; self.split_rejections += 1;
            self.split_cooldown_until.insert(uid, self.t + self.cfg.split_cooldown); self.split_candidates.remove(&uid); return false;
        }
        if cand.neg_n < self.cfg.split_min_branch || cand.pos_n < self.cfg.split_min_branch { return false; }
        let sel = match cand.selected { Some(v) => v, None => return false };
        let parent = self.prototypes[i].clone(); let parent_tensor = self.tensor_cells[i].clone();
        let u = cand.directions[sel].direction.clone();
        let neg_centroid: Vec<f64> = cand.neg_sum.iter().map(|v| v / cand.neg_n.max(1) as f64).collect();
        let pos_centroid: Vec<f64> = cand.pos_sum.iter().map(|v| v / cand.pos_n.max(1) as f64).collect();
        if rms(&neg_centroid, &pos_centroid) < 0.5 * self.cfg.redundancy_tolerance {
            self.split_rejections += 1; self.split_cooldown_until.insert(uid, self.t + self.cfg.split_cooldown); self.split_candidates.remove(&uid); return false;
        }
        let lineage = self.next_lineage; self.next_lineage += 1;
        let prior = self.cfg.prior_strength / 2.0;
        let depth = parent.refinement_depth + 1;
        let neg_uid = self.next_uid; self.next_uid += 1; let pos_uid = self.next_uid; self.next_uid += 1;
        let mut neg = Prototype { uid: neg_uid, centroid: neg_centroid, count: cand.neg_n as f64,
            successes: prior + cand.neg_success, failures: prior + (cand.neg_n as f64 - cand.neg_success),
            utility: (parent.utility * cand.neg_n as f64 / (cand.neg_n + cand.pos_n).max(1) as f64).max(1.0),
            last_used: self.t, created_t: self.t, lineage_id: Some(lineage), split_created_t: Some(self.t),
            recent_mean: 0.5, recent_obs: 0, refinement_depth: depth };
        neg.recent_mean = neg.mean();
        let mut pos = Prototype { uid: pos_uid, centroid: pos_centroid, count: cand.pos_n as f64,
            successes: prior + cand.pos_success, failures: prior + (cand.pos_n as f64 - cand.pos_success),
            utility: (parent.utility * cand.pos_n as f64 / (cand.neg_n + cand.pos_n).max(1) as f64).max(1.0),
            last_used: self.t, created_t: self.t, lineage_id: Some(lineage), split_created_t: Some(self.t),
            recent_mean: 0.5, recent_obs: 0, refinement_depth: depth };
        pos.recent_mean = pos.mean();
        self.prototypes[i] = neg; self.tensor_cells[i] = self.inherit_tensor_cell(&parent_tensor); self.bad_counts[i] = 0;
        let j = self.prototypes.len(); self.prototypes.push(pos); self.tensor_cells.push(self.inherit_tensor_cell(&parent_tensor)); self.bad_counts.push(0);
        let delta: Vec<f64> = x_now.iter().zip(&parent.centroid).map(|(a,b)| a-b).collect();
        self.current_id = Some(if dot(&u, &delta) >= 0.0 { j } else { i });
        self.revision_count += 1; self.split_promotions += 1; self.split_candidates.remove(&uid); self.split_cooldown_until.remove(&uid); self.lineage_active = true;
        true
    }

    fn merge_pair(&mut self, i: usize, j: usize) {
        let b = self.prototypes[j].clone();
        let (wa, wb) = (self.prototypes[i].count.max(1.0), b.count.max(1.0)); let w = wa + wb;
        let centroid: Vec<f64> = self.prototypes[i].centroid.iter().zip(&b.centroid).map(|(a,bb)| (wa*a + wb*bb)/w).collect();
        let depth = self.prototypes[i].refinement_depth.max(b.refinement_depth);
        {
            let a = &mut self.prototypes[i]; a.centroid = centroid; a.count = w; a.successes += b.successes; a.failures += b.failures;
            a.utility += b.utility; a.last_used = a.last_used.max(b.last_used); a.lineage_id = None; a.split_created_t = None;
            a.refinement_depth = depth.saturating_sub(1); a.recent_obs = 0; a.recent_mean = a.mean();
        }
        self.tensor_cells[i] = self.fresh_tensor_cell(); self.bad_counts[i] = 0;
        self.prototypes.remove(j); self.tensor_cells.remove(j); self.bad_counts.remove(j);
        if let Some(cur) = self.current_id { self.current_id = Some(if cur == j { i } else if cur > j { cur - 1 } else { cur }); }
        self.merge_promotions += 1; self.recompression_count += 1; self.purge_dead_candidates();
        self.lineage_active = self.prototypes.iter().any(|p| p.lineage_id.is_some());
    }

    fn maybe_merge_lineage(&mut self) -> bool {
        if !self.lineage_active || (self.t as isize - self.last_merge_check) < self.cfg.merge_interval as isize { return false; }
        self.last_merge_check = self.t as isize;
        let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, p) in self.prototypes.iter().enumerate() { if let Some(lid) = p.lineage_id { groups.entry(lid).or_default().push(i); } }
        let live: std::collections::HashSet<u64> = groups.keys().copied().collect();
        self.merge_evidence.retain(|k, _| live.contains(k));
        for (lid, ids) in groups {
            if ids.len() != 2 { self.merge_evidence.insert(lid, 0); continue; }
            let (i, j) = (ids[0], ids[1]); let (a, b) = (&self.prototypes[i], &self.prototypes[j]);
            let split_t = a.split_created_t.unwrap_or(self.t).max(b.split_created_t.unwrap_or(self.t));
            let age = self.t.saturating_sub(split_t);
            let mut cert = age >= self.cfg.merge_min_age && a.count.min(b.count) >= self.cfg.merge_min_count
                && a.recent_obs.min(b.recent_obs) >= self.cfg.merge_recent_min_obs
                && (a.recent_mean - b.recent_mean).abs() <= self.cfg.merge_pred_tolerance;
            let denom = (a.count + b.count).max(EPS);
            let c: Vec<f64> = a.centroid.iter().zip(&b.centroid).map(|(x,y)| (a.count*x + b.count*y)/denom).collect();
            if rms(&c, &a.centroid).max(rms(&c, &b.centroid)) > self.cfg.merge_dist_tolerance { cert = false; }
            let ev = self.merge_evidence.entry(lid).or_insert(0); *ev = if cert { *ev + 1 } else { 0 };
            if *ev >= self.cfg.merge_patience { self.merge_pair(i, j); self.merge_evidence.remove(&lid); return true; }
        }
        false
    }

    fn purge_dead_candidates(&mut self) {
        let alive: std::collections::HashSet<u64> = self.prototypes.iter().map(|p| p.uid).collect();
        self.split_candidates.retain(|k, _| alive.contains(k)); self.split_cooldown_until.retain(|k, _| alive.contains(k));
    }

    pub fn step(&mut self, x: &[f64], y: Option<f64>) -> Result<Read, String> {
        if x.len() != self.cfg.dim { return Err(format!("expected dim {}, got {}", self.cfg.dim, x.len())); }
        let pre_current = self.current_id;
        let pre_centroid = pre_current.and_then(|i| self.prototypes.get(i).map(|p| p.centroid.clone()));
        let pre_tensor_steps = pre_current.and_then(|i| self.tensor_cells.get(i).map(|c| c.tensor_steps)).unwrap_or(0);
        let mut read = self.step_base(x, y);
        let mut structural_edit = false;

        if let (Some(yy), Some(i)) = (y, self.current_id) {
            if let Some(lid) = self.prototypes[i].lineage_id {
                let _ = lid;
                let rr = self.cfg.merge_recent_lr;
                let current_mean = self.prototypes[i].recent_mean;
                self.prototypes[i].recent_mean = (1.0 - rr) * current_mean + rr * yy;
                self.prototypes[i].recent_obs += 1; self.lineage_active = true;
            }
            let tensor_advanced = self.tensor_cells[i].tensor_steps > pre_tensor_steps;
            let mut curvature_updated = false;
            if pre_current == Some(i) && pre_centroid.is_some() && tensor_advanced && !read.revision
                && self.tensor_cells[i].tensor_steps % self.cfg.curvature_stride.max(1) == 0 {
                self.update_curvature(i, x, yy, read.prediction, pre_centroid.as_ref().unwrap()); curvature_updated = true;
            }
            if curvature_updated { self.maybe_launch_split(i); }
            let uid = self.prototypes.get(i).map(|p| p.uid);
            let action = if uid.and_then(|u| self.split_candidates.get(&u)).is_some() { self.shadow_update_candidate(i, x, yy, read.prediction) } else { None };
            match action {
                Some(ShadowAction::Promote) => {
                    if let Some(u) = uid { if let Some(cand) = self.split_candidates.get(&u).cloned() { structural_edit = self.promote_split(i, cand, x); } }
                }
                Some(ShadowAction::Reject) => {
                    if let Some(u) = uid { self.split_rejections += 1; self.split_candidates.remove(&u); self.split_cooldown_until.insert(u, self.t + self.cfg.split_cooldown); }
                }
                None => {}
            }
        }
        if self.maybe_merge_lineage() { structural_edit = true; }
        self.purge_dead_candidates();

        if structural_edit {
            let d = self.distances(x); let m = self.memberships(&d);
            let nearest = d.iter().enumerate().min_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap_or(Ordering::Equal)).map(|x| x.0);
            let nd = nearest.map(|i| d[i]).unwrap_or(f64::INFINITY);
            read.memberships = m.clone(); read.active_ids = self.active_ids(&m, nearest); read.current_id = self.current_id;
            read.nearest_id = nearest; read.nearest_dist = nd; read.revision = true; read.stored = self.prototypes.len(); read.comparisons = d.len();
        }
        Ok(read)
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            t: self.t, current_id: self.current_id, pending_kind: self.pending_kind.clone(), pending_id: self.pending_id,
            pending_count: self.pending_count, pending_score: self.pending_score, pending_sum: self.pending_sum.clone(),
            sigma2: self.sigma2, hazard: self.hazard, prototypes: self.prototypes.clone(), tensor_cells: self.tensor_cells.clone(),
            revision_count: self.revision_count, reactivation_count: self.reactivation_count, discovery_count: self.discovery_count,
            unresolved_count: self.unresolved_count, budget_pressure_count: self.budget_pressure_count, recompression_count: self.recompression_count,
            split_proposals: self.split_proposals, split_rejections: self.split_rejections, split_promotions: self.split_promotions,
            merge_promotions: self.merge_promotions, split_budget_blocks: self.split_budget_blocks, rank_checks: self.rank_checks,
            curvature_updates: self.curvature_updates, tensor_updates: self.tensor_updates, tensor_awake_steps: self.tensor_awake_steps,
            tensor_sleep_steps: self.tensor_sleep_steps, tensor_refreshes: self.tensor_refreshes, tensor_wakes: self.tensor_wakes,
            tensor_sleeps: self.tensor_sleeps, split_candidates: self.split_candidates.clone(),
        }
    }
}
