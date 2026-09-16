//! Native bounded fuzzy memory for Cortex.
//!
//! This crate ports the v0.8 Fuzzy Accordion Memory contract: graded
//! membership, alpha-cut activation, certified recompression, and explicit
//! unresolved/budget-pressure behavior when compression would exceed the
//! declared distortion budget.

use serde::{Deserialize, Serialize};

const EPS: f64 = 1.0e-12;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub dim: usize,
    pub budget: usize,
    pub tau: f64,
    pub alpha: f64,
    pub fit_tolerance: f64,
    pub distortion_budget: f64,
    pub redundancy_tolerance: f64,
    pub decay: f64,
    pub recompress_interval: usize,
}

impl MemoryConfig {
    pub fn for_dim(dim: usize) -> Self {
        Self {
            dim,
            budget: 32,
            tau: 0.08,
            alpha: 0.12,
            fit_tolerance: 0.035,
            distortion_budget: 0.08,
            redundancy_tolerance: 0.045,
            decay: 0.995,
            recompress_interval: 16,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryRead {
    pub reconstruction: Vec<f64>,
    pub memberships: Vec<f64>,
    pub active_ids: Vec<usize>,
    pub nearest_id: Option<usize>,
    pub nearest_dist: f64,
    pub stored: usize,
    pub unresolved: bool,
    pub compressed: bool,
    pub structural_change: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub prototypes: Vec<Vec<f64>>,
    pub counts: Vec<f64>,
    pub utility: Vec<f64>,
    pub last_used: Vec<usize>,
    pub time: usize,
    pub unresolved_count: usize,
    pub recompression_count: usize,
    pub spawn_count: usize,
    pub prototype_updates: usize,
}

#[derive(Clone, Debug)]
pub struct FuzzyAccordionMemory {
    pub cfg: MemoryConfig,
    prototypes: Vec<Vec<f64>>,
    counts: Vec<f64>,
    utility: Vec<f64>,
    last_used: Vec<usize>,
    time: usize,
    last_recompress_check: isize,
    unresolved_count: usize,
    recompression_count: usize,
    spawn_count: usize,
    prototype_updates: usize,
}

impl FuzzyAccordionMemory {
    pub fn new(mut cfg: MemoryConfig) -> Result<Self, String> {
        if cfg.dim == 0 {
            return Err("dim must be positive".into());
        }
        if cfg.budget == 0 {
            return Err("budget must be positive".into());
        }
        if !cfg.tau.is_finite() || cfg.tau <= 0.0 {
            return Err("tau must be finite and positive".into());
        }
        if !cfg.alpha.is_finite() || !(0.0..=1.0).contains(&cfg.alpha) {
            return Err("alpha must lie in [0,1]".into());
        }
        if !cfg.decay.is_finite() || cfg.decay < 0.0 {
            return Err("decay must be finite and non-negative".into());
        }
        cfg.recompress_interval = cfg.recompress_interval.max(1);
        Ok(Self {
            cfg,
            prototypes: Vec::new(),
            counts: Vec::new(),
            utility: Vec::new(),
            last_used: Vec::new(),
            time: 0,
            last_recompress_check: -1_000_000_000,
            unresolved_count: 0,
            recompression_count: 0,
            spawn_count: 0,
            prototype_updates: 0,
        })
    }

    pub fn stored(&self) -> usize {
        self.prototypes.len()
    }

    pub fn snapshot(&self) -> MemorySnapshot {
        MemorySnapshot {
            prototypes: self.prototypes.clone(),
            counts: self.counts.clone(),
            utility: self.utility.clone(),
            last_used: self.last_used.clone(),
            time: self.time,
            unresolved_count: self.unresolved_count,
            recompression_count: self.recompression_count,
            spawn_count: self.spawn_count,
            prototype_updates: self.prototype_updates,
        }
    }

    pub fn memberships(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        self.validate(x)?;
        Ok(self.memberships_from_distances(&self.distances(x)))
    }

    pub fn reconstruct(&self, memberships: &[f64]) -> Result<Vec<f64>, String> {
        if self.prototypes.is_empty() {
            return Ok(vec![0.0; self.cfg.dim]);
        }
        if memberships.len() != self.prototypes.len() {
            return Err(format!(
                "expected {} memberships, got {}",
                self.prototypes.len(),
                memberships.len()
            ));
        }
        let sum: f64 = memberships.iter().sum();
        if sum <= EPS {
            // Match np.argmax on all-zero memberships: first maximum.
            return Ok(self.prototypes[0].clone());
        }
        let mut out = vec![0.0; self.cfg.dim];
        for (i, &m) in memberships.iter().enumerate() {
            for (f, slot) in out.iter_mut().enumerate() {
                *slot += m * self.prototypes[i][f];
            }
        }
        let denom = sum.max(EPS);
        for v in &mut out {
            *v /= denom;
        }
        Ok(out)
    }

    pub fn step(&mut self, x: &[f64]) -> Result<MemoryRead, String> {
        self.validate(x)?;
        self.time += 1;
        for u in &mut self.utility {
            *u *= self.cfg.decay;
        }

        let mut structural_change = false;
        let mut compressed = false;
        let mut unresolved = false;

        if self.prototypes.is_empty() {
            self.append(x);
            structural_change = true;
        } else {
            let d0 = self.distances(x);
            let nearest = first_argmin(&d0);
            let dn = d0[nearest];
            if dn <= self.cfg.fit_tolerance {
                let c = self.counts[nearest];
                for (f, &value) in x.iter().enumerate() {
                    self.prototypes[nearest][f] =
                        (c * self.prototypes[nearest][f] + value) / (c + 1.0);
                }
                self.counts[nearest] = c + 1.0;
                self.prototype_updates += 1;
                compressed = true;
            } else if self.prototypes.len() < self.cfg.budget {
                self.append(x);
                structural_change = true;
            } else if self.certified_recompress() {
                self.append(x);
                structural_change = true;
            } else {
                let m0 = self.memberships_from_distances(&d0);
                let r0 = self.reconstruct(&m0)?;
                if rms_distance(x, &r0) <= self.cfg.distortion_budget {
                    compressed = true;
                } else {
                    unresolved = true;
                    self.unresolved_count += 1;
                }
            }
        }

        let d = self.distances(x);
        let memberships = self.memberships_from_distances(&d);
        let reconstruction = self.reconstruct(&memberships)?;
        let (nearest_id, nearest_dist, mut active_ids) = if memberships.is_empty() {
            (None, f64::INFINITY, Vec::new())
        } else {
            let nearest = first_argmin(&d);
            let mut active: Vec<usize> = memberships
                .iter()
                .enumerate()
                .filter_map(|(i, &m)| (m >= self.cfg.alpha).then_some(i))
                .collect();
            if active.is_empty() {
                active.push(nearest);
            }
            (Some(nearest), d[nearest], active)
        };

        for &i in &active_ids {
            self.utility[i] += memberships[i];
            self.last_used[i] = self.time;
        }

        Ok(MemoryRead {
            reconstruction,
            memberships,
            active_ids: std::mem::take(&mut active_ids),
            nearest_id,
            nearest_dist,
            stored: self.prototypes.len(),
            unresolved,
            compressed,
            structural_change,
        })
    }

    fn validate(&self, x: &[f64]) -> Result<(), String> {
        if x.len() != self.cfg.dim {
            return Err(format!(
                "expected vector of length {}, got {}",
                self.cfg.dim,
                x.len()
            ));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err("memory observation contains non-finite values".into());
        }
        Ok(())
    }

    fn append(&mut self, x: &[f64]) -> usize {
        self.prototypes.push(x.to_vec());
        self.counts.push(1.0);
        self.utility.push(1.0);
        self.last_used.push(self.time);
        self.spawn_count += 1;
        self.prototypes.len() - 1
    }

    fn distances(&self, x: &[f64]) -> Vec<f64> {
        self.prototypes
            .iter()
            .map(|c| rms_distance(x, c))
            .collect()
    }

    fn memberships_from_distances(&self, distances: &[f64]) -> Vec<f64> {
        if distances.is_empty() {
            return Vec::new();
        }
        let mut scores: Vec<f64> = distances
            .iter()
            .map(|&d| (-d / self.cfg.tau.max(EPS)).exp())
            .collect();
        let sum: f64 = scores.iter().sum();
        if sum <= EPS {
            scores.fill(0.0);
            scores[first_argmin(distances)] = 1.0;
            return scores;
        }
        for s in &mut scores {
            *s /= sum;
        }
        scores
    }

    fn certified_recompress(&mut self) -> bool {
        let n = self.prototypes.len();
        if n < 2 {
            return false;
        }
        if (self.time as isize - self.last_recompress_check) < self.cfg.recompress_interval as isize
        {
            return false;
        }
        self.last_recompress_check = self.time as isize;

        let mut best: Option<(usize, usize, f64)> = None;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = rms_distance(&self.prototypes[i], &self.prototypes[j]);
                match best {
                    None => best = Some((i, j, d)),
                    Some((_, _, bd)) if d < bd => best = Some((i, j, d)),
                    _ => {}
                }
            }
        }
        let Some((i, j, best_d)) = best else {
            return false;
        };
        if !best_d.is_finite() || best_d > self.cfg.redundancy_tolerance {
            return false;
        }

        let wi = self.counts[i];
        let wj = self.counts[j];
        let mut merged = vec![0.0; self.cfg.dim];
        for (f, slot) in merged.iter_mut().enumerate() {
            *slot = (wi * self.prototypes[i][f] + wj * self.prototypes[j][f])
                / (wi + wj).max(EPS);
        }
        if rms_distance(&merged, &self.prototypes[i])
            .max(rms_distance(&merged, &self.prototypes[j]))
            > self.cfg.distortion_budget
        {
            return false;
        }

        self.prototypes[i] = merged;
        self.counts[i] = wi + wj;
        self.utility[i] += self.utility[j];
        self.last_used[i] = self.last_used[i].max(self.last_used[j]);
        self.prototypes.remove(j);
        self.counts.remove(j);
        self.utility.remove(j);
        self.last_used.remove(j);
        self.recompression_count += 1;
        true
    }
}

fn rms_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mean = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        / a.len().max(1) as f64;
    mean.sqrt()
}

fn first_argmin(values: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_value = f64::INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v < best_value {
            best = i;
            best_value = v;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearby_observation_updates_in_place() {
        let mut cfg = MemoryConfig::for_dim(3);
        cfg.fit_tolerance = 0.1;
        let mut mem = FuzzyAccordionMemory::new(cfg).unwrap();
        let first = mem.step(&[0.0, 0.0, 0.0]).unwrap();
        assert!(first.structural_change);
        let second = mem.step(&[0.03, 0.0, 0.0]).unwrap();
        assert!(second.compressed);
        assert!(!second.structural_change);
        assert_eq!(second.stored, 1);
    }

    #[test]
    fn budget_pressure_is_explicit_when_no_merge_is_certified() {
        let mut cfg = MemoryConfig::for_dim(2);
        cfg.budget = 2;
        cfg.fit_tolerance = 0.01;
        cfg.redundancy_tolerance = 0.01;
        cfg.distortion_budget = 0.05;
        cfg.recompress_interval = 1;
        let mut mem = FuzzyAccordionMemory::new(cfg).unwrap();
        mem.step(&[0.0, 0.0]).unwrap();
        mem.step(&[1.0, 1.0]).unwrap();
        let out = mem.step(&[3.0, -3.0]).unwrap();
        assert!(out.unresolved);
        assert_eq!(out.stored, 2);
    }

    #[test]
    fn certified_recompression_frees_capacity() {
        let mut cfg = MemoryConfig::for_dim(2);
        cfg.budget = 3;
        cfg.fit_tolerance = 0.001;
        cfg.redundancy_tolerance = 0.1;
        cfg.distortion_budget = 0.1;
        cfg.recompress_interval = 1;
        let mut mem = FuzzyAccordionMemory::new(cfg).unwrap();
        mem.step(&[0.0, 0.0]).unwrap();
        mem.step(&[0.08, 0.0]).unwrap();
        mem.step(&[1.0, 1.0]).unwrap();
        let out = mem.step(&[-1.0, -1.0]).unwrap();
        assert!(out.structural_change);
        assert_eq!(out.stored, 3);
        assert_eq!(mem.snapshot().recompression_count, 1);
    }
}
