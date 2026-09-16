use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub dim: usize,
    pub budget: usize,
    pub tau: f64,
    pub alpha_cut: f64,
    pub stay_tolerance: f64,
    pub recurrence_tolerance: f64,
    pub update_tolerance: f64,
    pub redundancy_tolerance: f64,
    pub centroid_lr_cap: f64,
    pub decay: f64,
    pub prior_strength: f64,
    pub recompress_interval: usize,
    pub evidence_decay: f64,
    pub noise_lr: f64,
    pub hazard_lr: f64,
    pub recurrence_threshold: f64,
    pub novelty_threshold: f64,
    pub hazard_discount_rec: f64,
    pub hazard_discount_nov: f64,
    pub min_sigma: f64,
    pub tensor_rates: Vec<f64>,
    pub tensor_min_obs: usize,
    pub tensor_coherence: f64,
    pub tensor_strength: f64,
    pub tensor_lr: f64,
    pub tensor_l2: f64,
    pub theta_cap: f64,
    pub tensor_warmup: usize,
    pub loss_lr: f64,
    pub initial_burst: usize,
    pub post_ready_burst: usize,
    pub refresh_interval: usize,
    pub refresh_burst: usize,
    pub degradation_ratio: f64,
    pub degradation_patience: usize,
    pub refresh_every_active: usize,
    pub curvature_lr: f64,
    pub curvature_stride: usize,
    pub rank_ratio_threshold: f64,
    pub rank_strength_threshold: f64,
    pub rank_patience: usize,
    pub split_min_parent_obs: usize,
    pub split_selection_obs: usize,
    pub split_validation_obs: usize,
    pub split_max_validation: usize,
    pub split_min_branch: usize,
    pub split_mean_gain: f64,
    pub split_complexity_penalty: f64,
    pub split_accept_margin: f64,
    pub shadow_bias_lr: f64,
    pub shadow_bias_cap: f64,
    pub split_cooldown: usize,
    pub recursive_split_penalty: f64,
    pub max_refinement_depth: usize,
    pub merge_interval: usize,
    pub merge_min_age: usize,
    pub merge_pred_tolerance: f64,
    pub merge_dist_tolerance: f64,
    pub merge_min_count: f64,
    pub merge_recent_lr: f64,
    pub merge_recent_min_obs: usize,
    pub merge_patience: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dim: 12,
            budget: 24,
            tau: 0.20,
            alpha_cut: 0.10,
            stay_tolerance: 0.28,
            recurrence_tolerance: 0.38,
            update_tolerance: 0.22,
            redundancy_tolerance: 0.14,
            centroid_lr_cap: 0.08,
            decay: 0.9995,
            prior_strength: 2.0,
            recompress_interval: 32,
            evidence_decay: 0.85,
            noise_lr: 0.02,
            hazard_lr: 0.35,
            recurrence_threshold: 0.75,
            novelty_threshold: 3.5,
            hazard_discount_rec: 0.55,
            hazard_discount_nov: 0.35,
            min_sigma: 0.08,
            tensor_rates: vec![0.22, 0.06, 0.015],
            tensor_min_obs: 20,
            tensor_coherence: 0.48,
            tensor_strength: 0.006,
            tensor_lr: 0.055,
            tensor_l2: 0.002,
            theta_cap: 3.0,
            tensor_warmup: 12,
            loss_lr: 0.05,
            initial_burst: 24,
            post_ready_burst: 8,
            refresh_interval: 144,
            refresh_burst: 4,
            degradation_ratio: 1.0,
            degradation_patience: 6,
            refresh_every_active: 12,
            curvature_lr: 0.055,
            curvature_stride: 2,
            rank_ratio_threshold: 0.25,
            rank_strength_threshold: 0.0012,
            rank_patience: 2,
            split_min_parent_obs: 48,
            split_selection_obs: 72,
            split_validation_obs: 128,
            split_max_validation: 224,
            split_min_branch: 24,
            split_mean_gain: 0.0100,
            split_complexity_penalty: 0.080,
            split_accept_margin: 0.25,
            shadow_bias_lr: 0.045,
            shadow_bias_cap: 2.5,
            split_cooldown: 288,
            recursive_split_penalty: 1.6,
            max_refinement_depth: 1,
            merge_interval: 64,
            merge_min_age: 192,
            merge_pred_tolerance: 0.090,
            merge_dist_tolerance: 0.16,
            merge_min_count: 24.0,
            merge_recent_lr: 0.035,
            merge_recent_min_obs: 28,
            merge_patience: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prototype {
    pub uid: u64,
    pub centroid: Vec<f64>,
    pub count: f64,
    pub successes: f64,
    pub failures: f64,
    pub utility: f64,
    pub last_used: usize,
    pub created_t: usize,
    pub lineage_id: Option<u64>,
    pub split_created_t: Option<usize>,
    pub recent_mean: f64,
    pub recent_obs: usize,
    pub refinement_depth: usize,
}

impl Prototype {
    pub fn mean(&self) -> f64 {
        self.successes / (self.successes + self.failures).max(1.0e-12)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorCell {
    /// feature x timescale x 2, row-major as ((feature * scales + scale) * 2 + channel)
    pub tensor: Vec<f64>,
    pub u: Vec<f64>,
    pub theta: f64,
    pub proj2: f64,
    pub obs: usize,
    pub coherence: f64,
    pub strength: f64,
    pub ready: bool,
    pub base_loss: f64,
    pub corr_loss: f64,
    pub burst_left: usize,
    pub since_burst: usize,
    pub tensor_steps: usize,
    pub refreshes: usize,
    pub wakes: usize,
    pub sleeps: usize,
    pub curvature: Vec<f64>,
    pub curvature_obs: usize,
    pub rank_ratio2: f64,
    pub rank_strength1: f64,
    pub rank_strength2: f64,
    pub rank_persist: usize,
    pub last_rank_refresh_seen: isize,
    pub rank_dirs: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionShadow {
    pub direction: Vec<f64>,
    pub bias_neg: f64,
    pub bias_pos: f64,
    pub parent_loss: f64,
    pub child_loss: f64,
    pub n: usize,
}

impl DirectionShadow {
    pub fn mean_gain(&self) -> f64 {
        (self.parent_loss - self.child_loss) / self.n.max(1) as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CandidatePhase { Select, Validate }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitCandidate {
    pub parent_uid: u64,
    pub started_t: usize,
    pub directions: Vec<DirectionShadow>,
    pub selection_target: usize,
    pub validation_target: usize,
    pub max_validation: usize,
    pub phase: CandidatePhase,
    pub selected: Option<usize>,
    pub validation_n: usize,
    pub validation_parent_loss: f64,
    pub validation_child_loss: f64,
    pub bias_neg: f64,
    pub bias_pos: f64,
    pub neg_n: usize,
    pub pos_n: usize,
    pub neg_success: f64,
    pub pos_success: f64,
    pub neg_sum: Vec<f64>,
    pub pos_sum: Vec<f64>,
}

impl SplitCandidate {
    pub fn validation_gain(&self) -> f64 {
        self.validation_parent_loss - self.validation_child_loss
    }
    pub fn validation_mean_gain(&self) -> f64 {
        self.validation_gain() / self.validation_n.max(1) as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Read {
    pub prediction: f64,
    pub memberships: Vec<f64>,
    pub active_ids: Vec<usize>,
    pub current_id: Option<usize>,
    pub nearest_id: Option<usize>,
    pub nearest_dist: f64,
    pub revision: bool,
    pub reactivated: bool,
    pub discovered: bool,
    pub unresolved: bool,
    pub budget_pressure: bool,
    pub stored: usize,
    pub comparisons: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub t: usize,
    pub current_id: Option<usize>,
    pub pending_kind: Option<String>,
    pub pending_id: Option<usize>,
    pub pending_count: usize,
    pub pending_score: f64,
    pub pending_sum: Vec<f64>,
    pub sigma2: f64,
    pub hazard: f64,
    pub prototypes: Vec<Prototype>,
    pub tensor_cells: Vec<TensorCell>,
    pub revision_count: usize,
    pub reactivation_count: usize,
    pub discovery_count: usize,
    pub unresolved_count: usize,
    pub budget_pressure_count: usize,
    pub recompression_count: usize,
    pub split_proposals: usize,
    pub split_rejections: usize,
    pub split_promotions: usize,
    pub merge_promotions: usize,
    pub split_budget_blocks: usize,
    pub rank_checks: usize,
    pub curvature_updates: usize,
    pub tensor_updates: usize,
    pub tensor_awake_steps: usize,
    pub tensor_sleep_steps: usize,
    pub tensor_refreshes: usize,
    pub tensor_wakes: usize,
    pub tensor_sleeps: usize,
    pub split_candidates: HashMap<u64, SplitCandidate>,
}
