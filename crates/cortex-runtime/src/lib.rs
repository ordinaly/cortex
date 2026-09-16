//! Coarse-grained native Cortex runtime.
//!
//! This crate composes the independently differential-tested native layers in
//! the same order as the frozen Python stack:
//!
//! structured frame -> articulation -> sparse relation/causal evidence
//!                  -> 12-D articulation summary
//!                  -> fuzzy historical memory + continual plasticity.
//!
//! Memory reconstruction is deliberately *not* fed into the continual core;
//! frozen v0.9.x sends the same public articulation summary to both consumers.

use cortex_articulation::{
    ArticulationConfig, ArticulationRead, ArticulationSnapshot, ArticulationState,
};
use cortex_core::{
    Config as ContinualConfig, CortexReasoner, Read as ContinualRead, Snapshot as ContinualSnapshot,
};
use cortex_graph::{EvidenceConfig, EvidenceRead, EvidenceSnapshot, SparseEvidenceGraph};
use cortex_memory::{FuzzyAccordionMemory, MemoryConfig, MemoryRead, MemorySnapshot};
use serde::{Deserialize, Serialize};

pub const ARTICULATION_VECTOR_DIM: usize = 12;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeConfig {
    pub articulation: ArticulationConfig,
    pub graph: EvidenceConfig,
    pub memory: MemoryConfig,
    pub continual: ContinualConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        let mut memory = MemoryConfig::for_dim(ARTICULATION_VECTOR_DIM);
        // Frozen v0.9 wraps v0.8 with the continual budget for both memories.
        memory.budget = 24;
        Self {
            articulation: ArticulationConfig::default(),
            graph: EvidenceConfig::default(),
            memory,
            continual: ContinualConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeRead {
    pub articulation: ArticulationRead,
    pub graph: EvidenceRead,
    pub articulation_vector: Vec<f64>,
    pub memory: MemoryRead,
    pub continual: ContinualRead,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    pub articulation: ArticulationSnapshot,
    pub graph: EvidenceSnapshot,
    pub memory: MemorySnapshot,
    pub continual: ContinualSnapshot,
    pub articulation_vector: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct NativeCortexRuntime {
    pub cfg: RuntimeConfig,
    articulation: ArticulationState,
    graph: SparseEvidenceGraph,
    memory: FuzzyAccordionMemory,
    continual: CortexReasoner,
}

impl NativeCortexRuntime {
    pub fn new(cfg: RuntimeConfig) -> Result<Self, String> {
        if cfg.memory.dim != ARTICULATION_VECTOR_DIM {
            return Err(format!(
                "runtime memory dim must be {ARTICULATION_VECTOR_DIM}, got {}",
                cfg.memory.dim
            ));
        }
        if cfg.continual.dim != ARTICULATION_VECTOR_DIM {
            return Err(format!(
                "runtime continual dim must be {ARTICULATION_VECTOR_DIM}, got {}",
                cfg.continual.dim
            ));
        }
        Ok(Self {
            articulation: ArticulationState::new(cfg.articulation.clone())?,
            graph: SparseEvidenceGraph::new(cfg.graph)?,
            memory: FuzzyAccordionMemory::new(cfg.memory.clone())?,
            continual: CortexReasoner::new(cfg.continual.clone())?,
            cfg,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        detections: &[Vec<f64>],
        relation_obs: &[(usize, usize, u8)],
        intervention_src_det: Option<usize>,
        outcomes: &[(usize, u8)],
        outcome: Option<f64>,
    ) -> Result<RuntimeRead, String> {
        let articulation = self.articulation.observe(detections)?;
        let mut bindings = Vec::with_capacity(articulation.bindings.len());
        for &id in &articulation.bindings {
            bindings.push(
                u32::try_from(id)
                    .map_err(|_| format!("entity id {id} exceeds native graph NodeId capacity"))?,
            );
        }
        let graph = self
            .graph
            .observe(&bindings, relation_obs, intervention_src_det, outcomes)?;
        let articulation_vector = self.articulation_vector();
        let memory = self.memory.step(&articulation_vector)?;
        let continual = self.continual.step(&articulation_vector, outcome)?;
        Ok(RuntimeRead {
            articulation,
            graph,
            articulation_vector,
            memory,
            continual,
        })
    }

    pub fn articulation_vector(&self) -> Vec<f64> {
        let art = self.articulation.snapshot();
        let graph = self.graph.snapshot();
        build_articulation_vector(&self.cfg.articulation, &art, &graph)
    }

    pub fn snapshot(&self) -> RuntimeSnapshot {
        RuntimeSnapshot {
            articulation: self.articulation.snapshot(),
            graph: self.graph.snapshot(),
            memory: self.memory.snapshot(),
            continual: self.continual.snapshot(),
            articulation_vector: self.articulation_vector(),
        }
    }
}

pub fn build_articulation_vector(
    cfg: &ArticulationConfig,
    art: &ArticulationSnapshot,
    graph: &EvidenceSnapshot,
) -> Vec<f64> {
    let ne = art.active_entities;
    let mut gvec = [0.0; 4];
    let candidates = resolved_group_candidates(cfg);
    for (i, g) in candidates.iter().take(4).enumerate() {
        if art.subgroup == *g {
            gvec[i] = 1.0;
        }
    }

    let mut structure_supported = 0usize;
    let mut unresolved_noise = 0usize;
    let mut noise_supported = 0usize;
    let mut noise_total = 0usize;
    for entity in &art.noise_state {
        for &state in entity {
            noise_total += 1;
            match state {
                -1 => structure_supported += 1,
                2 => noise_supported += 1,
                _ => unresolved_noise += 1,
            }
        }
    }
    let noise = if noise_total == 0 {
        [0.0, 1.0, 0.0]
    } else {
        [
            structure_supported as f64 / noise_total as f64,
            unresolved_noise as f64 / noise_total as f64,
            noise_supported as f64 / noise_total as f64,
        ]
    };

    // Sparse storage changes allocation, not the semantics of unresolved pairs.
    // Unrepresented active-entity pairs have state 0, exactly like frozen v0.7.
    let relation_den = ne.saturating_mul(ne.saturating_sub(1)) / 2;
    let causal_den = ne.saturating_mul(ne.saturating_sub(1));
    let relation_active = graph
        .relations
        .iter()
        .filter(|e| e.evidence.state == 1)
        .count();
    let relation_resolved = graph
        .relations
        .iter()
        .filter(|e| e.evidence.state != 0)
        .count();
    let causal_active = graph
        .causal
        .iter()
        .filter(|e| e.evidence.state == 1)
        .count();

    let rel_active_fraction = if relation_den == 0 {
        0.0
    } else {
        relation_active as f64 / relation_den as f64
    };
    let rel_resolved_fraction = if relation_den == 0 {
        0.0
    } else {
        relation_resolved as f64 / relation_den as f64
    };
    let causal_active_fraction = if causal_den == 0 {
        0.0
    } else {
        causal_active as f64 / causal_den as f64
    };

    vec![
        (ne as f64 / cfg.max_entities.max(1) as f64).min(1.0),
        (art.subgroup_margin.max(0.0) / 0.25).min(1.0),
        gvec[0],
        gvec[1],
        gvec[2],
        gvec[3],
        noise[0],
        noise[1],
        noise[2],
        rel_active_fraction,
        rel_resolved_fraction,
        causal_active_fraction,
    ]
}

fn resolved_group_candidates(cfg: &ArticulationConfig) -> Vec<Vec<usize>> {
    if let Some(groups) = &cfg.group_candidates {
        return groups.clone();
    }
    let d = cfg.feature_dim;
    let all: Vec<usize> = (0..d).collect();
    if d % 4 != 0 {
        vec![vec![0], all]
    } else {
        vec![
            vec![0],
            vec![0, d / 2],
            vec![0, d / 4, d / 2, 3 * d / 4],
            all,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_default_summary_has_frozen_dimension() {
        let rt = NativeCortexRuntime::new(RuntimeConfig::default()).unwrap();
        let v = rt.articulation_vector();
        assert_eq!(v.len(), ARTICULATION_VECTOR_DIM);
        assert_eq!(v[7], 1.0); // no entities => unresolved residual-state fraction
    }

    #[test]
    fn one_full_frame_updates_all_native_layers() {
        let mut cfg = RuntimeConfig::default();
        cfg.articulation.feature_dim = 8;
        cfg.articulation.max_entities = 8;
        let mut rt = NativeCortexRuntime::new(cfg).unwrap();
        let read = rt
            .step(
                &[
                    vec![0.0, 1.0, 0.2, -0.4, 0.8, 0.1, -0.7, 0.3],
                    vec![1.0, -0.2, 0.5, 0.7, -0.3, 0.9, 0.1, -0.5],
                ],
                &[(0, 1, 1)],
                Some(0),
                &[(1, 1)],
                Some(1.0),
            )
            .unwrap();
        assert_eq!(read.articulation.bindings.len(), 2);
        assert_eq!(read.graph.represented_relations, 1);
        assert_eq!(read.graph.represented_causal, 1);
        assert_eq!(read.articulation_vector.len(), 12);
        assert_eq!(read.memory.stored, 1);
        assert_eq!(read.continual.stored, 1);
    }
}
