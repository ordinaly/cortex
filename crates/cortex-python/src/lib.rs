mod graph;
mod runtime;

use cortex_articulation::{
    ArticulationConfig, ArticulationStageTimings, ArticulationState, ArticulationWork,
};
use cortex_core::{Config, CortexReasoner, VERSION};
use cortex_memory::{FuzzyAccordionMemory, MemoryConfig};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyclass(name = "CortexRead", module = "cortex._cortex_native", frozen)]
#[derive(Clone)]
pub struct PyCortexRead {
    #[pyo3(get)]
    pub prediction: f64,
    #[pyo3(get)]
    pub memberships: Vec<f64>,
    #[pyo3(get)]
    pub active_ids: Vec<usize>,
    #[pyo3(get)]
    pub current_id: Option<usize>,
    #[pyo3(get)]
    pub nearest_id: Option<usize>,
    #[pyo3(get)]
    pub nearest_dist: f64,
    #[pyo3(get)]
    pub revision: bool,
    #[pyo3(get)]
    pub reactivated: bool,
    #[pyo3(get)]
    pub discovered: bool,
    #[pyo3(get)]
    pub unresolved: bool,
    #[pyo3(get)]
    pub budget_pressure: bool,
    #[pyo3(get)]
    pub stored: usize,
    #[pyo3(get)]
    pub comparisons: usize,
}

#[pyclass(name = "Cortex", module = "cortex._cortex_native")]
pub struct PyCortex {
    inner: CortexReasoner,
}

#[pymethods]
impl PyCortex {
    #[new]
    #[pyo3(signature = (dim=12, budget=24, config_json=None))]
    fn new(dim: usize, budget: usize, config_json: Option<String>) -> PyResult<Self> {
        let mut cfg = if let Some(s) = config_json {
            serde_json::from_str::<Config>(&s).map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            Config::default()
        };
        cfg.dim = dim;
        cfg.budget = budget;
        let inner = CortexReasoner::new(cfg).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    fn step(&mut self, x: Vec<f64>, y: Option<f64>) -> PyResult<PyCortexRead> {
        let r = self.inner.step(&x, y).map_err(PyValueError::new_err)?;
        Ok(PyCortexRead {
            prediction: r.prediction,
            memberships: r.memberships,
            active_ids: r.active_ids,
            current_id: r.current_id,
            nearest_id: r.nearest_id,
            nearest_dist: r.nearest_dist,
            revision: r.revision,
            reactivated: r.reactivated,
            discovered: r.discovered,
            unresolved: r.unresolved,
            budget_pressure: r.budget_pressure,
            stored: r.stored,
            comparisons: r.comparisons,
        })
    }

    fn snapshot_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.snapshot())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn config_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[getter]
    fn stored(&self) -> usize {
        self.inner.prototypes.len()
    }

    #[getter]
    fn current_id(&self) -> Option<usize> {
        self.inner.current_id
    }
}

#[pyclass(name = "ArticulationRead", module = "cortex._cortex_native", frozen)]
#[derive(Clone)]
pub struct PyArticulationRead {
    #[pyo3(get)]
    pub bindings: Vec<usize>,
    #[pyo3(get)]
    pub shifts: Vec<usize>,
    #[pyo3(get)]
    pub subgroup: Vec<usize>,
    #[pyo3(get)]
    pub subgroup_margin: f64,
    #[pyo3(get)]
    pub entity_changed: Vec<usize>,
    #[pyo3(get)]
    pub noise_changed: Vec<usize>,
    #[pyo3(get)]
    pub active_entities: usize,
}

#[pyclass(
    name = "ArticulationStageTimings",
    module = "cortex._cortex_native",
    frozen
)]
#[derive(Clone)]
pub struct PyArticulationStageTimings {
    #[pyo3(get)]
    pub validation_ns: u64,
    #[pyo3(get)]
    pub match_scoring_ns: u64,
    #[pyo3(get)]
    pub assignment_ns: u64,
    #[pyo3(get)]
    pub bind_finalize_ns: u64,
    #[pyo3(get)]
    pub entity_update_ns: u64,
    #[pyo3(get)]
    pub subgroup_evidence_ns: u64,
    #[pyo3(get)]
    pub dedup_ns: u64,
    #[pyo3(get)]
    pub prototype_reliability_ns: u64,
    #[pyo3(get)]
    pub noise_state_ns: u64,
    #[pyo3(get)]
    pub subgroup_select_ns: u64,
    #[pyo3(get)]
    pub total_ns: u64,
}

impl From<ArticulationStageTimings> for PyArticulationStageTimings {
    fn from(t: ArticulationStageTimings) -> Self {
        Self {
            validation_ns: t.validation_ns,
            match_scoring_ns: t.match_scoring_ns,
            assignment_ns: t.assignment_ns,
            bind_finalize_ns: t.bind_finalize_ns,
            entity_update_ns: t.entity_update_ns,
            subgroup_evidence_ns: t.subgroup_evidence_ns,
            dedup_ns: t.dedup_ns,
            prototype_reliability_ns: t.prototype_reliability_ns,
            noise_state_ns: t.noise_state_ns,
            subgroup_select_ns: t.subgroup_select_ns,
            total_ns: t.total_ns,
        }
    }
}

#[pyclass(name = "ArticulationWork", module = "cortex._cortex_native", frozen)]
#[derive(Clone)]
pub struct PyArticulationWork {
    #[pyo3(get)]
    pub detections: usize,
    #[pyo3(get)]
    pub entities_before: usize,
    #[pyo3(get)]
    pub match_group_size: usize,
    #[pyo3(get)]
    pub pair_scores: usize,
    #[pyo3(get)]
    pub transform_evaluations: usize,
    #[pyo3(get)]
    pub assignment_rows: usize,
    #[pyo3(get)]
    pub assignment_cols: usize,
    #[pyo3(get)]
    pub spawned: usize,
    #[pyo3(get)]
    pub subgroup_evidence_detections: usize,
    #[pyo3(get)]
    pub subgroup_transform_evaluations: usize,
}

impl From<ArticulationWork> for PyArticulationWork {
    fn from(w: ArticulationWork) -> Self {
        Self {
            detections: w.detections,
            entities_before: w.entities_before,
            match_group_size: w.match_group_size,
            pair_scores: w.pair_scores,
            transform_evaluations: w.transform_evaluations,
            assignment_rows: w.assignment_rows,
            assignment_cols: w.assignment_cols,
            spawned: w.spawned,
            subgroup_evidence_detections: w.subgroup_evidence_detections,
            subgroup_transform_evaluations: w.subgroup_transform_evaluations,
        }
    }
}

#[pyclass(name = "Articulation", module = "cortex._cortex_native")]
pub struct PyArticulation {
    inner: ArticulationState,
}

#[pymethods]
impl PyArticulation {
    #[new]
    #[pyo3(signature = (feature_dim=16, max_entities=96, config_json=None))]
    fn new(feature_dim: usize, max_entities: usize, config_json: Option<String>) -> PyResult<Self> {
        let mut cfg = if let Some(s) = config_json {
            serde_json::from_str::<ArticulationConfig>(&s)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            ArticulationConfig::default()
        };
        cfg.feature_dim = feature_dim;
        cfg.max_entities = max_entities;
        Ok(Self {
            inner: ArticulationState::new(cfg).map_err(PyValueError::new_err)?,
        })
    }

    fn step(&mut self, detections: Vec<Vec<f64>>) -> PyResult<PyArticulationRead> {
        let r = self
            .inner
            .observe(&detections)
            .map_err(PyValueError::new_err)?;
        Ok(PyArticulationRead {
            bindings: r.bindings,
            shifts: r.shifts,
            subgroup: r.subgroup,
            subgroup_margin: r.subgroup_margin,
            entity_changed: r.entity_changed,
            noise_changed: r.noise_changed,
            active_entities: r.active_entities,
        })
    }

    fn profile_step(
        &mut self,
        detections: Vec<Vec<f64>>,
    ) -> PyResult<(
        PyArticulationRead,
        PyArticulationStageTimings,
        PyArticulationWork,
    )> {
        let profiled = self
            .inner
            .observe_profiled(&detections)
            .map_err(PyValueError::new_err)?;
        let r = profiled.read;
        Ok((
            PyArticulationRead {
                bindings: r.bindings,
                shifts: r.shifts,
                subgroup: r.subgroup,
                subgroup_margin: r.subgroup_margin,
                entity_changed: r.entity_changed,
                noise_changed: r.noise_changed,
                active_entities: r.active_entities,
            },
            profiled.timings.into(),
            profiled.work.into(),
        ))
    }

    fn snapshot_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.snapshot())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[getter]
    fn active_entities(&self) -> usize {
        self.inner.entity_count()
    }
}

#[pyclass(name = "FuzzyMemoryRead", module = "cortex._cortex_native", frozen)]
#[derive(Clone)]
pub struct PyFuzzyMemoryRead {
    #[pyo3(get)]
    pub reconstruction: Vec<f64>,
    #[pyo3(get)]
    pub memberships: Vec<f64>,
    #[pyo3(get)]
    pub active_ids: Vec<usize>,
    #[pyo3(get)]
    pub nearest_id: Option<usize>,
    #[pyo3(get)]
    pub nearest_dist: f64,
    #[pyo3(get)]
    pub stored: usize,
    #[pyo3(get)]
    pub unresolved: bool,
    #[pyo3(get)]
    pub compressed: bool,
    #[pyo3(get)]
    pub structural_change: bool,
}

#[pyclass(name = "FuzzyMemory", module = "cortex._cortex_native")]
pub struct PyFuzzyMemory {
    inner: FuzzyAccordionMemory,
}

#[pymethods]
impl PyFuzzyMemory {
    #[new]
    #[pyo3(signature = (
        dim,
        budget=32,
        tau=0.08,
        alpha=0.12,
        fit_tolerance=0.035,
        distortion_budget=0.08,
        redundancy_tolerance=0.045,
        decay=0.995,
        recompress_interval=16
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        budget: usize,
        tau: f64,
        alpha: f64,
        fit_tolerance: f64,
        distortion_budget: f64,
        redundancy_tolerance: f64,
        decay: f64,
        recompress_interval: usize,
    ) -> PyResult<Self> {
        let cfg = MemoryConfig {
            dim,
            budget,
            tau,
            alpha,
            fit_tolerance,
            distortion_budget,
            redundancy_tolerance,
            decay,
            recompress_interval,
        };
        Ok(Self {
            inner: FuzzyAccordionMemory::new(cfg).map_err(PyValueError::new_err)?,
        })
    }

    fn step(&mut self, x: Vec<f64>) -> PyResult<PyFuzzyMemoryRead> {
        let r = self.inner.step(&x).map_err(PyValueError::new_err)?;
        Ok(PyFuzzyMemoryRead {
            reconstruction: r.reconstruction,
            memberships: r.memberships,
            active_ids: r.active_ids,
            nearest_id: r.nearest_id,
            nearest_dist: r.nearest_dist,
            stored: r.stored,
            unresolved: r.unresolved,
            compressed: r.compressed,
            structural_change: r.structural_change,
        })
    }

    fn snapshot_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.snapshot())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[getter]
    fn stored(&self) -> usize {
        self.inner.stored()
    }
}

#[pyfunction]
fn native_version() -> &'static str {
    VERSION
}

#[pymodule]
fn _cortex_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCortex>()?;
    m.add_class::<PyCortexRead>()?;
    m.add_class::<PyArticulation>()?;
    m.add_class::<PyArticulationRead>()?;
    m.add_class::<PyArticulationStageTimings>()?;
    m.add_class::<PyArticulationWork>()?;
    m.add_class::<PyFuzzyMemory>()?;
    m.add_class::<PyFuzzyMemoryRead>()?;
    graph::register(m)?;
    runtime::register(m)?;
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    Ok(())
}
