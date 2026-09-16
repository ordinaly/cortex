use cortex_runtime::{NativeCortexRuntime, RuntimeConfig};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyclass(name = "CortexRuntimeRead", module = "cortex._cortex_native", frozen)]
#[derive(Clone)]
pub struct PyCortexRuntimeRead {
    #[pyo3(get)]
    pub bindings: Vec<usize>,
    #[pyo3(get)]
    pub subgroup: Vec<usize>,
    #[pyo3(get)]
    pub subgroup_margin: f64,
    #[pyo3(get)]
    pub relation_changed: Vec<(u32, u32)>,
    #[pyo3(get)]
    pub causal_changed: Vec<(u32, u32)>,
    #[pyo3(get)]
    pub articulation_vector: Vec<f64>,
    #[pyo3(get)]
    pub memory_stored: usize,
    #[pyo3(get)]
    pub memory_unresolved: bool,
    #[pyo3(get)]
    pub prediction: f64,
    #[pyo3(get)]
    pub memberships: Vec<f64>,
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
}

#[pyclass(name = "CortexRuntime", module = "cortex._cortex_native")]
pub struct PyCortexRuntime {
    inner: NativeCortexRuntime,
}

#[pymethods]
impl PyCortexRuntime {
    #[new]
    #[pyo3(signature = (feature_dim=16, max_entities=96, budget=24, config_json=None))]
    fn new(
        feature_dim: usize,
        max_entities: usize,
        budget: usize,
        config_json: Option<String>,
    ) -> PyResult<Self> {
        let cfg = if let Some(s) = config_json {
            serde_json::from_str::<RuntimeConfig>(&s)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            let mut cfg = RuntimeConfig::default();
            cfg.articulation.feature_dim = feature_dim;
            cfg.articulation.max_entities = max_entities;
            cfg.memory.budget = budget;
            cfg.continual.budget = budget;
            cfg
        };
        Ok(Self {
            inner: NativeCortexRuntime::new(cfg).map_err(PyValueError::new_err)?,
        })
    }

    #[pyo3(signature = (
        detections,
        relation_obs=Vec::new(),
        intervention_src_det=None,
        outcomes=Vec::new(),
        outcome=None
    ))]
    fn step(
        &mut self,
        detections: Vec<Vec<f64>>,
        relation_obs: Vec<(usize, usize, u8)>,
        intervention_src_det: Option<usize>,
        outcomes: Vec<(usize, u8)>,
        outcome: Option<f64>,
    ) -> PyResult<PyCortexRuntimeRead> {
        let read = self
            .inner
            .step(
                &detections,
                &relation_obs,
                intervention_src_det,
                &outcomes,
                outcome,
            )
            .map_err(PyValueError::new_err)?;
        Ok(PyCortexRuntimeRead {
            bindings: read.articulation.bindings,
            subgroup: read.articulation.subgroup,
            subgroup_margin: read.articulation.subgroup_margin,
            relation_changed: read.graph.relation_changed,
            causal_changed: read.graph.causal_changed,
            articulation_vector: read.articulation_vector,
            memory_stored: read.memory.stored,
            memory_unresolved: read.memory.unresolved,
            prediction: read.continual.prediction,
            memberships: read.continual.memberships,
            current_id: read.continual.current_id,
            nearest_id: read.continual.nearest_id,
            nearest_dist: read.continual.nearest_dist,
            revision: read.continual.revision,
            reactivated: read.continual.reactivated,
            discovered: read.continual.discovered,
            unresolved: read.continual.unresolved,
            budget_pressure: read.continual.budget_pressure,
            stored: read.continual.stored,
        })
    }

    fn snapshot_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.snapshot())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn config_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn articulation_vector(&self) -> Vec<f64> {
        self.inner.articulation_vector()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCortexRuntime>()?;
    m.add_class::<PyCortexRuntimeRead>()?;
    Ok(())
}
