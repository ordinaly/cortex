use cortex_graph::{EvidenceConfig, SparseEvidenceGraph};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyclass(name = "GraphEvidenceRead", module = "cortex._cortex_native", frozen)]
#[derive(Clone)]
pub struct PyGraphEvidenceRead {
    #[pyo3(get)]
    pub relation_changed: Vec<(u32, u32)>,
    #[pyo3(get)]
    pub causal_changed: Vec<(u32, u32)>,
    #[pyo3(get)]
    pub represented_relations: usize,
    #[pyo3(get)]
    pub represented_causal: usize,
}

#[pyclass(name = "GraphEvidence", module = "cortex._cortex_native")]
pub struct PyGraphEvidence {
    inner: SparseEvidenceGraph,
}

#[pymethods]
impl PyGraphEvidence {
    #[new]
    #[pyo3(signature = (config_json=None))]
    fn new(config_json: Option<String>) -> PyResult<Self> {
        let cfg = if let Some(s) = config_json {
            serde_json::from_str::<EvidenceConfig>(&s)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            EvidenceConfig::default()
        };
        Ok(Self {
            inner: SparseEvidenceGraph::new(cfg).map_err(PyValueError::new_err)?,
        })
    }

    #[pyo3(signature = (bindings, relation_obs, intervention_src_det=None, outcomes=Vec::new()))]
    fn step(
        &mut self,
        bindings: Vec<u32>,
        relation_obs: Vec<(usize, usize, u8)>,
        intervention_src_det: Option<usize>,
        outcomes: Vec<(usize, u8)>,
    ) -> PyResult<PyGraphEvidenceRead> {
        let r = self
            .inner
            .observe(
                &bindings,
                &relation_obs,
                intervention_src_det,
                &outcomes,
            )
            .map_err(PyValueError::new_err)?;
        Ok(PyGraphEvidenceRead {
            relation_changed: r.relation_changed,
            causal_changed: r.causal_changed,
            represented_relations: r.represented_relations,
            represented_causal: r.represented_causal,
        })
    }

    fn snapshot_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.snapshot())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[getter]
    fn represented_relations(&self) -> usize {
        self.inner.represented_relations()
    }

    #[getter]
    fn represented_causal(&self) -> usize {
        self.inner.represented_causal()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraphEvidence>()?;
    m.add_class::<PyGraphEvidenceRead>()?;
    Ok(())
}
