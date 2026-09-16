use cortex_core::{Config, CortexReasoner, VERSION};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyclass(name = "CortexRead", module = "cortex._cortex_native", frozen)]
#[derive(Clone)]
pub struct PyCortexRead {
    #[pyo3(get)] pub prediction: f64,
    #[pyo3(get)] pub memberships: Vec<f64>,
    #[pyo3(get)] pub active_ids: Vec<usize>,
    #[pyo3(get)] pub current_id: Option<usize>,
    #[pyo3(get)] pub nearest_id: Option<usize>,
    #[pyo3(get)] pub nearest_dist: f64,
    #[pyo3(get)] pub revision: bool,
    #[pyo3(get)] pub reactivated: bool,
    #[pyo3(get)] pub discovered: bool,
    #[pyo3(get)] pub unresolved: bool,
    #[pyo3(get)] pub budget_pressure: bool,
    #[pyo3(get)] pub stored: usize,
    #[pyo3(get)] pub comparisons: usize,
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
        serde_json::to_string(&self.inner.snapshot()).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn config_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[getter]
    fn stored(&self) -> usize { self.inner.prototypes.len() }

    #[getter]
    fn current_id(&self) -> Option<usize> { self.inner.current_id }
}

#[pyfunction]
fn native_version() -> &'static str { VERSION }

#[pymodule]
fn _cortex_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCortex>()?;
    m.add_class::<PyCortexRead>()?;
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    Ok(())
}
