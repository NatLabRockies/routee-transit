pub mod transit_app_py;

use pyo3::prelude::*;
use transit_app_py::TransitCompassAppPy;

// inject transit builders into the CompassBuilderInventory on library load
inventory::submit! { routee_transit::model::builder::BUILDER_REGISTRATION }

/// A Python module implemented in Rust.
#[pymodule]
fn routee_transit_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TransitCompassAppPy>()?;
    Ok(())
}
