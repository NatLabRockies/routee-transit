use pyo3::prelude::*;
use routee_transit;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    let res = routee_transit::sum(a, b);
    Ok(res.to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn routee_transit_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
