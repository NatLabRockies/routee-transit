use routee_compass_core::model::{network::EdgeId, traversal::TraversalModelError};
use serde::Deserialize;
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

/// A mapping from trip_id to the set of edge_ids that have stops on them.
pub type StopEdgeMapping = HashMap<String, HashSet<EdgeId>>;

/// A single row in the stop-edge mapping CSV file.
#[derive(Debug, Deserialize)]
struct StopEdgeRecord {
    trip_id: String,
    edge_id: usize,
}

/// Loads a pre-computed stop-edge mapping from a CSV file.
///
/// The CSV file should have columns: `trip_id,edge_id`
/// where each row indicates that the given trip has a stop on the given edge.
///
/// # Arguments
///
/// * `path` - path to the CSV file
///
/// # Returns
///
/// A `StopEdgeMapping` (HashMap<String, HashSet<EdgeId>>) mapping each trip_id
/// to the set of edge_ids that contain stops for that trip.
pub fn load_stop_edge_mapping(path: &Path) -> Result<StopEdgeMapping, TraversalModelError> {
    let mut reader = csv::Reader::from_path(path).map_err(|e| {
        TraversalModelError::BuildError(format!(
            "failed to open stop-edge mapping file '{}': {e}",
            path.display()
        ))
    })?;

    let mut mapping: StopEdgeMapping = HashMap::new();

    for result in reader.deserialize() {
        let record: StopEdgeRecord = result.map_err(|e| {
            TraversalModelError::BuildError(format!(
                "failed to parse stop-edge mapping record: {e}"
            ))
        })?;
        mapping
            .entry(record.trip_id)
            .or_default()
            .insert(EdgeId(record.edge_id));
    }

    log::info!("loaded stop-edge mapping with {} trips", mapping.len());

    Ok(mapping)
}
