use ndarray::Array2;
use ort::session::Session;
use routee_compass_core::{
    algorithm::search::SearchTree,
    model::{
        network::Vertex,
        state::{InputFeature, StateModel, StateVariable, StateVariableConfig},
        traversal::{
            EdgeFrontierContext, TraversalModel, TraversalModelBuilder, TraversalModelError,
            TraversalModelService,
        },
        unit::SpeedUnit,
    },
};
use serde::Deserialize;
use std::sync::{Arc, Mutex};
use uom::si::{f64::Velocity, velocity::mile_per_hour};

/// Name of the state feature this model writes: a per-edge, per-query
/// ML-predicted transit bus speed (mph), for consumption by an energy model
/// configured with `speed_feature_name = "transit_speed"` instead of the
/// default OSM-derived `edge_speed`.
pub const TRANSIT_SPEED: &str = "transit_speed";

/// Below this, division-by-speed and kinetic-energy computations downstream
/// become unstable; matches the SPEED_FLOOR_MPH used to clean the training data.
const MIN_SPEED_MPH: f64 = 1.0;

/// The subset of `input_order` that is resolved per-query (from
/// `start_time`/`start_weekday`) rather than read from per-edge state.
const TEMPORAL_FEATURE_NAMES: [&str; 3] = ["hour", "is_weekday", "is_peak"];

/// Minimal view of the manifest JSON written by
/// `scripts/gtfs_realtime/fit_speed_models.py` — only the field ordering is
/// needed here; the rest (feature engineering, medians, etc.) is Python-side
/// bookkeeping for reproducing the same preprocessing when writing the
/// per-edge feature files consumed by `custom` traversal models upstream.
#[derive(Deserialize)]
struct SpeedModelManifest {
    input_order: Vec<String>,
}

/// Shared, query-independent state: the loaded ONNX session (behind a mutex,
/// since `Session::run` requires `&mut self`) and the exact ordering of
/// features the model expects as input.
struct TransitSpeedModelCore {
    session: Mutex<Session>,
    input_order: Vec<String>,
}

/// Per-query transit bus speed model. Static per-edge features (road
/// attributes, functional_class one-hot, missing-value indicators) are read
/// from state, having been populated by upstream `custom` traversal models
/// (one per feature, loaded from per-edge dense files). Temporal features
/// (`hour`/`is_weekday`/`is_peak`) are resolved once per query from
/// `start_time`/`start_weekday` and closed over here, since they don't vary
/// by edge.
#[derive(Clone)]
pub struct TransitSpeedModel {
    core: Arc<TransitSpeedModelCore>,
    hour: f64,
    is_weekday: f64,
    is_peak: f64,
}

impl TransitSpeedModel {
    fn predict_mph(
        &self,
        state: &[StateVariable],
        state_model: &StateModel,
    ) -> Result<f64, TraversalModelError> {
        let mut features: Vec<f32> = Vec::with_capacity(self.core.input_order.len());
        for name in self.core.input_order.iter() {
            let value = match name.as_str() {
                "hour" => self.hour,
                "is_weekday" => self.is_weekday,
                "is_peak" => self.is_peak,
                other => state_model.get_custom_f64(state, other).map_err(|e| {
                    TraversalModelError::TraversalModelFailure(format!(
                        "transit speed model missing feature '{other}': {e}"
                    ))
                })?,
            };
            features.push(value as f32);
        }

        let n_features = features.len();
        let array = Array2::from_shape_vec((1, n_features), features).map_err(|e| {
            TraversalModelError::TraversalModelFailure(format!(
                "failed to build ONNX input array: {e}"
            ))
        })?;
        let input_tensor = ort::value::Value::from_array(array).map_err(|e| {
            TraversalModelError::TraversalModelFailure(format!(
                "failed to build ONNX input tensor: {e}"
            ))
        })?;

        let mut session = self.core.session.lock().map_err(|e| {
            TraversalModelError::TraversalModelFailure(format!("failed to lock ONNX session: {e}"))
        })?;
        let outputs = session
            .run(ort::inputs!["input" => input_tensor])
            .map_err(|e| {
                TraversalModelError::TraversalModelFailure(format!(
                    "transit speed ONNX inference failed: {e}"
                ))
            })?;
        let (_name, output_tensor) = outputs.iter().next().ok_or_else(|| {
            TraversalModelError::TraversalModelFailure(
                "transit speed ONNX model returned no outputs".to_string(),
            )
        })?;
        let tensor_data = output_tensor.try_extract_tensor::<f32>().map_err(|e| {
            TraversalModelError::TraversalModelFailure(format!(
                "failed to extract transit speed ONNX output: {e}"
            ))
        })?;
        let predicted_mph = *tensor_data.1.first().ok_or_else(|| {
            TraversalModelError::TraversalModelFailure(
                "transit speed ONNX output tensor is empty".to_string(),
            )
        })? as f64;

        Ok(predicted_mph.max(MIN_SPEED_MPH))
    }
}

impl TraversalModel for TransitSpeedModel {
    fn name(&self) -> String {
        "Transit Speed Model (ML, ONNX)".to_string()
    }

    fn input_features(&self) -> Vec<InputFeature> {
        self.core
            .input_order
            .iter()
            .filter(|name| !TEMPORAL_FEATURE_NAMES.contains(&name.as_str()))
            .map(|name| InputFeature::Custom {
                name: name.clone(),
                unit: "raw".to_string(),
            })
            .collect()
    }

    fn output_features(&self) -> Vec<(String, StateVariableConfig)> {
        vec![(
            String::from(TRANSIT_SPEED),
            StateVariableConfig::Speed {
                initial: Velocity::new::<mile_per_hour>(0.0),
                accumulator: false,
                output_unit: Some(SpeedUnit::MPH),
            },
        )]
    }

    fn traverse_edge(
        &self,
        _ctx: &EdgeFrontierContext,
        state: &mut Vec<StateVariable>,
        state_model: &StateModel,
    ) -> Result<(), TraversalModelError> {
        let predicted_mph = self.predict_mph(state, state_model)?;
        let speed = Velocity::new::<mile_per_hour>(predicted_mph);
        state_model.set_speed(state, TRANSIT_SPEED, &speed)?;
        Ok(())
    }

    fn estimate_traversal(
        &self,
        _od: (&Vertex, &Vertex),
        _state: &mut Vec<StateVariable>,
        _tree: &SearchTree,
        _state_model: &StateModel,
    ) -> Result<(), TraversalModelError> {
        // No per-edge context is available for an OD-pair heuristic estimate
        // (no edge_id to look up static features for), so transit_speed keeps
        // its initial value here; this only affects A* pruning, not reported
        // trip results, which are computed via traverse_edge.
        Ok(())
    }
}

fn parse_start_hour(start_time: Option<&str>) -> f64 {
    start_time
        .and_then(|s| s.split(':').next())
        .and_then(|h| h.parse::<u32>().ok())
        .map(|h| (h % 24) as f64)
        .unwrap_or(12.0)
}

fn is_weekday_name(day: &str) -> bool {
    matches!(
        day.to_lowercase().as_str(),
        "monday" | "tuesday" | "wednesday" | "thursday" | "friday"
    )
}

fn is_peak_hour(hour: f64) -> bool {
    (7.0..=9.0).contains(&hour) || (16.0..=18.0).contains(&hour)
}

/// Service that holds the shared ONNX session and builds per-query models,
/// resolving `hour`/`is_weekday`/`is_peak` from `start_time`/`start_weekday`
/// query parameters (already sent on every routing query — see
/// `deadhead_router.gtfs_time_to_query_time` / `predictor._shape_start_times`).
pub struct TransitSpeedModelService {
    core: Arc<TransitSpeedModelCore>,
}

impl TraversalModelService for TransitSpeedModelService {
    fn build(
        &self,
        query: &serde_json::Value,
    ) -> Result<Arc<dyn TraversalModel>, TraversalModelError> {
        let start_time = query.get("start_time").and_then(|v| v.as_str());
        let start_weekday = query
            .get("start_weekday")
            .and_then(|v| v.as_str())
            .unwrap_or("monday");

        let hour = parse_start_hour(start_time);
        let is_weekday = if is_weekday_name(start_weekday) {
            1.0
        } else {
            0.0
        };
        let is_peak = if is_peak_hour(hour) { 1.0 } else { 0.0 };

        let model = TransitSpeedModel {
            core: self.core.clone(),
            hour,
            is_weekday,
            is_peak,
        };
        Ok(Arc::new(model))
    }
}

pub struct TransitSpeedModelBuilder {}

impl TraversalModelBuilder for TransitSpeedModelBuilder {
    fn build(
        &self,
        parameters: &serde_json::Value,
    ) -> Result<Arc<dyn TraversalModelService>, TraversalModelError> {
        let onnx_path = parameters
            .get("onnx_model_input_file")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                TraversalModelError::BuildError(String::from("missing key 'onnx_model_input_file'"))
            })?;
        let manifest_path = parameters
            .get("manifest_input_file")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                TraversalModelError::BuildError(String::from("missing key 'manifest_input_file'"))
            })?;

        let manifest_str = std::fs::read_to_string(manifest_path).map_err(|e| {
            TraversalModelError::BuildError(format!(
                "failed to read transit speed manifest file '{manifest_path}': {e}"
            ))
        })?;
        let manifest: SpeedModelManifest = serde_json::from_str(&manifest_str).map_err(|e| {
            TraversalModelError::BuildError(format!(
                "failed to parse transit speed manifest file '{manifest_path}': {e}"
            ))
        })?;

        let session = Session::builder()
            .map_err(|e| {
                TraversalModelError::BuildError(format!(
                    "failed to build ONNX Runtime session: {e}"
                ))
            })?
            .commit_from_file(onnx_path)
            .map_err(|e| {
                TraversalModelError::BuildError(format!(
                    "failed to load transit speed ONNX model '{onnx_path}': {e}"
                ))
            })?;

        let core = Arc::new(TransitSpeedModelCore {
            session: Mutex::new(session),
            input_order: manifest.input_order,
        });

        Ok(Arc::new(TransitSpeedModelService { core }))
    }
}
