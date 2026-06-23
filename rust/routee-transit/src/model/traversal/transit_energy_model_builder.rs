use super::gtfs_stops::load_stop_edge_mapping;
use super::transit_bev_energy_model::TransitBevEnergyModelService;
use super::transit_energy_model_service::TransitEnergyModelService;
use super::transit_ice_energy_model::TransitIceEnergyModelService;
use routee_compass_core::{
    config::ops::strip_type_from_config,
    config::ConfigJsonExtensions,
    model::traversal::{TraversalModelBuilder, TraversalModelError, TraversalModelService},
    model::unit::EnergyUnit,
};
use routee_compass_powertrain::model::prediction::{PredictionModelConfig, PredictionModelRecord};
use serde::Deserialize;
use std::{collections::HashMap, path::Path, sync::Arc};

/// Builder that reads multiple vehicle config files and constructs a
/// `TransitEnergyModelService` dispatch layer. Each vehicle file must
/// contain a `name` and `type` field (`"bev"` or `"ice"`).
pub struct TransitEnergyModelBuilder {}

/// BEV vehicle config. Mirrors `routee_compass_powertrain`'s
/// `BevEnergyModelConfig`: the prediction-model fields are flattened in and the
/// transit-specific wrapper fields are declared alongside them. The
/// `#[serde(flatten)]` field also relaxes `deny_unknown_fields`, so otherwise
/// unused keys carried by the shared config files (e.g. `distance_unit`) are
/// ignored rather than rejected.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TransitBevVehicleConfig {
    #[serde(flatten)]
    prediction_model: PredictionModelConfig,
    battery_capacity: f64,
    battery_capacity_unit: EnergyUnit,
    include_trip_energy: Option<bool>,
}

/// ICE vehicle config. Mirrors `routee_compass_powertrain`'s
/// `IceEnergyModelConfig` (see [`TransitBevVehicleConfig`] for the flatten
/// rationale).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TransitIceVehicleConfig {
    #[serde(flatten)]
    prediction_model: PredictionModelConfig,
    include_trip_energy: Option<bool>,
}

impl TraversalModelBuilder for TransitEnergyModelBuilder {
    fn build(
        &self,
        parameters: &serde_json::Value,
    ) -> Result<Arc<dyn TraversalModelService>, TraversalModelError> {
        let parent_key = String::from("transit energy traversal model");

        let vehicle_files = parameters
            .get_config_array(&"vehicle_input_files", &parent_key)
            .map_err(|e| TraversalModelError::BuildError(e.to_string()))?;

        // Load the stop-edge mapping (shared across all vehicles)
        let stop_edge_mapping_file = parameters
            .get("stop_edge_mapping_input_file")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                TraversalModelError::BuildError(String::from(
                    "missing key 'stop_edge_mapping_input_file'",
                ))
            })?;
        let stop_edge_mapping =
            load_stop_edge_mapping(&std::path::PathBuf::from(stop_edge_mapping_file))?;

        // Read optional include_trip_energy at the top level (can be overridden per vehicle)
        let top_level_include_trip_energy: Option<bool> = parameters
            .get("include_trip_energy")
            .and_then(|v| v.as_bool());

        // Read all vehicle configurations from files
        let mut vehicle_library: HashMap<String, Arc<dyn TraversalModelService>> = HashMap::new();
        for vehicle_file in vehicle_files {
            let file_path = vehicle_file.as_str().ok_or_else(|| {
                TraversalModelError::BuildError("vehicle file path must be a string".to_string())
            })?;

            let vehicle_config = config::Config::builder()
                .add_source(config::File::with_name(file_path))
                .build()
                .map_err(|e| {
                    TraversalModelError::BuildError(format!(
                        "failed to read vehicle config file '{}': {}",
                        file_path, e
                    ))
                })?;

            let mut vehicle_json = vehicle_config
                .try_deserialize::<serde_json::Value>()
                .map_err(|e| {
                    TraversalModelError::BuildError(format!(
                        "failed to parse vehicle config file '{}': {}",
                        file_path, e
                    ))
                })?
                .normalize_file_paths(Path::new(file_path), None)
                .map_err(|e| {
                    TraversalModelError::BuildError(format!(
                        "failed to normalize file paths in vehicle config file '{}': {}",
                        file_path, e
                    ))
                })?;

            // Inject include_trip_energy if specified at the top level
            if let Some(include_trip_energy) = top_level_include_trip_energy {
                vehicle_json["include_trip_energy"] = serde_json::Value::Bool(include_trip_energy);
            }

            // Strip and capture the `type` discriminator, leaving a JSON object
            // whose remaining keys deserialize into the per-type vehicle config.
            let (vehicle_json, vehicle_type) =
                strip_type_from_config(&vehicle_json).map_err(|e| {
                    TraversalModelError::BuildError(format!(
                        "vehicle model missing 'type' field in '{}': {}",
                        file_path, e
                    ))
                })?;

            let (model_name, service): (String, Arc<dyn TraversalModelService>) =
                match vehicle_type.as_str() {
                    "bev" => {
                        let config: TransitBevVehicleConfig =
                            serde_json::from_value(vehicle_json).map_err(|e| {
                                TraversalModelError::BuildError(format!(
                                    "failure reading vehicle configuration for '{}': {e}",
                                    file_path
                                ))
                            })?;
                        let prediction_model =
                            PredictionModelRecord::try_from(&config.prediction_model)?;
                        let battery_capacity =
                            config.battery_capacity_unit.to_uom(config.battery_capacity);

                        let service = TransitBevEnergyModelService::new(
                            Arc::new(prediction_model),
                            battery_capacity,
                            config.include_trip_energy.unwrap_or(true),
                            stop_edge_mapping.clone(),
                        );
                        (config.prediction_model.name, Arc::new(service))
                    }
                    "ice" => {
                        let config: TransitIceVehicleConfig =
                            serde_json::from_value(vehicle_json).map_err(|e| {
                                TraversalModelError::BuildError(format!(
                                    "failure reading vehicle configuration for '{}': {e}",
                                    file_path
                                ))
                            })?;
                        let prediction_model =
                            PredictionModelRecord::try_from(&config.prediction_model)?;

                        let service = TransitIceEnergyModelService::new(
                            Arc::new(prediction_model),
                            config.include_trip_energy.unwrap_or(true),
                            stop_edge_mapping.clone(),
                        );
                        (config.prediction_model.name, Arc::new(service))
                    }
                    _ => {
                        return Err(TraversalModelError::BuildError(format!(
                            "unknown vehicle model type in '{}': {}",
                            file_path, vehicle_type
                        )));
                    }
                };

            vehicle_library.insert(model_name, service);
        }

        let service = TransitEnergyModelService::new(vehicle_library)?;

        Ok(Arc::new(service))
    }
}
