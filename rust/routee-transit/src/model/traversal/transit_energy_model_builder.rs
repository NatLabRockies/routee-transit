use super::gtfs_stops::load_stop_edge_mapping;
use super::transit_bev_energy_model::TransitBevEnergyModelService;
use super::transit_energy_model_service::TransitEnergyModelService;
use super::transit_ice_energy_model::TransitIceEnergyModelService;
use routee_compass_core::{
    config::ConfigJsonExtensions,
    model::traversal::{TraversalModelBuilder, TraversalModelError, TraversalModelService},
    model::unit::EnergyUnit,
};
use routee_compass_powertrain::model::prediction::{PredictionModelConfig, PredictionModelRecord};
use std::{collections::HashMap, path::Path, sync::Arc};

/// Builder that reads multiple vehicle config files and constructs a
/// `TransitEnergyModelService` dispatch layer. Each vehicle file must
/// contain a `name` and `type` field (`"bev"` or `"ice"`).
pub struct TransitEnergyModelBuilder {}

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

            let model_name = vehicle_json
                .get_config_string(&"name", &parent_key)
                .map_err(|e| {
                    TraversalModelError::BuildError(format!(
                        "vehicle model missing 'name' field in '{}': {}",
                        file_path, e
                    ))
                })?;
            let vehicle_type = vehicle_json
                .get_config_string(&"type", &parent_key)
                .map_err(|e| {
                    TraversalModelError::BuildError(format!(
                        "vehicle model missing 'type' field in '{}': {}",
                        file_path, e
                    ))
                })?;

            let include_trip_energy = match vehicle_json.get("include_trip_energy") {
                Some(v) => v.as_bool().ok_or_else(|| {
                    TraversalModelError::BuildError(
                        "Failed to parse 'include_trip_energy' as boolean".to_string(),
                    )
                })?,
                None => true,
            };

            let service: Arc<dyn TraversalModelService> = match vehicle_type.as_str() {
                "bev" => {
                    // Parse BEV-specific config
                    let config: PredictionModelConfig =
                        serde_json::from_value(vehicle_json.clone()).map_err(|e| {
                            TraversalModelError::BuildError(format!(
                                "failure reading prediction model configuration for '{}': {e}",
                                file_path
                            ))
                        })?;
                    let prediction_model = PredictionModelRecord::try_from(&config)?;

                    let battery_capacity_conf =
                        vehicle_json.get("battery_capacity").ok_or_else(|| {
                            TraversalModelError::BuildError(format!(
                                "missing key 'battery_capacity' in '{}'",
                                file_path
                            ))
                        })?;
                    let battery_energy_unit_conf =
                        vehicle_json.get("battery_capacity_unit").ok_or_else(|| {
                            TraversalModelError::BuildError(format!(
                                "missing key 'battery_capacity_unit' in '{}'",
                                file_path
                            ))
                        })?;
                    let battery_capacity = serde_json::from_value::<f64>(
                        battery_capacity_conf.clone(),
                    )
                    .map_err(|e| {
                        TraversalModelError::BuildError(format!(
                            "failed to parse battery capacity in '{}': {e}",
                            file_path
                        ))
                    })?;
                    let battery_energy_unit =
                        serde_json::from_value::<EnergyUnit>(battery_energy_unit_conf.clone())
                            .map_err(|e| {
                                TraversalModelError::BuildError(format!(
                                    "failed to parse battery capacity unit in '{}': {e}",
                                    file_path
                                ))
                            })?;
                    let battery_capacity = battery_energy_unit.to_uom(battery_capacity);

                    Arc::new(TransitBevEnergyModelService::new(
                        Arc::new(prediction_model),
                        battery_capacity,
                        include_trip_energy,
                        stop_edge_mapping.clone(),
                    ))
                }
                "ice" => {
                    // Parse ICE-specific config
                    let config: PredictionModelConfig =
                        serde_json::from_value(vehicle_json.clone()).map_err(|e| {
                            TraversalModelError::BuildError(format!(
                                "failure reading prediction model configuration for '{}': {e}",
                                file_path
                            ))
                        })?;
                    let prediction_model = PredictionModelRecord::try_from(&config)?;

                    Arc::new(TransitIceEnergyModelService::new(
                        Arc::new(prediction_model),
                        include_trip_energy,
                        stop_edge_mapping.clone(),
                    ))
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
