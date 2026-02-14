use routee_compass_core::{
    algorithm::search::SearchTree,
    model::{
        network::{Edge, EdgeId, Vertex},
        state::{InputFeature, StateModel, StateVariable, StateVariableConfig},
        traversal::{TraversalModel, TraversalModelError, TraversalModelService},
        unit::{EnergyRateUnit, RatioUnit},
    },
};
use routee_compass_powertrain::model::{
    energy_model_ops, fieldname, prediction::PredictionModelRecord,
};
use std::{collections::HashSet, sync::Arc};
use uom::{
    si::f64::{Energy, Ratio},
    ConstZero,
};

use super::gtfs_stops::StopEdgeMapping;

/// A transit BEV energy traversal model that extends the standard BEV energy model
/// with GTFS stop penalties. When a bus must stop at a transit stop, extra energy
/// is needed to decelerate to zero and re-accelerate to the link speed.
///
/// The stop penalty is estimated as kinetic energy: KE = 0.5 * mass * v²
/// where mass comes from the prediction model record and v is the average link speed.
#[derive(Clone)]
pub struct TransitBevEnergyModel {
    prediction_model_record: Arc<PredictionModelRecord>,
    battery_capacity: Energy,
    starting_soc: Ratio,
    include_trip_energy: bool,
    /// The set of edge_ids that have stops for this specific trip
    stop_edges: HashSet<EdgeId>,
    /// Whether to include the stop energy penalty during traversal
    include_stop_penalty: bool,
}

impl TransitBevEnergyModel {
    pub fn new(
        prediction_model_record: Arc<PredictionModelRecord>,
        battery_capacity: Energy,
        starting_battery_energy: Energy,
        include_trip_energy: bool,
        stop_edges: HashSet<EdgeId>,
        include_stop_penalty: bool,
    ) -> Result<Self, TraversalModelError> {
        let starting_soc =
            energy_model_ops::soc_from_energy(starting_battery_energy, battery_capacity).map_err(
                |e| {
                    TraversalModelError::BuildError(format!(
                        "Error building Transit BEV Energy model due to {e}"
                    ))
                },
            )?;
        Ok(Self {
            prediction_model_record,
            battery_capacity,
            starting_soc,
            include_trip_energy,
            stop_edges,
            include_stop_penalty,
        })
    }

    /// Compute the stop energy penalty for a given edge.
    ///
    /// Uses kinetic energy formula: KE = 0.5 * mass * v²
    /// This represents the energy needed to accelerate from 0 to the average
    /// link speed after stopping at a transit stop.
    fn compute_stop_penalty(
        &self,
        edge: &Edge,
        state: &[StateVariable],
        state_model: &StateModel,
    ) -> Result<Energy, TraversalModelError> {
        if !self.include_stop_penalty || !self.stop_edges.contains(&edge.edge_id) {
            return Ok(Energy::ZERO);
        }

        // Get speed from state (the average link speed)
        let speed = state_model.get_speed(state, fieldname::EDGE_SPEED)?;

        // Get mass from the prediction model record
        let mass = self.prediction_model_record.mass_estimate;

        // KE = 0.5 * mass * v^2
        let ke = 0.5 * mass * speed * speed;

        Ok(ke)
    }
}

impl TraversalModel for TransitBevEnergyModel {
    fn name(&self) -> String {
        format!(
            "Transit BEV Energy Model: {}",
            self.prediction_model_record.name
        )
    }

    fn input_features(&self) -> Vec<InputFeature> {
        let mut input_features = vec![
            InputFeature::Distance {
                name: String::from(fieldname::EDGE_DISTANCE),
                unit: None,
            },
            InputFeature::Speed {
                name: String::from(fieldname::EDGE_SPEED),
                unit: None,
            },
        ];
        input_features.extend(self.prediction_model_record.input_features.clone());
        input_features
    }

    fn output_features(&self) -> Vec<(String, StateVariableConfig)> {
        let mut features = vec![
            (
                String::from(fieldname::EDGE_ENERGY_ELECTRIC),
                StateVariableConfig::Energy {
                    initial: Energy::ZERO,
                    accumulator: false,
                    output_unit: Some(
                        self.prediction_model_record
                            .energy_rate_unit
                            .associated_energy_unit(),
                    ),
                },
            ),
            (
                String::from(fieldname::TRIP_SOC),
                StateVariableConfig::Ratio {
                    initial: self.starting_soc,
                    accumulator: true,
                    output_unit: Some(RatioUnit::default()),
                },
            ),
            (
                String::from(fieldname::BATTERY_CAPACITY),
                StateVariableConfig::Energy {
                    initial: self.battery_capacity,
                    accumulator: false,
                    output_unit: Some(
                        self.prediction_model_record
                            .energy_rate_unit
                            .associated_energy_unit(),
                    ),
                },
            ),
        ];
        if self.include_trip_energy {
            features.push((
                String::from(fieldname::TRIP_ENERGY_ELECTRIC),
                StateVariableConfig::Energy {
                    initial: Energy::ZERO,
                    accumulator: true,
                    output_unit: Some(
                        self.prediction_model_record
                            .energy_rate_unit
                            .associated_energy_unit(),
                    ),
                },
            ));
        }
        features
    }

    fn traverse_edge(
        &self,
        trajectory: (&Vertex, &Edge, &Vertex),
        state: &mut Vec<StateVariable>,
        _tree: &SearchTree,
        state_model: &StateModel,
    ) -> Result<(), TraversalModelError> {
        let (_src, edge, _dst) = trajectory;

        // Gather state variables
        let start_soc = state_model.get_ratio(state, fieldname::TRIP_SOC)?;

        // Generate energy for link traversal using the prediction model
        let base_energy = self.prediction_model_record.predict(state, state_model)?;

        // Compute stop penalty if applicable
        let stop_penalty = self.compute_stop_penalty(edge, state, state_model)?;

        // Total energy = base traversal energy + stop penalty
        let total_energy = base_energy + stop_penalty;

        if self.include_trip_energy {
            state_model.add_energy(state, fieldname::TRIP_ENERGY_ELECTRIC, &total_energy)?;
        }
        state_model.set_energy(state, fieldname::EDGE_ENERGY_ELECTRIC, &total_energy)?;

        let end_soc =
            energy_model_ops::update_soc_percent(start_soc, total_energy, self.battery_capacity)?;
        state_model.set_ratio(state, fieldname::TRIP_SOC, &end_soc)?;

        Ok(())
    }

    fn estimate_traversal(
        &self,
        _od: (&Vertex, &Vertex),
        state: &mut Vec<StateVariable>,
        _tree: &SearchTree,
        state_model: &StateModel,
    ) -> Result<(), TraversalModelError> {
        // Use the same heuristic as the standard BEV model (no stop penalty
        // in the heuristic to keep it admissible for A*)
        let distance = state_model.get_distance(state, fieldname::EDGE_DISTANCE)?;
        let start_soc = state_model.get_ratio(state, fieldname::TRIP_SOC)?;

        let energy = match self.prediction_model_record.energy_rate_unit {
            EnergyRateUnit::KWHPM => {
                let distance_miles = distance.get::<uom::si::length::mile>();
                let energy_kwh =
                    self.prediction_model_record.a_star_heuristic_energy_rate * distance_miles;
                Energy::new::<uom::si::energy::kilowatt_hour>(energy_kwh)
            }
            EnergyRateUnit::KWHPKM => {
                let distance_km = distance.get::<uom::si::length::kilometer>();
                let energy_kwh =
                    self.prediction_model_record.a_star_heuristic_energy_rate * distance_km;
                Energy::new::<uom::si::energy::kilowatt_hour>(energy_kwh)
            }
            _ => {
                return Err(TraversalModelError::BuildError(format!(
                    "unsupported energy rate unit: {}",
                    self.prediction_model_record.energy_rate_unit
                )));
            }
        };

        let end_soc =
            energy_model_ops::update_soc_percent(start_soc, energy, self.battery_capacity)?;

        if self.include_trip_energy {
            state_model.add_energy(state, fieldname::TRIP_ENERGY_ELECTRIC, &energy)?;
        }
        state_model.set_energy(state, fieldname::EDGE_ENERGY_ELECTRIC, &energy)?;
        state_model.set_ratio(state, fieldname::TRIP_SOC, &end_soc)?;
        Ok(())
    }
}

/// Service that holds the shared BEV model state and builds per-query traversal models.
pub struct TransitBevEnergyModelService {
    prediction_model_record: Arc<PredictionModelRecord>,
    battery_capacity: Energy,
    include_trip_energy: bool,
    stop_edge_mapping: StopEdgeMapping,
}

impl TransitBevEnergyModelService {
    pub fn new(
        prediction_model_record: Arc<PredictionModelRecord>,
        battery_capacity: Energy,
        include_trip_energy: bool,
        stop_edge_mapping: StopEdgeMapping,
    ) -> Self {
        Self {
            prediction_model_record,
            battery_capacity,
            include_trip_energy,
            stop_edge_mapping,
        }
    }
}

impl TraversalModelService for TransitBevEnergyModelService {
    fn build(
        &self,
        query: &serde_json::Value,
    ) -> Result<Arc<dyn TraversalModel>, TraversalModelError> {
        // Check for starting SOC override in query
        let starting_energy =
            match energy_model_ops::get_query_start_energy(query, self.battery_capacity)? {
                Some(energy) => energy,
                None => self.battery_capacity, // default to full charge
            };

        // Read trip_id from query
        let trip_id = query
            .get("trip_id")
            .and_then(|v| v.as_str())
            .map(String::from);

        // Read include_stop_penalty from query (default: true)
        let include_stop_penalty = query
            .get("include_stop_penalty")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        // Look up the stop edges for this trip
        let stop_edges = match &trip_id {
            Some(tid) => self.stop_edge_mapping.get(tid).cloned().unwrap_or_default(),
            None => HashSet::new(),
        };

        if let Some(tid) = &trip_id {
            log::debug!(
                "transit BEV query for trip_id='{}': {} stop edges, include_stop_penalty={}",
                tid,
                stop_edges.len(),
                include_stop_penalty
            );
        }

        let model = TransitBevEnergyModel::new(
            self.prediction_model_record.clone(),
            self.battery_capacity,
            starting_energy,
            self.include_trip_energy,
            stop_edges,
            include_stop_penalty,
        )?;

        Ok(Arc::new(model))
    }
}
