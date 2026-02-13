use routee_compass_core::{
    algorithm::search::SearchTree,
    model::{
        network::{Edge, EdgeId, Vertex},
        state::{InputFeature, StateModel, StateVariable, StateVariableConfig},
        traversal::{TraversalModel, TraversalModelError, TraversalModelService},
        unit::{EnergyRateUnit, EnergyUnit},
    },
};
use routee_compass_powertrain::model::{fieldname, prediction::PredictionModelRecord};
use std::{collections::HashSet, sync::Arc};
use uom::{si::f64::Energy, ConstZero};

use super::gtfs_stops::StopEdgeMapping;

/// A transit ICE energy traversal model that extends the standard ICE energy model
/// with GTFS stop penalties. When a bus must stop at a transit stop, extra energy
/// is needed to decelerate to zero and re-accelerate to the link speed.
///
/// The stop penalty is estimated as kinetic energy: KE = 0.5 * mass * v²
/// where mass comes from the prediction model record and v is the average link speed.
/// The kinetic energy (in Joules) is converted to the appropriate liquid fuel unit.
#[derive(Clone)]
pub struct TransitIceEnergyModel {
    prediction_model_record: Arc<PredictionModelRecord>,
    include_trip_energy: bool,
    /// The set of edge_ids that have stops for this specific trip
    stop_edges: HashSet<EdgeId>,
    /// Whether to include the stop energy penalty during traversal
    include_stop_penalty: bool,
}

impl TransitIceEnergyModel {
    pub fn new(
        prediction_model_record: Arc<PredictionModelRecord>,
        include_trip_energy: bool,
        stop_edges: HashSet<EdgeId>,
        include_stop_penalty: bool,
    ) -> Self {
        Self {
            prediction_model_record,
            include_trip_energy,
            stop_edges,
            include_stop_penalty,
        }
    }

    /// Compute the stop energy penalty for a given edge.
    ///
    /// Uses kinetic energy formula: KE = 0.5 * mass * v²
    /// This represents the energy needed to accelerate from 0 to the average
    /// link speed after stopping at a transit stop. The result is converted
    /// from Joules to the liquid fuel energy unit.
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

        // KE = 0.5 * m * v^2 (in Joules, since mass is in kg and speed in m/s)
        let ke = 0.5 * mass * speed * speed;

        Ok(ke)
    }
}

impl TraversalModel for TransitIceEnergyModel {
    fn name(&self) -> String {
        format!(
            "Transit ICE Energy Model: {}",
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
        let mut features = vec![(
            String::from(fieldname::EDGE_ENERGY_LIQUID),
            StateVariableConfig::Energy {
                initial: Energy::ZERO,
                accumulator: false,
                output_unit: Some(
                    self.prediction_model_record
                        .energy_rate_unit
                        .associated_energy_unit(),
                ),
            },
        )];
        if self.include_trip_energy {
            features.push((
                String::from(fieldname::TRIP_ENERGY_LIQUID),
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

        // Generate energy for link traversal using the prediction model
        let base_energy = self.prediction_model_record.predict(state, state_model)?;

        // Compute stop penalty if applicable
        let stop_penalty = self.compute_stop_penalty(edge, state, state_model)?;

        // Total energy = base traversal energy + stop penalty
        let total_energy = base_energy + stop_penalty;

        if self.include_trip_energy {
            state_model.add_energy(state, fieldname::TRIP_ENERGY_LIQUID, &total_energy)?;
        }
        state_model.set_energy(state, fieldname::EDGE_ENERGY_LIQUID, &total_energy)?;

        Ok(())
    }

    fn estimate_traversal(
        &self,
        _od: (&Vertex, &Vertex),
        state: &mut Vec<StateVariable>,
        _tree: &SearchTree,
        state_model: &StateModel,
    ) -> Result<(), TraversalModelError> {
        // Use a flat energy rate heuristic (no stop penalty to keep admissible for A*)
        let distance = state_model.get_distance(state, fieldname::EDGE_DISTANCE)?;

        let energy = match self.prediction_model_record.energy_rate_unit {
            EnergyRateUnit::GGPM => {
                let distance_miles = distance.get::<uom::si::length::mile>();
                let energy_gallons_gas =
                    self.prediction_model_record.a_star_heuristic_energy_rate * distance_miles;
                EnergyUnit::GallonsGasolineEquivalent.to_uom(energy_gallons_gas)
            }
            EnergyRateUnit::GDPM => {
                let distance_miles = distance.get::<uom::si::length::mile>();
                let energy_gallons_diesel =
                    self.prediction_model_record.a_star_heuristic_energy_rate * distance_miles;
                EnergyUnit::GallonsDieselEquivalent.to_uom(energy_gallons_diesel)
            }
            _ => {
                return Err(TraversalModelError::BuildError(format!(
                    "unsupported energy rate unit for ICE model: {}",
                    self.prediction_model_record.energy_rate_unit
                )));
            }
        };

        if self.include_trip_energy {
            state_model.add_energy(state, fieldname::TRIP_ENERGY_LIQUID, &energy)?;
        }
        state_model.set_energy(state, fieldname::EDGE_ENERGY_LIQUID, &energy)?;
        Ok(())
    }
}

/// Service that holds the shared ICE model state and builds per-query traversal models.
pub struct TransitIceEnergyModelService {
    prediction_model_record: Arc<PredictionModelRecord>,
    include_trip_energy: bool,
    stop_edge_mapping: StopEdgeMapping,
}

impl TransitIceEnergyModelService {
    pub fn new(
        prediction_model_record: Arc<PredictionModelRecord>,
        include_trip_energy: bool,
        stop_edge_mapping: StopEdgeMapping,
    ) -> Self {
        Self {
            prediction_model_record,
            include_trip_energy,
            stop_edge_mapping,
        }
    }
}

impl TraversalModelService for TransitIceEnergyModelService {
    fn build(
        &self,
        query: &serde_json::Value,
    ) -> Result<Arc<dyn TraversalModel>, TraversalModelError> {
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
                "transit ICE query for trip_id='{}': {} stop edges, include_stop_penalty={}",
                tid,
                stop_edges.len(),
                include_stop_penalty
            );
        }

        let model = TransitIceEnergyModel::new(
            self.prediction_model_record.clone(),
            self.include_trip_energy,
            stop_edges,
            include_stop_penalty,
        );

        Ok(Arc::new(model))
    }
}
