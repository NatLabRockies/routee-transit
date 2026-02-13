use routee_compass_core::algorithm::search::SearchTree;
use routee_compass_core::model::network::{Edge, Vertex};
use routee_compass_core::model::state::{
    InputFeature, StateModel, StateVariable, StateVariableConfig,
};
use routee_compass_core::model::traversal::{
    TraversalModel, TraversalModelBuilder, TraversalModelError, TraversalModelService,
};
use std::sync::Arc;

pub struct TransitTraversalModel {}

impl TraversalModel for TransitTraversalModel {
    fn name(&self) -> String {
        "TransitTraversalModel".to_string()
    }

    fn input_features(&self) -> Vec<InputFeature> {
        vec![]
    }

    fn output_features(&self) -> Vec<(String, StateVariableConfig)> {
        vec![]
    }

    fn traverse_edge(
        &self,
        _trajectory: (&Vertex, &Edge, &Vertex),
        _state: &mut Vec<StateVariable>,
        _tree: &SearchTree,
        _state_model: &StateModel,
    ) -> Result<(), TraversalModelError> {
        Ok(())
    }

    fn estimate_traversal(
        &self,
        _od: (&Vertex, &Vertex),
        _state: &mut Vec<StateVariable>,
        _tree: &SearchTree,
        _state_model: &StateModel,
    ) -> Result<(), TraversalModelError> {
        Ok(())
    }
}

pub struct TransitTraversalService {
    pub model: Arc<TransitTraversalModel>,
}

impl TraversalModelService for TransitTraversalService {
    fn build(
        &self,
        _parameters: &serde_json::Value,
    ) -> Result<Arc<dyn TraversalModel>, TraversalModelError> {
        Ok(self.model.clone())
    }
}

pub struct TransitTraversalBuilder {}

impl TraversalModelBuilder for TransitTraversalBuilder {
    fn build(
        &self,
        _parameters: &serde_json::Value,
    ) -> Result<Arc<dyn TraversalModelService>, TraversalModelError> {
        let model = Arc::new(TransitTraversalModel {});
        let service = Arc::new(TransitTraversalService { model });
        Ok(service)
    }
}
