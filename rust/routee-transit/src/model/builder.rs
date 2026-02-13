use crate::model::traversal::transit_traversal::TransitTraversalBuilder;
use routee_compass::app::compass::BuilderRegistration;
use routee_compass_core::model::traversal::TraversalModelBuilder;
use std::rc::Rc;

pub const BUILDER_REGISTRATION: BuilderRegistration = BuilderRegistration(|builders| {
    builders.add_traversal_model(
        "transit".to_string(),
        Rc::new(TransitTraversalBuilder {}) as Rc<dyn TraversalModelBuilder>,
    );
    Ok(())
});
