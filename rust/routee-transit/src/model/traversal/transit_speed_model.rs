use routee_compass_core::{
    algorithm::search::SearchTree,
    model::{
        network::Vertex,
        state::{InputFeature, StateModel, StateVariable, StateVariableConfig},
        traversal::{
            default::fieldname, EdgeFrontierContext, TraversalModel, TraversalModelBuilder,
            TraversalModelError, TraversalModelService,
        },
        unit::SpeedUnit,
    },
};
use serde::Deserialize;
use std::{str::FromStr, sync::Arc};
use uom::si::{f64::Velocity, time::second, velocity::mile_per_hour};

pub const TRANSIT_SPEED: &str = "transit_speed";
const BIN_START_HOURS: [usize; 6] = [0, 6, 9, 15, 19, 22];
const BINS: usize = 12;
const WEEKDAYS: [&str; 7] = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
];

struct SpeedTraces {
    edge_traces: Box<[u32]>,
    speeds: Box<[f32]>,
    max_speed: Velocity,
}

impl SpeedTraces {
    fn new(edge_traces: Vec<u32>, speeds: Vec<f32>) -> Result<Self, TraversalModelError> {
        if edge_traces.is_empty()
            || speeds.is_empty()
            || !speeds.len().is_multiple_of(BINS)
            || speeds.iter().any(|s| !s.is_finite() || *s < 1.0)
            || edge_traces
                .iter()
                .any(|id| *id as usize >= speeds.len() / BINS)
        {
            return Err(TraversalModelError::BuildError(
                "invalid transit speed traces or edge-to-trace IDs".into(),
            ));
        }
        let max_mph = speeds.iter().copied().fold(1.0_f32, f32::max);
        Ok(Self {
            edge_traces: edge_traces.into_boxed_slice(),
            speeds: speeds.into_boxed_slice(),
            max_speed: Velocity::new::<mile_per_hour>(max_mph as f64),
        })
    }

    fn speed(&self, edge_id: usize, bin: usize) -> Result<Velocity, TraversalModelError> {
        let trace_id = self.edge_traces.get(edge_id).ok_or_else(|| {
            TraversalModelError::TraversalModelFailure(format!(
                "no transit speed trace for edge {edge_id}"
            ))
        })?;
        Ok(Velocity::new::<mile_per_hour>(
            self.speeds[*trace_id as usize * BINS + bin] as f64,
        ))
    }
}

#[derive(Clone)]
pub struct TransitSpeedModel {
    traces: Arc<SpeedTraces>,
    start_seconds: f64,
    start_weekday: usize,
    speed_limit: Option<Velocity>,
}

impl TransitSpeedModel {
    fn bin(&self, elapsed_seconds: f64) -> Result<usize, TraversalModelError> {
        let seconds = self.start_seconds + elapsed_seconds;
        if !seconds.is_finite() || !elapsed_seconds.is_finite() || elapsed_seconds < 0.0 {
            return Err(TraversalModelError::TraversalModelFailure(
                "transit speed lookup requires finite, nonnegative elapsed time".into(),
            ));
        }
        let day = (self.start_weekday + (seconds / 86400.0).floor().rem_euclid(7.0) as usize) % 7;
        let hour = (seconds.rem_euclid(86400.0) / 3600.0).floor() as usize;
        let period = BIN_START_HOURS.partition_point(|start| *start <= hour) - 1;
        Ok(usize::from(day >= 5) * 6 + period)
    }

    fn set_speed(
        &self,
        speed: Velocity,
        state: &mut [StateVariable],
        state_model: &StateModel,
    ) -> Result<(), TraversalModelError> {
        let speed = self.speed_limit.map_or(speed, |limit| speed.min(limit));
        state_model.set_speed(state, fieldname::EDGE_SPEED, &speed)?;
        state_model.set_speed(state, TRANSIT_SPEED, &speed)?;
        Ok(())
    }
}

impl TraversalModel for TransitSpeedModel {
    fn name(&self) -> String {
        "Transit Speed Traces".into()
    }

    fn input_features(&self) -> Vec<InputFeature> {
        // trip_time is accumulated by the subsequent time model, not an
        // upstream dependency (which would create a dependency cycle).
        vec![]
    }

    fn output_features(&self) -> Vec<(String, StateVariableConfig)> {
        [
            (fieldname::EDGE_SPEED, SpeedUnit::KPH),
            (TRANSIT_SPEED, SpeedUnit::MPH),
        ]
        .iter()
        .map(|(name, unit)| {
            (
                name.to_string(),
                StateVariableConfig::Speed {
                    initial: self.traces.max_speed,
                    accumulator: false,
                    output_unit: Some(*unit),
                },
            )
        })
        .collect()
    }

    fn traverse_edge(
        &self,
        ctx: &EdgeFrontierContext,
        state: &mut Vec<StateVariable>,
        state_model: &StateModel,
    ) -> Result<(), TraversalModelError> {
        let elapsed = state_model.get_time(state, fieldname::TRIP_TIME)?;
        let bin = self.bin(elapsed.get::<second>())?;
        self.set_speed(
            self.traces.speed(ctx.edge.edge_id.0, bin)?,
            state,
            state_model,
        )
    }

    fn estimate_traversal(
        &self,
        _od: (&Vertex, &Vertex),
        state: &mut Vec<StateVariable>,
        _tree: &SearchTree,
        state_model: &StateModel,
    ) -> Result<(), TraversalModelError> {
        self.set_speed(self.traces.max_speed, state, state_model)
    }
}

fn parse_clock(time: &str, weekday: &str) -> Result<(f64, usize), TraversalModelError> {
    let invalid = || {
        TraversalModelError::BuildError(format!(
            "invalid transit speed start_time '{time}' or start_weekday '{weekday}'"
        ))
    };
    let parts: Vec<u32> = time
        .split(':')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .map_err(|_| invalid())?;
    if parts.len() != 3 || parts[1] >= 60 || parts[2] >= 60 {
        return Err(invalid());
    }
    let day = WEEKDAYS
        .iter()
        .position(|name| *name == weekday.to_lowercase())
        .ok_or_else(invalid)?;
    Ok((
        parts[0] as f64 * 3600.0 + parts[1] as f64 * 60.0 + parts[2] as f64,
        day,
    ))
}

pub struct TransitSpeedModelService {
    traces: Arc<SpeedTraces>,
}

impl TraversalModelService for TransitSpeedModelService {
    fn build(
        &self,
        query: &serde_json::Value,
    ) -> Result<Arc<dyn TraversalModel>, TraversalModelError> {
        let string = |key: &str, default: &str| -> Result<String, TraversalModelError> {
            match query.get(key) {
                None => Ok(default.to_string()),
                Some(value) => value.as_str().map(str::to_string).ok_or_else(|| {
                    TraversalModelError::BuildError(format!("'{key}' must be a string"))
                }),
            }
        };
        let (start_seconds, start_weekday) = parse_clock(
            &string("start_time", "12:00:00")?,
            &string("start_weekday", "monday")?,
        )?;
        let speed_limit = match query.get("speed_limit") {
            None => None,
            Some(value) => {
                let limit = value
                    .as_f64()
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .ok_or_else(|| {
                        TraversalModelError::BuildError(
                            "speed_limit must be positive and finite".into(),
                        )
                    })?;
                let unit = SpeedUnit::from_str(&string("speed_limit_unit", "")?).map_err(|_| {
                    TraversalModelError::BuildError(
                        "speed_limit requires a valid speed_limit_unit".into(),
                    )
                })?;
                Some(unit.to_uom(limit))
            }
        };
        Ok(Arc::new(TransitSpeedModel {
            traces: self.traces.clone(),
            start_seconds,
            start_weekday,
            speed_limit,
        }))
    }
}

#[derive(Deserialize)]
struct TraceConfig {
    format_version: u32,
    edge_trace_input_file: String,
    traces_input_file: String,
}

fn read_words<T>(
    path: &str,
    convert: impl Fn([u8; 4]) -> T,
) -> Result<Vec<T>, TraversalModelError> {
    let bytes = std::fs::read(path).map_err(|e| {
        TraversalModelError::BuildError(format!("cannot read transit speed file '{path}': {e}"))
    })?;
    if !bytes.len().is_multiple_of(4) {
        return Err(TraversalModelError::BuildError(format!(
            "truncated transit speed file '{path}'"
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| convert([b[0], b[1], b[2], b[3]]))
        .collect())
}

pub struct TransitSpeedModelBuilder {}

impl TraversalModelBuilder for TransitSpeedModelBuilder {
    fn build(
        &self,
        parameters: &serde_json::Value,
    ) -> Result<Arc<dyn TraversalModelService>, TraversalModelError> {
        let config: TraceConfig = serde_json::from_value(parameters.clone()).map_err(|e| {
            TraversalModelError::BuildError(format!("invalid transit speed trace config: {e}"))
        })?;
        if config.format_version != 2 {
            return Err(TraversalModelError::BuildError(
                "unsupported transit speed trace format; rebuild the Compass dataset".into(),
            ));
        }
        let traces = SpeedTraces::new(
            read_words(&config.edge_trace_input_file, u32::from_le_bytes)?,
            read_words(&config.traces_input_file, f32::from_le_bytes)?,
        )?;
        Ok(Arc::new(TransitSpeedModelService {
            traces: Arc::new(traces),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(time: &str, day: &str) -> TransitSpeedModel {
        let (start_seconds, start_weekday) = parse_clock(time, day).unwrap();
        TransitSpeedModel {
            traces: Arc::new(
                SpeedTraces::new(vec![0, 0], (1..=12).map(|n| n as f32).collect()).unwrap(),
            ),
            start_seconds,
            start_weekday,
            speed_limit: None,
        }
    }

    #[test]
    fn clock_advances_hour_and_weekday() {
        let friday = model("23:59:00", "friday");
        assert_eq!(friday.bin(0.0).unwrap(), 5);
        assert_eq!(friday.bin(60.0).unwrap(), 6);
        assert_eq!(model("23:00:00", "sunday").bin(3600.0).unwrap(), 0);
        assert_eq!(model("25:30:00", "friday").bin(0.0).unwrap(), 6);
        assert_eq!(model("12:00:00", "monday").bin(7.0 * 86400.0).unwrap(), 2);
        for (hour, expected) in [(0, 0), (6, 1), (9, 2), (15, 3), (19, 4), (22, 5)] {
            assert_eq!(
                model(&format!("{hour:02}:00:00"), "monday")
                    .bin(0.0)
                    .unwrap(),
                expected
            );
            if hour > 0 {
                assert_eq!(
                    model(&format!("{:02}:59:59", hour - 1), "monday")
                        .bin(0.0)
                        .unwrap(),
                    expected - 1
                );
            }
        }
        assert!(friday.bin(f64::NAN).is_err());
        assert!(friday.bin(-1.0).is_err());
    }

    #[test]
    fn validates_clock_and_traces() {
        for time in ["bad", "12:60:00", "12:00", "-1:00:00"] {
            assert!(parse_clock(time, "monday").is_err());
        }
        assert!(parse_clock("12:00:00", "holiday").is_err());
        assert!(SpeedTraces::new(vec![1], vec![10.0; BINS]).is_err());
        assert!(SpeedTraces::new(vec![0], vec![f32::NAN; BINS]).is_err());
        assert!(SpeedTraces::new(vec![0], vec![0.0; BINS]).is_err());
        assert!(SpeedTraces::new(vec![0], vec![10.0; BINS - 1]).is_err());
        let m = model("12:00:00", "monday");
        assert!((m.traces.speed(1, 2).unwrap().get::<mile_per_hour>() - 3.0).abs() < 1e-10);
        assert!(m.traces.speed(2, 2).is_err());
    }
}
