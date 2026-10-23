# Data Requirements

(gtfs-reqs)=
## General Transit Feed Specification (GTFS)
RouteE-Transit relies on input data in the static [GTFS format](https://gtfs.org/) to generate energy predictions for each trip. Some features, such as complete shape traces for each trip, are optional per the GTFS standard but mandatory for RouteE-Transit analysis. In addition to all files and columns required by the GTFS standard, RouteE-Transit requires the following GTFS tables and fields:

- *trips.txt*: `trip_id` and `shape_id` for all trips
- *shapes.txt*: complete shape traces (`shape_pt_lat`, `shape_pt_lon`, `shape_pt_sequence`) for each trip
- *stop_times.txt*: all stop time data for each `trip_id` included
- *stops.txt*: stop coordinates for any stop served on any trip to be analyzed


## Elevation Data
Road grade is computed automatically by RouteE-Compass using OSMnx to fetch elevation data from the USGS [National Map](https://www.usgs.gov/core-science-systems/national-geospatial-program/national-map). No manual download or configuration is required. For large geographic areas, the initial OSMnx network download may take additional time, but results are cached for subsequent runs.

(speed-model)=
## Transit Speed Model

RouteE-Transit predicts the average *moving* speed of a transit bus on every
OpenStreetMap road link with a machine-learning model trained on archived GTFS
Realtime (GTFS-RT) vehicle positions. These speeds replace OSM posted speeds
for map matching, deadhead routing, travel time, and energy prediction. The
model is bundled with the package
(`routee/transit/resources/speed_models/random-forest/`), so no realtime data
is needed to run predictions.

The model is a single pooled random forest intended to generalize to agencies
it has never seen. It uses only features available for any U.S. road network
plus a static GTFS feed. It is a proof of concept: see
{ref}`accuracy <speed-model-accuracy>` and
{ref}`limitations <speed-model-limitations>` below.

### Inputs and time bins

Each link gets 12 speeds: six local-time periods for weekdays, followed by the
same six for weekends.

| Period | Local time |
|---|---|
| Overnight | 00:00–06:00 |
| Morning peak | 06:00–09:00 |
| Midday | 09:00–15:00 |
| Afternoon peak | 15:00–19:00 |
| Evening | 19:00–22:00 |
| Late evening | 22:00–24:00 |

Boundaries belong to the period starting at that time. The model has 24 inputs:

- **Road attributes from OSM:** posted speed (mph), lane count, link length (km),
  signed and absolute grade, and FHWA-style functional class derived from the
  OSM `highway` tag (freeway, principal arterial, minor arterial, collector,
  local; one-hot encoded).
- **Transit context:** number of GTFS stops nearest to the link.
- **Time:** six period indicators and a weekday/weekend indicator.
- **Missing-value flags:** one per numeric road attribute. Missing values are
  imputed with training-set medians stored in the model manifest.

Scheduled GTFS speed, agency identity, route identity, and road identity are
deliberately *not* inputs. Scheduled speed is unavailable on arbitrary network
links, and the identifiers would not generalize to new agencies.

### How speeds are used at runtime

During Compass dataset generation, RouteE-Transit runs the ONNX speed
model over all road links in bounded batches. Predictions are floored at
1 mph; invalid model outputs stop preprocessing. Speeds are rounded to 0.1 mph,
then identical 12-value traces share a trace ID. The cache stores a dense
little-endian uint32 edge-to-trace mapping in `transit_speed_edge_traces.bin`
and consecutive 12-value little-endian float32 mph traces in
`transit_speed_traces.bin`. Storage is 4 bytes per edge plus 48 bytes per
unique trace, excluding file metadata.

A single Compass application shares these immutable tables across queries.
Map matching, deadhead routing, elapsed travel time, and energy calculations
all use the same lookup speed (`edge_speed`, also exposed as `transit_speed`).
`edge_speed` is in km/h and `transit_speed` is in mph. The lookup selects the
bin at entry to each edge using departure time plus accumulated travel time,
including turn delays. Hours and weekdays roll forward across midnight,
including GTFS departures beyond 24:00:00. Speeds are constant within each
period and are not interpolated within an edge. A* estimates use the maximum
trace speed, subject to a query's speed limit.

Energy queries are not expanded across service dates: path calculation runs
once per matched shape and vehicle model. Trips sharing a shape use its
earliest departure. Without a date filter, the representative service day
defaults to Monday. Calendar expansion and HVAC modeling happen afterward. Energy
estimates are therefore representative per shape, not trip-specific by
departure time.

ONNX Runtime is a runtime dependency because it runs during preprocessing.
Rust traversal does not use it. Cached datasets are reused when `output_dir`
is provided and `overwrite=False`. Datasets with an old speed configuration,
missing trace files, or changed model/manifest files are rebuilt
automatically. Adding vehicle models to an otherwise valid cache preserves the
traces without repeating inference. To use a different trained model, pass
`speed_model_dir` to `GTFSEnergyPredictor`.

### Training data

Training data came from the public [gtfsrt.io](https://gtfsrt.io) archive of
GTFS-RT feeds, read from its Parquet bucket. For each agency, the archived
vehicle positions were paired with the static GTFS schedule snapshot in effect
on the same dates. Three full Monday–Sunday weeks were sampled per agency,
spread across its archived date range. The archive dates used span January to
September 2026.

Eleven bus agencies with both vehicle positions and schedules in the archive
were attempted. Nine produced usable data:

| Agency | Archive slug | Days | Road links | Training bins | Trip-link speeds | GPS points |
|---|---|---:|---:|---:|---:|---:|
| AC Transit (Oakland, CA) | `actransit` | 21 | 6,479 | 28,949 | 529,810 | 2.26 M |
| Broward County Transit (FL) | `broward` | 21 | 7,424 | 48,930 | 1,162,614 | 4.46 M |
| Charlotte Area Transit System (NC) | `cats` | 21 | 6,165 | 35,938 | 852,177 | 3.19 M |
| CityBus of Greater Lafayette (IN) | `citybus-lafayette` | 20 | 876 | 3,741 | 84,728 | 0.32 M |
| Greater Dayton RTA (OH) | `dayton` | 21 | 3,078 | 16,338 | 248,260 | 1.10 M |
| Greater Cleveland RTA (OH) | `gcrta` | 21 | 8,176 | 53,182 | 1,675,578 | 7.41 M |
| Metro Transit (Madison, WI) | `metro-transit-madison` | 17 | 2,576 | 11,804 | 245,883 | 0.98 M |
| Mountain Line (Missoula, MT) | `mountainline` | 21 | 1,081 | 4,709 | 92,722 | 0.35 M |
| Western Reserve Transit Authority (Youngstown, OH) | `wrta-youngstown` | 18 | 1,633 | 5,538 | 80,743 | 0.42 M |
| **Total** | | **181** | **37,488** | **209,129** | **4,972,515** | **20.5 M** |

*Trip-link speeds* and *GPS points* count only cleaned observations
contributing to training bins. Big Blue Bus (Santa Monica, CA) was excluded
because its realtime trip IDs did not match its static GTFS. Metro Transit
(Minneapolis, MN) was excluded because its OSM network download failed.

### Estimating link speeds from GTFS-RT

`scripts/gtfs_realtime/archive_speeds.py` estimates one speed per trip and
road link from each trip's vehicle positions, using the shared library in
`realtime_speeds.py`:

1. **Pair realtime with schedules.** Download vehicle positions for a day and
   join them to the matching static GTFS `trips.txt`. Days where fewer than 50%
   of realtime trip IDs match the static feed are skipped.
2. **Clean each trip's trace.** Drop duplicate reports and sort
   chronologically.
3. **Map match.** Match the trip's GPS trace to the OSM network with
   RouteE-Compass. Assign each GPS point to a matched edge, and project it onto
   the route to get cumulative distance. Distance is forced to be
   non-decreasing because the bus only moves forward along its route.
4. **Interpolate link traversal times.** Linearly interpolate timestamps at
   each link boundary from cumulative distance. Travel time on a link is exit
   time minus entry time.
5. **Remove dwell time.** Consecutive points on the same link less than 10 m
   apart and more than 15 s apart count as dwelling. Moving speed is link
   length divided by (travel time − dwell time).
6. **Attach features.** Record OSM attributes, grade, and the number of GTFS
   stops projected onto the link. Links with at least two GPS points are
   labeled `observed`; others are `interpolated`.

Output is one row per trip and link, partitioned as
`link_observations/agency=<slug>/date=<YYYY-MM-DD>/`.

### Model training

`scripts/gtfs_realtime/fit_poc_speed_model.py` trains the bundled model:

1. **Filter observations.** Keep only `observed` links with a finite moving
   speed between 1 and 65 mph. Remove outliers outside 1.5 × IQR per road link
   (for links with at least four observations).
2. **Assign local time bins.** Convert archive timestamps, which are naive
   UTC, to each agency's local time zone, then assign period and weekday/weekend
   from the link entry time.
3. **Aggregate.** Group by road link, period, and day type. The target is the
   GPS-point-weighted mean moving speed (`mph_moving_mean`). Keep groups with
   at least three trip-link observations.
4. **Weight agencies equally.** Sample weights are point-weighted within each
   agency and normalized so each agency has equal total influence.
5. **Fit.** Train a `RandomForestRegressor` with 100 trees, maximum depth 8,
   minimum leaf size 10, `max_features=0.5`, and `random_state=42`. There is
   no hyperparameter search.
6. **Evaluate, then refit.** Evaluate with three agency-grouped folds,
   learning imputation medians from training agencies only. Then refit the
   final model on all nine agencies.
7. **Export.** Export to ONNX with `skl2onnx`, validate against
   scikit-learn predictions, and write a JSON manifest. The manifest records
   input order, medians, the functional class mapping, time bin boundaries, and
   training metadata.

See the [training instructions](https://github.com/NatLabRockies/routee-transit/blob/main/scripts/gtfs_realtime/README.md#speed-model-training-pipeline-current)
for commands.

(speed-model-accuracy)=
### Accuracy

Accuracy is measured on agencies held out of training entirely. This reflects
the main use case: predicting speeds for an agency without realtime data.
Errors are GPS-point-weighted over each held-out agency's road/period/day-type
bins.

| Held-out agency | MAE (mph) | RMSE (mph) |
|---|---:|---:|
| AC Transit | 4.90 | 6.61 |
| Broward County Transit | 4.17 | 6.76 |
| Charlotte Area Transit System | 4.31 | 6.52 |
| CityBus of Greater Lafayette | 3.14 | 4.81 |
| Greater Dayton RTA | 3.76 | 6.05 |
| Greater Cleveland RTA | 4.31 | 5.92 |
| Metro Transit (Madison) | 3.16 | 4.88 |
| Mountain Line | 3.58 | 4.68 |
| Western Reserve Transit Authority | 3.88 | 5.68 |
| **Equal-agency average** | **3.91** | **5.77** |

For scale, the mean target across training bins is 13.4 mph with a standard
deviation of 8.6 mph.

Additional experiments used the same folds:

| Variant | MAE (mph) | RMSE (mph) |
|---|---:|---:|
| **Bundled model** (12 bins) | **3.91** | **5.77** |
| Same model without time features | 3.91 | 5.74 |
| `max_features=1.0` | 4.00 | 5.89 |
| `max_features=1.0`, depth 12, leaf size 20 | 4.17 | 6.06 |
| Previous variant + stop density (stops/km) | 4.14 | 6.03 |
| Bundled settings + stop density | 3.91 | 5.78 |

Time-of-day features do not measurably improve accuracy on held-out agencies.
Larger forests overfit to training agencies. The bundled model keeps the 12
bins so the runtime supports time-varying speeds, but its predictions vary
little by time of day.

Random-forest feature importances (mean decrease in impurity) for the bundled
model:

| Feature | Importance |
|---|---:|
| Link length | 81.3% |
| GTFS stop count | 6.2% |
| Absolute grade | 4.2% |
| Signed grade | 3.7% |
| Functional class (5 indicators) | 1.9% |
| Lane count | 1.1% |
| Posted speed | 0.9% |
| Missing-value flags | 0.5% |
| Time period (6 indicators) | 0.1% |
| Weekday/weekend | < 0.1% |

Impurity-based importances favor continuous features and split credit among
correlated ones, such as the two grade features. They show what the forest
splits on, not how much real-world speed each feature explains.

(speed-model-limitations)=
### Limitations

- **Small, regional sample.** Nine mostly small and mid-sized agencies, five
  of them in the Eastern time zone, over three sampled weeks each. Large,
  congested systems and rail-like bus operations are underrepresented.
- **Link length dominates.** Short links tend to have lower measured
  speeds, partly because of acceleration and deceleration near intersections and
  stops. Some of the effect may also come from how GPS points are assigned to
  links and boundary times are interpolated.
- **Weak temporal signal.** Peak and off-peak speeds are nearly identical in
  predictions.
- **Moving speed only.** Dwell time at stops is excluded from the target.
- **Training/inference differences.** Training stop counts come from stops
  projected onto each trip's matched route. At inference, every GTFS stop is
  assigned to its nearest network edge. The model is trained only on links that
  buses traversed but is applied to every link, including those used only by
  deadhead routing.
- **Representative departure time.** As described above, each shape uses one
  representative departure time rather than each trip's own time.