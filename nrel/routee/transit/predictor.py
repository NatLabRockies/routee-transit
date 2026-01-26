"""
Transit energy prediction from GTFS data.

This module provides the main GTFSEnergyPredictor class, which encapsulates
the complete workflow for predicting transit bus energy consumption from GTFS data.
"""

import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path

import nrel.routee.powertrain as pt
import numpy as np
import pandas as pd
from gtfsblocks import Feed, filter_blocks_by_route
from typing_extensions import Self, Union

from .deadhead_router import NetworkRouter
from .depot_deadhead import (
    create_depot_deadhead_stops,
    create_depot_deadhead_trips,
    get_default_depot_path,
    infer_depot_trip_endpoints,
)
from .grade.add_grade import run_gradeit_parallel
from .grade.tile_resolution import TileResolution
from .gtfs_processing import (
    extend_trip_traces,
    match_shape_to_osm,
    upsample_shape,
)
from .mid_block_deadhead import (
    create_mid_block_deadhead_stops,
    create_mid_block_deadhead_trips,
)
from .thermal_energy import add_HVAC_energy

logger = logging.getLogger(__name__)

MI_PER_KM = 0.6213712


class GTFSEnergyPredictor:
    """
    Predict transit bus energy consumption from GTFS data.

    This class provides a complete workflow for RouteE-Transit, including:
    - Loading and filtering GTFS data
    - Adding deadhead trips (between trips and to/from depot)
    - Matching shapes to road networks (OpenStreetMap by default)
    - Adding road grade information
    - Predicting energy consumption with RouteE-Powertrain models
    - Adding HVAC energy impacts

    The class is designed to be easily extended via inheritance. For example, a
    subclass can override network matching methods to use TomTom instead of OSM.

    Typical usage:
        >>> predictor = GTFSEnergyPredictor(
        ...     gtfs_path="data/gtfs",
        ...     # depot_path is optional - uses NTD depot locations by default
        ... )
        >>> predictor.load_gtfs_data()
        >>> predictor.filter_trips(date="2023-08-02", routes=["205"])
        >>> predictor.add_mid_block_deadhead()
        >>> predictor.add_depot_deadhead()  # Uses NTD depot locations
        >>> predictor.get_link_level_inputs(add_grade=True)
        >>> results = predictor.predict_energy(["Transit_Bus_Battery_Electric"])

    For extending with custom network data:
        >>> class TomTomEnergyPredictor(GTFSEnergyPredictor):
        ...     def _match_shapes_to_network(self, upsampled_shapes):
        ...         # Custom TomTom matching logic
        ...         return matched_shapes

    Attributes:
        gtfs_path (Path): Path to GTFS feed directory
        depot_path (Path | None): Path to depot shapefile directory
        n_processes (int): Number of parallel processes to use
        feed (Feed | None): Loaded GTFS feed object
        trips (pd.DataFrame | None): Trips DataFrame (initially all, can be filtered)
        shapes (pd.DataFrame | None): Shapes DataFrame for loaded trips
        matched_shapes (pd.DataFrame | None): Shapes matched to road network
        routee_inputs (pd.DataFrame | None): Link-level features for RouteE
        energy_predictions (dict[str, pd.DataFrame]): Energy predictions by vehicle model
    """

    def __init__(
        self,
        gtfs_path: str | Path,
        depot_path: str | Path | None = None,
        n_processes: int | None = None,
    ):
        """
        Initialize the GTFSEnergyPredictor.

        Args:
            gtfs_path: Path to directory containing GTFS feed files
            depot_path: Path to directory containing depot shapefile (Transit_Depot.shp).
                If None (default), uses depot locations from the National Transit Database's
                "Public Transit Facilities and Stations - 2023" dataset. This dataset covers
                depot/facility locations for transit agencies across the United States.
                Data source: https://data.transportation.gov/stories/s/gd62-jzra
            n_processes: Number of parallel processes for processing. Defaults to CPU count.
        """
        self.gtfs_path = Path(gtfs_path)
        if depot_path is None:
            self.depot_path = get_default_depot_path()
        else:
            self.depot_path = Path(depot_path)
        self.n_processes = n_processes if n_processes is not None else mp.cpu_count()

        # Internal state - populated by various methods
        self.feed: Feed | None = None
        self.trips: pd.DataFrame = pd.DataFrame()
        self.shapes: pd.DataFrame = pd.DataFrame()
        self.matched_shapes: pd.DataFrame = pd.DataFrame()
        self.routee_inputs: pd.DataFrame = pd.DataFrame()
        self.energy_predictions: dict[str, pd.DataFrame] = {}

        logger.info(f"Initialized GTFSEnergyPredictor for {self.gtfs_path}")

    def run(
        self,
        vehicle_models: list[str] | str,
        *,
        # Filtering options
        date: str | None = None,
        routes: list[str] | None = None,
        # Processing options
        add_mid_block_deadhead: bool = True,
        add_depot_deadhead: bool = True,
        add_grade: bool = True,
        # Energy prediction options
        add_hvac: bool = True,
        # Output options
        output_dir: str | Path | None = None,
        save_results: bool = True,
    ) -> pd.DataFrame:
        """
        Run the complete energy prediction pipeline with a single method call.

        This is a convenience method that chains together all processing steps:
        1. Load GTFS data
        2. Optionally filter trips (`date`, `routes`)
        3. Optionally add deadhead trips (`add_mid_block_deadhead`, `add_depot_deadhead`)
        4. Match shapes to OpenStreetMap network
        5. Optionally add road grade (`add_grade`)
        6. Predict energy consumption
        7. Optionally save results (`save_results`)

        For more control over individual steps, use the individual methods
        (load_gtfs_data, filter_trips, add_mid_block_deadhead, etc.).

        Parameters
        ----------
        vehicle_models : list[str] or str
            RouteE vehicle model name(s) to use for predictions.
            If a single string, will be converted to a list.
        date : str, optional
            Filter trips to a specific service date (format: "YYYY-MM-DD" or "YYYY/MM/DD").
            If None, all trips across all service dates are included.
        routes : list[str], optional
            Filter trips to specific route IDs. If None, all routes are included.
        add_mid_block_deadhead : bool, default=True
            Whether to add deadhead trips between consecutive revenue trips.
        add_depot_deadhead : bool, default=True
            Whether to add deadhead trips from/to depots at start/end of blocks.
            Requires depot_path to be set during initialization.
        add_grade : bool, default=True
            Whether to add road grade information using elevation data.
        add_hvac : bool, default=True
            Whether to add HVAC energy consumption based on ambient temperature.
        output_dir : str or Path, optional
            Directory to save results. If None and save_results=True, saves to
            current working directory.
        save_results : bool, default=True
            Whether to save results to files.

        Returns
        -------
        pd.DataFrame
            Trip-level energy predictions with columns for each vehicle model.

        Examples
        --------
        Simple usage - predict energy for all trips:

        >>> predictor = GTFSEnergyPredictor("data/gtfs")
        >>> results = predictor.run(vehicle_models=["BEB", "Diesel"])

        Filter to specific date and routes:

        >>> results = predictor.run(
        ...     vehicle_models="BEB",
        ...     date="2023-08-02",
        ...     routes=["205", "209"],
        ...     output_dir="reports/saltlake"
        ... )

        Minimal processing (no deadhead, no grade):

        >>> results = predictor.run(
        ...     vehicle_models="BEB",
        ...     add_mid_block_deadhead=False,
        ...     add_depot_deadhead=False,
        ...     add_grade=False,
        ...     save_results=False
        ... )
        """
        # Convert single model to list
        if isinstance(vehicle_models, str):
            vehicle_models = [vehicle_models]

        # Step 1: Load GTFS data
        self.load_gtfs_data()

        # Step 2: Filter trips if requested
        if date is not None or routes is not None:
            self.filter_trips(date=date, routes=routes)

        # Step 3: Add deadhead trips
        if add_mid_block_deadhead:
            self.add_mid_block_deadhead()

        if add_depot_deadhead:
            if self.depot_path is None:
                logger.warning(
                    "Cannot add depot deadhead: depot_path not provided during initialization"
                )
            else:
                self.add_depot_deadhead()

        # Step 4: Match shapes to network
        self.get_link_level_inputs()

        # Step 5: Add grade if requested
        if add_grade:
            self.add_road_grade()

        # Step 6: Predict energy
        self.predict_energy(vehicle_models=vehicle_models, add_hvac=add_hvac)

        # Step 7: Save results if requested
        if save_results:
            save_path = Path(output_dir) if output_dir else Path.cwd()
            self.save_results(save_path)

        # Return trip-level predictions
        return self.get_trip_predictions()

    def load_gtfs_data(self) -> Self:
        """
        Load GTFS data from the feed directory.

        This method reads the complete GTFS feed. Use filter_trips() afterwards
        if you want to restrict to specific dates or routes.

        Returns:
            Self for method chaining
        """
        logger.info("Loading GTFS data...")

        # Load feed with required columns
        req_cols = {
            "stop_times": [
                "arrival_time",
                "departure_time",
                "stop_id",
            ],
        }
        self.feed = Feed.from_dir(self.gtfs_path, columns=req_cols)

        agencies = self.feed.agency.agency_name.unique().tolist()
        logger.info(
            f"Feed includes {len(agencies)} agencies: {agencies}. "
            f"Total trips: {len(self.feed.trips)}, "
            f"shapes: {self.feed.shapes.shape_id.nunique()}"
        )

        # Initialize with all trips and shapes
        service_ids = self.feed.trips.service_id.unique().tolist()
        self.trips = self.feed.get_trips_from_sids(service_ids)

        shape_ids = self.trips.shape_id.unique()
        self.shapes = self.feed.shapes[self.feed.shapes.shape_id.isin(shape_ids)]

        logger.info(f"Loaded {len(self.trips)} trips and {len(shape_ids)} shapes")
        return self

    def filter_trips(
        self,
        date: str | None = None,
        routes: list[str] | None = None,
    ) -> Self:
        """
        Filter trips by date and/or routes.

        This method can be called after load_gtfs_data() to restrict the analysis
        to specific dates or routes. Can be called multiple times to refine filters.

        Args:
            date: Date to filter trips (format: "YYYY-MM-DD" or datetime object).
                If None, keeps all currently loaded trips.
            routes: List of route_short_name values to filter by.
                If None, keeps all currently loaded routes.

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If GTFS data hasn't been loaded yet
            ValueError: If no trips match the specified filters
        """
        if self.feed is None or self.trips.empty:
            raise RuntimeError("Must call load_gtfs_data() before filtering trips")

        logger.info(f"Filtering trips (date={date}, routes={routes})...")

        # Filter by date
        if date is not None:
            self.trips = self.feed.get_trips_from_date(date)
            if len(self.trips) == 0:
                raise ValueError(f"Feed does not contain any trips on {date}")

        # Filter by routes
        if routes is not None:
            self.trips = filter_blocks_by_route(
                trips=self.trips,
                routes=routes,
                route_column="route_short_name",
                route_method="exclusive",
            )
            if len(self.trips) == 0:
                raise ValueError("No trips found for the selected routes and date.")

        # Update shapes to match filtered trips
        shape_ids = self.trips.shape_id.unique()
        self.shapes = self.feed.shapes[self.feed.shapes.shape_id.isin(shape_ids)]

        logger.info(f"Filtered to {len(self.trips)} trips and {len(shape_ids)} shapes")

        # Add trip durations
        st_incl = self.feed.stop_times[
            self.feed.stop_times["trip_id"].isin(self.trips["trip_id"].unique())
        ]
        trip_times = st_incl.groupby("trip_id").agg(
            start_time=("arrival_time", "min"), end_time=("arrival_time", "max")
        )
        trip_times["trip_duration_minutes"] = (
            trip_times["end_time"] - trip_times["start_time"]
        ).dt.total_seconds() / 60
        self.trips = self.trips.merge(
            trip_times[["trip_duration_minutes"]],
            left_on="trip_id",
            right_index=True,
        )

        # Add number of times each trip is run
        sid_counts = (
            self.feed.get_service_ids_all_dates()
            .groupby("service_id")["date"]
            .count()
            .rename("trip_count")
        )
        self.trips = self.trips.merge(
            sid_counts,
            left_on="service_id",
            right_index=True,
        )

        return self

    def add_mid_block_deadhead(self) -> Self:
        """
        Add deadhead trips between consecutive trips in each block.

        This method creates synthetic trips representing buses traveling empty
        between the end of one scheduled trip and the start of the next trip
        within the same block. It updates the internal trips, shapes, and feed
        objects to include these deadhead trips.

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If GTFS data hasn't been loaded yet
        """
        if self.feed is None or self.trips.empty or self.shapes.empty:
            raise RuntimeError(
                "Must call load_gtfs_data() before adding deadhead trips"
            )

        logger.info("Adding between-trip deadhead trips...")

        # Create between-trip deadhead trips
        deadhead_trips = create_mid_block_deadhead_trips(
            self.trips, self.feed.stop_times
        )

        # Create stops and stop_times for deadhead trips
        deadhead_stop_times, deadhead_stops, deadhead_ods = (
            create_mid_block_deadhead_stops(self.feed, deadhead_trips)
        )

        # Remove ODs with same origin and destination (no travel needed)
        deadhead_ods = deadhead_ods[
            deadhead_ods.geometry_origin != deadhead_ods.geometry_destination
        ]

        # Generate shapes for deadhead trips
        all_points = pd.concat(
            [deadhead_ods["geometry_origin"], deadhead_ods["geometry_destination"]]
        )
        router = NetworkRouter.from_geometries(all_points)
        deadhead_shapes = router.create_deadhead_shapes(df=deadhead_ods, n_processes=1)

        # Filter deadhead trips to only those with generated shapes
        deadhead_trips = deadhead_trips[
            deadhead_trips["shape_id"].isin(deadhead_shapes["shape_id"].unique())
        ]

        # Update internal state
        self.trips = pd.concat([self.trips, deadhead_trips], ignore_index=True)
        self.shapes = pd.concat([self.shapes, deadhead_shapes], ignore_index=True)
        self.feed.trips = pd.concat(
            [self.feed.trips, deadhead_trips], ignore_index=True
        )
        self.feed.shapes = pd.concat(
            [self.feed.shapes, deadhead_shapes], ignore_index=True
        )
        self.feed.stop_times = pd.concat(
            [self.feed.stop_times, deadhead_stop_times], ignore_index=True
        )
        self.feed.stops = pd.concat(
            [self.feed.stops, deadhead_stops], ignore_index=True
        )

        logger.info(f"Added {len(deadhead_trips)} between-trip deadhead trips")
        return self

    def add_depot_deadhead(self) -> Self:
        """
        Add deadhead trips from depot to first stop and from last stop to depot.

        This method creates synthetic trips representing buses traveling from the
        depot to start their first scheduled trip of the day, and from their last
        stop back to the depot. Depot locations are matched from the National Transit
        Database's "Public Transit Facilities and Stations - 2023" dataset unless
        a custom depot_path was provided during initialization.

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If GTFS data hasn't been loaded or depot_path not specified
        """
        if self.feed is None or self.trips.empty or self.shapes.empty:
            raise RuntimeError(
                "Must call load_gtfs_data() before adding deadhead trips"
            )

        if self.depot_path is None:
            raise RuntimeError(
                "depot_path must be specified in __init__() to add depot deadhead trips"
            )

        logger.info("Adding depot deadhead trips...")

        # Create depot deadhead trip records
        deadhead_trips = create_depot_deadhead_trips(self.trips, self.feed.stop_times)

        # Infer depot locations for each block's first and last stops
        depot_shapefile = self.depot_path / "Transit_Depot.shp"
        first_stops_gdf, last_stops_gdf = infer_depot_trip_endpoints(
            self.trips, self.feed, depot_shapefile
        )

        # Create stop_times and stops for depot deadhead trips
        deadhead_stop_times, deadhead_stops = create_depot_deadhead_stops(
            first_stops_gdf, last_stops_gdf, deadhead_trips
        )

        # Generate shapes for trips from depot to first stop
        first_points = pd.concat(
            [
                first_stops_gdf["geometry_origin"],
                first_stops_gdf["geometry_destination"],
            ]
        )
        from_depot_router = NetworkRouter.from_geometries(first_points)
        from_depot_shapes = from_depot_router.create_deadhead_shapes(
            df=first_stops_gdf, n_processes=1
        )
        from_depot_shapes["shape_id"] = from_depot_shapes["shape_id"].apply(
            lambda x: f"from_depot_{x}"
        )

        # Generate shapes for trips from last stop to depot
        last_points = pd.concat(
            [last_stops_gdf["geometry_origin"], last_stops_gdf["geometry_destination"]]
        )
        to_depot_router = NetworkRouter.from_geometries(last_points)
        to_depot_shapes = to_depot_router.create_deadhead_shapes(
            df=last_stops_gdf, n_processes=1
        )
        to_depot_shapes["shape_id"] = to_depot_shapes["shape_id"].apply(
            lambda x: f"to_depot_{x}"
        )

        # Combine all depot deadhead shapes
        deadhead_shapes = pd.concat(
            [from_depot_shapes, to_depot_shapes], ignore_index=True
        )

        # Filter deadhead trips to only those with generated shapes
        deadhead_trips = deadhead_trips[
            deadhead_trips["shape_id"].isin(deadhead_shapes["shape_id"].unique())
        ]

        # Update internal state
        self.trips = pd.concat([self.trips, deadhead_trips], ignore_index=True)
        self.shapes = pd.concat([self.shapes, deadhead_shapes], ignore_index=True)
        self.feed.trips = pd.concat(
            [self.feed.trips, deadhead_trips], ignore_index=True
        )
        self.feed.shapes = pd.concat(
            [self.feed.shapes, deadhead_shapes], ignore_index=True
        )
        self.feed.stop_times = pd.concat(
            [self.feed.stop_times, deadhead_stop_times], ignore_index=True
        )
        self.feed.stops = pd.concat(
            [self.feed.stops, deadhead_stops], ignore_index=True
        )

        logger.info(f"Added {len(deadhead_trips)} depot deadhead trips")
        return self

    @staticmethod
    def aggregate_inputs_by_link(trips_ext: pd.DataFrame) -> pd.DataFrame:
        """After map matching all trips, aggregate the data by road link."""
        df_by_link = (
            trips_ext.groupby(by=["trip_id", "shape_id", "road_id"])
            .agg(
                start_lat=pd.NamedAgg("shape_pt_lat", "first"),
                start_lon=pd.NamedAgg("shape_pt_lon", "first"),
                end_lat=pd.NamedAgg("shape_pt_lat", "last"),
                end_lon=pd.NamedAgg("shape_pt_lon", "last"),
                geom=pd.NamedAgg("geom", "first"),
                start_timestamp=pd.NamedAgg("timestamp", "first"),
                end_timestamp=pd.NamedAgg("timestamp", "last"),
                kilometers=pd.NamedAgg("kilometers", "mean"),
                travel_time_minutes=pd.NamedAgg("travel_time", "mean"),
            )
            .reset_index()
        )
        df_by_link["travel_time_minutes"] /= 60
        return df_by_link

    def get_link_level_inputs(self) -> Self:
        """
        Match GTFS shapes to road network and prepare RouteE inputs.

        This method performs the following steps:
        1. Upsamples shapes to ~1 Hz GPS traces
        2. Matches shapes to OpenStreetMap road network
        3. Extends trips with stop and schedule information
        4. Aggregates data at road link level
        5. Optionally adds road grade information

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If GTFS data hasn't been loaded yet
        """
        if self.feed is None or self.trips.empty or self.shapes.empty:
            raise RuntimeError(
                "Must call load_gtfs_data() before matching shapes to network"
            )

        logger.info("Matching shapes to road network...")

        # Step 1: Upsample all shapes to ~1 Hz
        shape_groups = [group for _, group in self.shapes.groupby("shape_id")]
        with mp.Pool(self.n_processes) as pool:
            upsampled_shapes = pool.map(upsample_shape, shape_groups)

        logger.debug(f"Upsampled {len(shape_groups)} shapes")

        # Step 2: Match to network
        matched_shapes = self._match_shapes_to_network(upsampled_shapes)
        self.matched_shapes = matched_shapes

        logger.info("Finished map matching")

        # Step 3: Extend trip data with stop and schedule information
        trips_ext = extend_trip_traces(
            trips_df=self.trips,
            matched_shapes_df=matched_shapes,
            feed=self.feed,
            add_stop_flag=False,
            n_processes=self.n_processes,
        )

        # Step 4: Aggregate data at road link level
        self.routee_inputs = self.aggregate_inputs_by_link(trips_ext)

        return self

    def add_road_grade(
        self,
        tile_resolution: TileResolution | str = TileResolution.ONE_THIRD_ARC_SECOND,
    ) -> Self:
        if self.routee_inputs.empty:
            raise RuntimeError("Must run get_link_level_inputs() before adding grade")

        logger.info("Adding road grade information...")
        trip_groups = [t_df for _, t_df in self.routee_inputs.groupby("trip_id")]
        self.routee_inputs = run_gradeit_parallel(
            trip_dfs_list=trip_groups,
            tile_resolution=tile_resolution,
            n_processes=self.n_processes,
        )

        logger.info(
            f"Generated {len(self.routee_inputs)} link-level features for RouteE"
        )
        return self

    def _match_shapes_to_network(
        self, upsampled_shapes: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Match upsampled shapes to OpenStreetMap road network.

        Args:
            upsampled_shapes: List of upsampled shape DataFrames

        Returns:
            DataFrame with matched shapes including network attributes
        """
        # Run map matching in parallel for each shape
        with mp.Pool(self.n_processes) as pool:
            matched_shapes = pool.map(match_shape_to_osm, upsampled_shapes)

        return pd.concat(matched_shapes)

    def predict_energy(
        self,
        vehicle_models: Union[str, list[str], Path, list[Path]],
        add_hvac: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Predict energy consumption using RouteE-Powertrain vehicle models.

        This method runs energy prediction for one or more vehicle models and
        returns both link-level and trip-level results. Results are stored
        internally and also returned.

        Args:
            vehicle_models: RouteE vehicle model name(s) or path(s). Can be:
                - Single model name: "Transit_Bus_Battery_Electric"
                - List of models: ["BEB", "Diesel_2016_Bus"]
                - Path to custom model JSON
                - List of paths to custom model JSON
            add_hvac: Whether to add HVAC energy consumption to trip-level results

        Returns:
            Dictionary with keys:
                - 'link': DataFrame with link-level predictions for all models
                - 'trip': DataFrame with trip-level predictions for all models
                - '<model_name>_link': Link-level predictions for specific model
                - '<model_name>_trip': Trip-level predictions for specific model

        Raises:
            RuntimeError: If routee_inputs haven't been generated yet
        """
        if self.routee_inputs.empty:
            raise RuntimeError(
                "Must call get_link_level_inputs() before predicting energy"
            )

        vehicle_models_list: list[str | Path]
        if isinstance(vehicle_models, (str, Path)):
            vehicle_models_list = [vehicle_models]
        elif isinstance(vehicle_models, list):
            # Create a new list to satisfy mypy type variance rules
            vehicle_models_list = [item for item in vehicle_models]
        else:
            raise ValueError(
                f"Incompatible type for vehicle_models: {type(vehicle_models)}"
            )

        logger.info(
            f"Predicting energy for {len(vehicle_models_list)} vehicle model(s)..."
        )

        all_link_results: list[pd.DataFrame] = []
        all_trip_results: list[pd.DataFrame] = []

        for model in vehicle_models_list:
            logger.info(f"Processing model: {model}")

            # Run link-level prediction
            link_results = self._predict_for_model(model)
            link_results["vehicle"] = str(model)

            # Aggregate to trip level
            trip_results = self._aggregate_predictions_by_trip(link_results, str(model))

            # Optionally add HVAC to trip-level results
            if add_hvac:
                logger.info("Adding HVAC energy impacts...")
                if self.feed is None or self.trips.empty:
                    raise RuntimeError(
                        "Feed and trips must be loaded to add HVAC energy"
                    )
                hvac_energy = add_HVAC_energy(self.feed, self.trips)
                trip_results = trip_results.merge(hvac_energy, on="trip_id", how="left")
                # Add HVAC energy to powertrain energy
                trip_results["energy_used"] += trip_results["hvac_energy_kWh"]

            # Store results
            self.energy_predictions[f"{model}_link"] = link_results
            self.energy_predictions[f"{model}_trip"] = trip_results
            all_link_results.append(link_results)
            all_trip_results.append(trip_results)

        # Combine all models
        if all_link_results:
            self.energy_predictions["link"] = pd.concat(
                all_link_results, ignore_index=True
            )
        if all_trip_results:
            self.energy_predictions["trip"] = pd.concat(
                all_trip_results, ignore_index=True
            )

        logger.info("Energy prediction complete")
        return self.energy_predictions

    def _predict_for_model(self, model: str | Path) -> pd.DataFrame:
        """
        Run energy prediction for a single vehicle model.

        Args:
            model: RouteE vehicle model name or path

        Returns:
            DataFrame with link-level energy predictions
        """
        if self.routee_inputs.empty:
            raise RuntimeError("No RouteE inputs available for predictions.")

        # Split data into batches for multiprocessing
        link_batches = np.array_split(self.routee_inputs, self.n_processes)
        # Run prediction in parallel
        # Note: We pass the model path/name, not the loaded model, to avoid pickling issues
        predict_partial = partial(self._predict_trip_with_model, model_path=model)
        with mp.Pool(self.n_processes) as pool:
            predictions = pool.map(predict_partial, link_batches)

        all_predictions = pd.concat(predictions, ignore_index=True)

        results = pd.concat([self.routee_inputs.reset_index(), all_predictions], axis=1)

        # Select relevant columns
        pred_cols = list(all_predictions.columns)
        result_cols = [
            "trip_id",
            "shape_id",
            "road_id",
            "kilometers",
            "travel_time_minutes",
        ]
        if "grade" in results.columns:
            result_cols.append("grade")
        result_cols.extend(pred_cols)

        return results[result_cols]

    def _predict_trip_with_model(
        self, trip_df: pd.DataFrame, model_path: str | Path
    ) -> pd.DataFrame:
        """
        Predict energy for a single trip, loading the model within the worker process.

        This method loads the RouteE model inside each worker to avoid pickling issues
        with ONNX runtime models in multiprocessing.

        Args:
            trip_df: DataFrame with trip link data
            model_path: Path or name of RouteE model to load

        Returns:
            DataFrame with energy predictions
        """
        # Load model in this worker process (avoids pickling)
        routee_model = pt.load_model(model_path)

        # Calculate speed and convert to mph
        trip_df["miles"] = MI_PER_KM * trip_df["kilometers"]
        trip_df["gpsspeed"] = trip_df["miles"] / (trip_df["travel_time_minutes"] / 60)

        # Run prediction
        result = routee_model.predict(links_df=trip_df)
        return result.copy()

    def _aggregate_predictions_by_trip(
        self, link_results: pd.DataFrame, vehicle_name: str
    ) -> pd.DataFrame:
        """
        Aggregate link-level predictions to trip level.

        Args:
            link_results: DataFrame with link-level predictions
            vehicle_name: Name of vehicle model

        Returns:
            DataFrame with trip-level aggregated results
        """
        vehicle_units = {
            "Transit_Bus_Diesel": {
                "routee_unit": "gallons",
                "unit_name": "gallons_diesel",
            },
            "Transit_Bus_Battery_Electric": {"routee_unit": "kWhs", "unit_name": "kWh"},
        }

        # Determine which energy columns to aggregate
        agg_col = vehicle_units[vehicle_name]["routee_unit"]

        if agg_col not in link_results.columns:
            raise ValueError(
                f"No energy prediction found with unit {agg_col} for {vehicle_name}."
                f"Results columns: {link_results.columns}"
            )

        energy_by_trip = link_results.groupby("trip_id").agg(
            {"kilometers": "sum", agg_col: "sum"}
        )

        energy_by_trip["miles"] = MI_PER_KM * energy_by_trip["kilometers"]
        energy_by_trip["vehicle"] = vehicle_name
        energy_by_trip["energy_used"] = energy_by_trip[agg_col]
        energy_by_trip["energy_unit"] = vehicle_units[vehicle_name]["unit_name"]

        return energy_by_trip.drop(columns=["kilometers", agg_col])

    def get_link_predictions(self, vehicle_model: str | None = None) -> pd.DataFrame:
        """
        Get link-level energy predictions.

        Args:
            vehicle_model: Specific model name, or None for all models

        Returns:
            DataFrame with predictions, or None if not yet computed
        """
        key = f"{vehicle_model}_link" if vehicle_model else "link"
        if key not in self.energy_predictions:
            raise KeyError(
                f"No link-level predictions found for '{key}'. "
                "Call predict_energy() before accessing results."
            )
        return self.energy_predictions[key]

    def get_trip_predictions(self, vehicle_model: str | None = None) -> pd.DataFrame:
        """
        Get trip-level energy predictions.

        Args:
            vehicle_model: Specific model name, or None for all models

        Returns:
            DataFrame with predictions

        Raises:
            KeyError: If predictions have not been generated yet
        """
        key = f"{vehicle_model}_trip" if vehicle_model else "trip"
        if key not in self.energy_predictions:
            raise KeyError(
                f"No trip-level predictions found for '{key}'. "
                "Call predict_energy() before accessing results."
            )
        return self.energy_predictions[key]

    def save_results(
        self,
        output_dir: str | Path,
        save_geometry: bool = True,
        save_inputs: bool = True,
    ) -> None:
        """
        Save prediction results to CSV files.

        Args:
            output_dir: Directory to save results
            save_geometry: Whether to save link geometry separately
            save_inputs: Whether to save RouteE input features

        Raises:
            RuntimeError: If no predictions have been generated yet
        """
        if not self.energy_predictions:
            raise RuntimeError("No predictions to save. Call predict_energy() first.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save link-level predictions
        if "link" in self.energy_predictions:
            link_df = self.energy_predictions["link"].copy()

            # Optionally save geometry separately
            if save_geometry and "geom" in link_df.columns:
                geom_df = pd.concat([link_df["road_id"], link_df.pop("geom")], axis=1)
                geom_df = geom_df.drop_duplicates(subset="geom")
                geom_path = output_path / "link_geometry.csv"
                geom_df.to_csv(geom_path, index=False)
                logger.info(f"Saved link geometry to {geom_path}")

            link_path = output_path / "link_energy_predictions.csv"
            link_df.to_csv(link_path, index=False)
            logger.info(f"Saved link predictions to {link_path}")

        # Save trip-level predictions
        if "trip" in self.energy_predictions:
            trip_path = output_path / "trip_energy_predictions.csv"
            self.energy_predictions["trip"].to_csv(trip_path, index=False)
            logger.info(f"Saved trip predictions to {trip_path}")

        # Save RouteE inputs
        if save_inputs and self.routee_inputs is not None:
            inputs_df = self.routee_inputs.copy()
            if "geom" in inputs_df.columns:
                inputs_df = inputs_df.drop(columns="geom")
            inputs_path = output_path / "routee_inputs.csv"
            inputs_df.to_csv(inputs_path, index=False)
            logger.info(f"Saved RouteE inputs to {inputs_path}")
