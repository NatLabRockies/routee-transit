"""
Transit energy prediction from GTFS data.

This module provides the main GTFSEnergyPredictor class, which encapsulates
the complete workflow for predicting transit bus energy consumption from GTFS data.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import pandas as pd
from gtfsblocks import Feed, filter_blocks_by_route
from nrel.routee.compass import CompassApp
from nrel.routee.compass.io.generate_dataset import GeneratePipelinePhase
from nrel.routee.compass.map_matching.utils import match_result_to_geopandas

from routee.transit.deadhead_router import create_deadhead_shapes
from routee.transit.depot_deadhead import (
    create_depot_deadhead_stops,
    create_depot_deadhead_trips,
    get_default_depot_path,
    infer_depot_trip_endpoints,
)
from routee.transit.gtfs_processing import (
    extend_trip_traces,
    upsample_shape,
)
from routee.transit.mid_block_deadhead import (
    create_mid_block_deadhead_stops,
    create_mid_block_deadhead_trips,
)
from routee.transit.thermal_energy import add_HVAC_energy

logger = logging.getLogger(__name__)

MI_PER_KM = 0.6213712
MILES_PER_GALLON_TO_KWH = 33.7  # 1 gallon gasoline equivalent = 33.7 kWh

# Vehicle model configuration: maps model names to their CompassApp traversal summary fields
VEHICLE_MODELS: dict[str, dict[str, str]] = {
    "Transit_Bus_Battery_Electric": {
        "energy_field": "trip_energy_electric",
        "unit": "kWh",
    },
    "Transit_Bus_Diesel": {
        "energy_field": "trip_energy_liquid",
        "unit": "gallons_diesel",
    },
}


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
        >>> predictor.get_link_level_inputs()
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
        compass_app: CompassApp | None = None,
        output_dir: str | Path | None = None,
        vehicle_models: list[str] | None = None,
        overwrite: bool = True,
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
            compass_app: An optional pre-initialized CompassApp instance.
        """
        self.gtfs_path = Path(gtfs_path)
        if depot_path is None:
            self.depot_path = get_default_depot_path()
        else:
            self.depot_path = Path(depot_path)
        self.n_processes = n_processes if n_processes is not None else mp.cpu_count()
        self.app = compass_app
        self.output_dir = Path(output_dir) if output_dir else None
        self.vehicle_models = vehicle_models
        self.overwrite = overwrite

        # Internal state - populated by various methods
        self.feed: Feed | None = None
        self.trips: pd.DataFrame = pd.DataFrame()
        self.shapes: pd.DataFrame = pd.DataFrame()
        self.matched_shapes: pd.DataFrame = pd.DataFrame()
        self.routee_inputs: pd.DataFrame = pd.DataFrame()
        self.energy_predictions: dict[str, pd.DataFrame] = {}
        self._bbox: tuple[float, float, float, float] | None = None

        logger.info(f"Initialized GTFSEnergyPredictor for {self.gtfs_path}")

    def add_trip_times(self) -> None:
        """Add trip time columns to self.trips"""
        # Make sure trips are available
        if self.feed is None:
            raise ValueError("Must call load_gtfs_data() before add_trip_times()")
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

        # Convert start/end times to GTFS-style strings
        def format_timedelta(td: pd.Timedelta) -> str:
            if pd.isna(td):
                return ""
            total_seconds = int(td.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours:02}:{minutes:02}:{seconds:02}"

        trip_times["start_time"] = trip_times["start_time"].apply(format_timedelta)
        trip_times["end_time"] = trip_times["end_time"].apply(format_timedelta)

        self.trips = self.trips.merge(
            trip_times[["start_time", "end_time", "trip_duration_minutes"]],
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

    def run(
        self,
        *,
        # Filtering options
        date: str | None = None,
        routes: list[str] | None = None,
        # Processing options
        add_mid_block_deadhead: bool = True,
        add_depot_deadhead: bool = True,
        # Energy prediction options
        add_hvac: bool = True,
        save_results: bool = True,
    ) -> pd.DataFrame:
        """
        Run the complete energy prediction pipeline with a single method call.

        This is a convenience method that chains together all processing steps:
        1. Load GTFS data
        2. Optionally filter trips (`date`, `routes`)
        3. Optionally add deadhead trips (`add_mid_block_deadhead`, `add_depot_deadhead`)
        4. Run map matching and predict energy consumption using CompassApp
        5. Optionally save results (`save_results`)

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
        add_hvac : bool, default=True
            Whether to add HVAC energy consumption based on ambient temperature.
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

        Minimal processing (no deadhead, no HVAC):

        >>> results = predictor.run(
        ...     vehicle_models="BEB",
        ...     add_mid_block_deadhead=False,
        ...     add_depot_deadhead=False,
        ...     save_results=False
        ... )
        """

        # Step 1: Load GTFS data
        self.load_gtfs_data()

        # Step 2: Filter trips if requested
        if date is not None or routes is not None:
            self.filter_trips(date=date, routes=routes)

        # Add start time, end time, and duration of each trip
        self.add_trip_times()

        # Step 3: Prepare deadhead metadata (gather O-D pairs without routing)
        # This allows us to compute the full bounding box before loading CompassApp
        extra_geoms: list[gpd.GeoDataFrame | pd.DataFrame] = []
        mid_block_metadata = None
        depot_metadata = None

        if add_mid_block_deadhead:
            mid_block_metadata = self._prepare_mid_block_deadhead()
            if mid_block_metadata is not None:
                extra_geoms.append(mid_block_metadata["deadhead_ods"])

        if add_depot_deadhead:
            if self.depot_path is None:
                logger.warning(
                    "Cannot add depot deadhead: depot_path not provided during initialization"
                )
            else:
                depot_metadata = self._prepare_depot_deadhead()
                if depot_metadata is not None:
                    extra_geoms.extend(
                        [
                            depot_metadata["first_stops_gdf"],
                            depot_metadata["last_stops_gdf"],
                        ]
                    )

        # Step 4: Load CompassApp once with comprehensive bounding box
        self.load_compass_app(extra_geoms=extra_geoms if extra_geoms else None)

        # Step 5: Route deadhead trips using the loaded app
        if mid_block_metadata is not None:
            self._route_mid_block_deadhead(mid_block_metadata)

        if depot_metadata is not None:
            self._route_depot_deadhead(depot_metadata)

        # Step 6: Predict energy using CompassApp
        self.predict_energy(add_hvac=add_hvac)

        # Step 7: Save results if requested
        if save_results:
            self.save_results()

        # Return trip-level predictions
        return self.get_trip_predictions()

    def load_gtfs_data(self) -> "GTFSEnergyPredictor":
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
        self.trips["trip_type"] = "service"

        shape_ids = self.trips.shape_id.unique()
        self.shapes = self.feed.shapes[self.feed.shapes.shape_id.isin(shape_ids)]

        logger.info(f"Loaded {len(self.trips)} trips and {len(shape_ids)} shapes")
        return self

    def load_compass_app(
        self,
        buffer_deg: float = 0.05,
        n_processes: int | None = None,
        extra_geoms: list[gpd.GeoDataFrame | pd.DataFrame] | None = None,
    ) -> None:
        """
        Initialize the CompassApp using the bounding box of the loaded shapes.

        Args:
            buffer_deg: Buffer in degrees to add to the bounding box.
            n_processes: Number of processes for parallelism.
            extra_geoms: Optional list of GeoDataFrames or DataFrames with geometry to include in bbox.
        """
        if n_processes is not None:
            self.n_processes = n_processes

        if self.shapes.empty:
            raise ValueError(
                "Must load GTFS data (and shapes) before initializing CompassApp"
            )

        # Calculate requested bounding box
        lons = [
            float(self.shapes.shape_pt_lon.min()),
            float(self.shapes.shape_pt_lon.max()),
        ]
        lats = [
            float(self.shapes.shape_pt_lat.min()),
            float(self.shapes.shape_pt_lat.max()),
        ]

        if extra_geoms:
            for df in extra_geoms:
                if isinstance(df, gpd.GeoDataFrame):
                    bounds = df.total_bounds
                    lons.extend([float(bounds[0]), float(bounds[2])])
                    lats.extend([float(bounds[1]), float(bounds[3])])
                elif (
                    isinstance(df, pd.DataFrame)
                    and "geometry_origin" in df.columns
                    and "geometry_destination" in df.columns
                ):
                    # Handle deadhead O-D dataframes
                    for col in ["geometry_origin", "geometry_destination"]:
                        lons.extend(
                            [
                                df[col].apply(lambda p: p.x).min(),
                                df[col].apply(lambda p: p.x).max(),
                            ]
                        )
                        lats.extend(
                            [
                                df[col].apply(lambda p: p.y).min(),
                                df[col].apply(lambda p: p.y).max(),
                            ]
                        )

        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        new_bbox = (
            float(min_lon) - buffer_deg,
            float(min_lat) - buffer_deg,
            float(max_lon) + buffer_deg,
            float(max_lat) + buffer_deg,
        )

        if self._bbox is not None and (
            new_bbox[0] < self._bbox[0]
            or new_bbox[1] < self._bbox[1]
            or new_bbox[2] > self._bbox[2]
            or new_bbox[3] > self._bbox[3]
        ):
            # If the user has not set overwrite to True, we warn them that their results may be outside the map bounds
            if not self.overwrite:
                logger.warning(
                    "Some geometries are outside the current CompassApp bounding box. "
                    "Routing may fail. Set overwrite=True in GTFSEnergyPredictor to reload the map."
                )

        # Check for existing CompassApp in output_dir
        cache_dir = None
        phases = [
            GeneratePipelinePhase.GRAPH,
            GeneratePipelinePhase.CONFIG,
            GeneratePipelinePhase.POWERTRAIN,
        ]

        # Determine which vehicle models to load
        compass_vehicle_models = self.vehicle_models
        if compass_vehicle_models is None:
            compass_vehicle_models = list(VEHICLE_MODELS.keys())

        if self.output_dir is not None:
            cache_dir = self.output_dir / "compass_app"
            config_file = "osm_default_energy.toml"
            config_path = cache_dir / config_file

            if config_path.exists() and not self.overwrite:
                if self.app is None:
                    logger.info(f"Loading existing CompassApp from {cache_dir}")
                    self.app = CompassApp.from_config_file(
                        config_path, parallelism=self.n_processes
                    )
                    self._bbox = new_bbox
                    return

        if not self.overwrite and self.app is not None:
            return

        import osmnx as ox
        import shutil

        if self.overwrite and cache_dir and cache_dir.exists():
            logger.info(f"Clearing CompassApp cache at {cache_dir}")
            shutil.rmtree(cache_dir)

        logger.info(f"Building CompassApp from bounding box: {new_bbox}")

        graph = ox.graph_from_bbox(
            bbox=new_bbox,
            network_type="drive",
        )
        self.app = CompassApp.from_graph(
            graph,
            cache_dir=cache_dir,
            phases=phases,
            vehicle_models=compass_vehicle_models,
            parallelism=self.n_processes,
        )
        self._bbox = new_bbox
        logger.info("CompassApp initialized")

    def filter_trips(
        self,
        date: str | None = None,
        routes: list[str] | None = None,
    ) -> "GTFSEnergyPredictor":
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
            sids = self.feed.get_service_ids_from_date(date)
            self.trips = self.trips[self.trips["service_id"].isin(sids)].copy()

            if len(self.trips) == 0:
                raise ValueError(f"Feed does not contain any bus trips on {date}")

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

        return self

    def _prepare_mid_block_deadhead(self) -> dict[str, Any] | None:
        """
        Prepare mid-block deadhead metadata without routing.

        This method gathers O-D pairs for mid-block deadhead trips but does not
        perform routing. This allows the bounding box to be computed before
        loading CompassApp.

        Returns:
            Dictionary with deadhead trip metadata, or None if no deadhead needed
        """
        if self.feed is None or self.trips.empty or self.shapes.empty:
            raise RuntimeError(
                "Must call load_gtfs_data() before adding deadhead trips"
            )

        logger.info("Preparing mid-block deadhead trips...")

        # Create between-trip deadhead trips
        deadhead_trips = create_mid_block_deadhead_trips(
            self.trips, self.feed.stop_times
        )

        if deadhead_trips.empty:
            logger.info("No mid-block deadhead trips needed")
            return None

        # Create stops and stop_times for deadhead trips
        deadhead_stop_times, deadhead_stops, deadhead_ods = (
            create_mid_block_deadhead_stops(self.feed, deadhead_trips)
        )

        # Remove ODs with same origin and destination (no travel needed)
        deadhead_ods = deadhead_ods[
            deadhead_ods.geometry_origin != deadhead_ods.geometry_destination
        ]

        if deadhead_ods.empty:
            logger.info("No mid-block deadhead O-D pairs after filtering")
            return None

        return {
            "deadhead_trips": deadhead_trips,
            "deadhead_stop_times": deadhead_stop_times,
            "deadhead_stops": deadhead_stops,
            "deadhead_ods": deadhead_ods,
        }

    def _route_mid_block_deadhead(self, metadata: dict[str, Any]) -> None:
        """
        Route mid-block deadhead trips using the loaded CompassApp.

        Args:
            metadata: Dictionary from _prepare_mid_block_deadhead()
        """
        assert self.app is not None, "CompassApp must be loaded before routing"

        deadhead_trips = metadata["deadhead_trips"]
        deadhead_stop_times = metadata["deadhead_stop_times"]
        deadhead_stops = metadata["deadhead_stops"]
        deadhead_ods = metadata["deadhead_ods"]

        logger.info("Routing mid-block deadhead trips...")

        # Generate shapes for unique O-D pairs
        deadhead_shapes, od_mapping = create_deadhead_shapes(
            app=self.app, df=deadhead_ods
        )

        # Assign shape_id to each trip based on O-D mapping
        # The od_mapping has block_id which matches trip_id for mid-block deadhead
        trip_to_shape = od_mapping.set_index("block_id")["shape_id"].to_dict()
        deadhead_trips["shape_id"] = deadhead_trips["trip_id"].map(trip_to_shape)

        # Filter deadhead trips to only those with generated shapes
        deadhead_trips = deadhead_trips[
            deadhead_trips["shape_id"].isin(deadhead_shapes["shape_id"].unique())
        ]

        # Add trip start time, end time, and duration to deadhead trips
        deadhead_trip_times = (
            deadhead_stop_times.groupby("trip_id")
            .agg(start_time=("arrival_time", "min"), end_time=("arrival_time", "max"))
            .reset_index()
        )

        deadhead_trip_times["trip_duration_minutes"] = (
            pd.to_timedelta(
                deadhead_trip_times["end_time"] - deadhead_trip_times["start_time"]
            ).dt.total_seconds()
            / 60
        ).round(2)

        # Convert start/end times to GTFS-style strings
        def format_timedelta(td: pd.Timedelta) -> str:
            if pd.isna(td):
                return ""
            total_seconds = int(td.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours:02}:{minutes:02}:{seconds:02}"

        deadhead_trip_times["start_time"] = deadhead_trip_times["start_time"].apply(
            format_timedelta
        )
        deadhead_trip_times["end_time"] = deadhead_trip_times["end_time"].apply(
            format_timedelta
        )

        deadhead_trips = deadhead_trips.merge(
            deadhead_trip_times[
                ["trip_id", "start_time", "end_time", "trip_duration_minutes"]
            ],
            on="trip_id",
            how="left",
        )

        # Add trip count column to deadhead trips, let it be the same as the service trips
        # before or after the deadhead trip
        deadhead_trips["before_trip"] = deadhead_trips["trip_id"].apply(
            lambda x: x.split("_to_")[0]
        )
        trip_counts = self.trips.set_index("trip_id")["trip_count"].to_dict()
        deadhead_trips["trip_count"] = deadhead_trips["before_trip"].map(trip_counts)
        deadhead_trips = deadhead_trips.drop(columns=["before_trip"])

        # Update internal state
        assert self.feed is not None, "GTFS feed must be loaded"
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

        logger.info(f"Added {len(deadhead_trips)} mid-block deadhead trips")

    def _prepare_depot_deadhead(self) -> dict[str, Any] | None:
        """
        Prepare depot deadhead metadata without routing.

        This method gathers depot endpoints for deadhead trips but does not
        perform routing. This allows the bounding box to be computed before
        loading CompassApp.

        Returns:
            Dictionary with depot deadhead metadata, or None if no deadhead needed
        """
        if self.feed is None or self.trips.empty or self.shapes.empty:
            raise RuntimeError(
                "Must call load_gtfs_data() before adding deadhead trips"
            )

        if self.depot_path is None:
            raise RuntimeError(
                "depot_path must be specified in __init__() to add depot deadhead trips"
            )

        logger.info("Preparing depot deadhead trips...")

        # Create depot deadhead trip records
        deadhead_trips = create_depot_deadhead_trips(self.trips, self.feed.stop_times)

        if deadhead_trips.empty:
            logger.info("No depot deadhead trips needed")
            return None

        # Infer depot locations for each block's first and last stops
        depot_shapefile = self.depot_path / "Transit_Depot.shp"
        first_stops_gdf, last_stops_gdf = infer_depot_trip_endpoints(
            self.trips, self.feed, depot_shapefile
        )

        # Create stop_times and stops for depot deadhead trips
        deadhead_stop_times, deadhead_stops = create_depot_deadhead_stops(
            first_stops_gdf, last_stops_gdf, deadhead_trips
        )

        return {
            "deadhead_trips": deadhead_trips,
            "deadhead_stop_times": deadhead_stop_times,
            "deadhead_stops": deadhead_stops,
            "first_stops_gdf": first_stops_gdf,
            "last_stops_gdf": last_stops_gdf,
        }

    def _route_depot_deadhead(self, metadata: dict[str, Any]) -> None:
        """
        Route depot deadhead trips using the loaded CompassApp.

        Args:
            metadata: Dictionary from _prepare_depot_deadhead()
        """
        assert self.app is not None, "CompassApp must be loaded before routing"

        deadhead_trips = metadata["deadhead_trips"]
        deadhead_stop_times = metadata["deadhead_stop_times"]
        deadhead_stops = metadata["deadhead_stops"]
        first_stops_gdf = metadata["first_stops_gdf"]
        last_stops_gdf = metadata["last_stops_gdf"]

        logger.info("Routing depot deadhead trips...")

        # Generate shapes for trips from depot to first stop
        from_depot_shapes, from_depot_mapping = create_deadhead_shapes(
            app=self.app, df=first_stops_gdf
        )
        from_depot_shapes["shape_id"] = from_depot_shapes["shape_id"].apply(
            lambda x: f"from_depot_{x}"
        )
        from_depot_mapping["shape_id"] = from_depot_mapping["shape_id"].apply(
            lambda x: f"from_depot_{x}"
        )

        # Generate shapes for trips from last stop to depot
        to_depot_shapes, to_depot_mapping = create_deadhead_shapes(
            app=self.app, df=last_stops_gdf
        )
        to_depot_shapes["shape_id"] = to_depot_shapes["shape_id"].apply(
            lambda x: f"to_depot_{x}"
        )
        to_depot_mapping["shape_id"] = to_depot_mapping["shape_id"].apply(
            lambda x: f"to_depot_{x}"
        )

        # Combine all depot deadhead shapes
        deadhead_shapes = pd.concat(
            [from_depot_shapes, to_depot_shapes], ignore_index=True
        )

        # Assign shape_id to each trip based on O-D mapping
        # For pull-out trips: block_id -> from_depot shape
        # For pull-in trips: block_id -> to_depot shape
        from_depot_shape_map = from_depot_mapping.set_index("block_id")[
            "shape_id"
        ].to_dict()
        to_depot_shape_map = to_depot_mapping.set_index("block_id")[
            "shape_id"
        ].to_dict()

        deadhead_trips["shape_id"] = [
            from_depot_shape_map.get(b)
            if t == "pull-out"
            else to_depot_shape_map.get(b)
            for t, b in zip(deadhead_trips["trip_type"], deadhead_trips["block_id"])
        ]

        # Filter deadhead trips to only those with generated shapes
        deadhead_trips = deadhead_trips[
            deadhead_trips["shape_id"].isin(deadhead_shapes["shape_id"].unique())
        ]

        # Add trip start time, end time, and duration to deadhead trips
        deadhead_trip_times = (
            deadhead_stop_times.groupby("trip_id")
            .agg(start_time=("arrival_time", "min"), end_time=("arrival_time", "max"))
            .reset_index()
        )

        deadhead_trip_times["trip_duration_minutes"] = (
            pd.to_timedelta(
                deadhead_trip_times["end_time"] - deadhead_trip_times["start_time"]
            ).dt.total_seconds()
            / 60
        ).round(2)

        # Convert start/end times to GTFS-style strings
        def format_timedelta(td: pd.Timedelta) -> str:
            if pd.isna(td):
                return ""
            total_seconds = int(td.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours:02}:{minutes:02}:{seconds:02}"

        deadhead_trip_times["start_time"] = deadhead_trip_times["start_time"].apply(
            format_timedelta
        )
        deadhead_trip_times["end_time"] = deadhead_trip_times["end_time"].apply(
            format_timedelta
        )

        deadhead_trips = deadhead_trips.merge(
            deadhead_trip_times[
                ["trip_id", "start_time", "end_time", "trip_duration_minutes"]
            ],
            on="trip_id",
            how="left",
        )

        # Add trip count column to deadhead trips, let it be the same as the service trips
        # before or after the deadhead trip
        deadhead_trips["before_or_after_trip"] = deadhead_trips["trip_id"].apply(
            lambda x: (
                x.split("depot_to_")[1] if "depot_to_" in x else x.split("_to_depot")[0]
            )
        )
        trip_counts = self.trips.set_index("trip_id")["trip_count"].to_dict()
        deadhead_trips["trip_count"] = deadhead_trips["before_or_after_trip"].map(
            trip_counts
        )
        deadhead_trips = deadhead_trips.drop(columns=["before_or_after_trip"])

        # Update internal state
        assert self.feed is not None, "GTFS feed must be loaded"
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

    def get_link_level_inputs(self) -> "GTFSEnergyPredictor":
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

        # Step 2: Match to network (no energy model — just geometry + road attributes)
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

    def _match_shapes_to_network(
        self, upsampled_shapes: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Match upsampled shapes to the road network using CompassApp.

        This performs map matching to snap GPS traces to the road
        network and retrieve network attributes for each matched link.

        Args:
            upsampled_shapes: List of upsampled shape DataFrames

        Returns:
            DataFrame with matched shapes including network attributes
        """
        if self.app is None:
            raise RuntimeError(
                "CompassApp must be initialized before map matching. "
                "Call load_compass_app() first."
            )

        # Determine model_name for map matching search parameters
        # (needed when using energy config which requires a model_name
        # for the internal path recalculation during map matching)
        if self.vehicle_models is not None:
            if isinstance(self.vehicle_models, str):
                mm_model_name = self.vehicle_models
            else:
                mm_model_name = list(self.vehicle_models)[0]
        else:
            mm_model_name = list(VEHICLE_MODELS.keys())[0]

        # Build queries for all shapes
        queries = [
            self._create_map_match_query(shape_df, model_name=mm_model_name)
            for shape_df in upsampled_shapes
        ]
        shape_ids = [df["shape_id"].iloc[0] for df in upsampled_shapes]

        logger.info(f"Running map matching for {len(queries)} shapes...")

        # Run map matching with CompassApp (handles parallelism natively)
        results = self.app.map_match(queries)

        # Process results into a combined DataFrame
        return self._process_map_match_results(results, shape_ids)

    @staticmethod
    def _create_map_match_query(
        shape_df: pd.DataFrame,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a CompassApp map matching query from a GTFS shape DataFrame.

        Args:
            shape_df: DataFrame with columns 'shape_pt_lon', 'shape_pt_lat'
            model_name: Optional vehicle model name to use for map matching
                search parameters. Required when using energy config to override
                the default model_name.

        Returns:
            Dictionary suitable for CompassApp.map_match
        """
        trace = [
            {"x": float(row["shape_pt_lon"]), "y": float(row["shape_pt_lat"])}
            for _, row in shape_df.iterrows()
        ]

        query: dict[str, Any] = {
            "trace": trace,
        }
        if model_name is not None:
            query["search_parameters"] = {"model_name": model_name}
        return query

    def _process_map_match_results(
        self, results: list[dict[str, Any]] | dict[str, Any], shape_ids: list[str]
    ) -> pd.DataFrame:
        """
        Process CompassApp map matching results into a DataFrame.

        Args:
            results: Map matching results from CompassApp
            shape_ids: List of shape IDs corresponding to results

        Returns:
            DataFrame with matched shape data including geometry and energy
        """
        # Use match_result_to_geopandas to get link-level data
        gdf = match_result_to_geopandas(results)

        if gdf.empty:
            logger.warning("No map matching results returned")
            return pd.DataFrame()

        # Add shape_id to each result
        if isinstance(results, dict):
            results = [results]

        # Build shape_id mapping from match_id
        shape_id_map = {i: sid for i, sid in enumerate(shape_ids)}
        gdf["shape_id"] = gdf["match_id"].map(shape_id_map)

        # edge_distance is in miles (from compass config with distance_unit = "miles")
        # Keep as-is for powertrain which expects miles
        return cast(pd.DataFrame, gdf)

    def predict_energy(
        self,
        add_hvac: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Predict energy consumption by map matching once, then running
        CompassApp.run_calculate_path for each vehicle model.

        This method:
        1. Runs map matching ONCE to get road-level attributes (distance, speed, grade)
        2. Extracts edge_ids from the map-matched paths
        3. Runs CompassApp.run_calculate_path for each vehicle model with model_name

        This is much more efficient than the previous approach of running
        map matching per vehicle model, since map matching is computationally
        expensive and the road attributes are the same regardless of vehicle type.

        Energy modeling is handled entirely by RouteE-Compass's powertrain
        traversal models, eliminating the need for the nrel.routee.powertrain package.

        Args:
            add_hvac: Whether to add HVAC energy consumption to trip-level results

        Returns:
            Dictionary with keys:
                - 'link': DataFrame with link-level predictions for all models
                - 'trip': DataFrame with trip-level predictions for all models
                - '<model_name>_link': Link-level predictions for specific model
                - '<model_name>_trip': Trip-level predictions for specific model

        Raises:
            RuntimeError: If GTFS data hasn't been loaded yet
            ValueError: If vehicle model is not supported
        """

        if self.feed is None or self.trips.empty or self.shapes.empty:
            raise RuntimeError("Must call load_gtfs_data() before predicting energy")

        if self.vehicle_models is None:
            vehicle_models_list = list(VEHICLE_MODELS.keys())
        elif isinstance(self.vehicle_models, str):
            vehicle_models_list = [self.vehicle_models]
        else:
            vehicle_models_list = list(self.vehicle_models)

        # Validate vehicle models
        for model in vehicle_models_list:
            if model not in VEHICLE_MODELS:
                raise ValueError(
                    f"Unsupported vehicle model: {model}. "
                    f"Supported models: {list(VEHICLE_MODELS.keys())}"
                )

        logger.info(
            f"Predicting energy for {len(vehicle_models_list)} vehicle model(s)..."
        )

        if self.app is None:
            self.load_compass_app()

        # Run map matching ONCE to get edge-level results
        shape_groups = [group for _, group in self.shapes.groupby("shape_id")]
        with mp.Pool(self.n_processes) as pool:
            upsampled_shapes = pool.map(upsample_shape, shape_groups)

        logger.debug(f"Upsampled {len(shape_groups)} shapes")

        link_results = self._match_shapes_to_network(upsampled_shapes)
        if link_results.empty:
            logger.warning("No map matching results — cannot predict energy")
            return self.energy_predictions

        self.matched_shapes = link_results

        # Extract edge_ids per shape from map-matched results
        shapes_edge_ids: dict[str, list[dict[str, int]]] = {}
        for shape_id, group in link_results.groupby("shape_id"):
            edges: list[dict[str, int]] = []
            for _, row in group.iterrows():
                edge_entry: dict[str, int] = {"edge_id": int(row["edge_id"])}
                if "edge_list_id" in row and pd.notna(row["edge_list_id"]):
                    edge_entry["edge_list_id"] = int(row["edge_list_id"])
                edges.append(edge_entry)
            shapes_edge_ids[str(shape_id)] = edges

        # Run run_calculate_path for each vehicle model
        all_link_results: list[pd.DataFrame] = []
        all_trip_results: list[pd.DataFrame] = []

        for model_name in vehicle_models_list:
            logger.info(f"Running energy prediction via CompassApp for: {model_name}")

            model_config = VEHICLE_MODELS[model_name]
            energy_field = model_config["energy_field"]

            # Build queries with model_name parameter
            shape_id_list = list(shapes_edge_ids.keys())
            queries: list[dict[str, Any]] = [
                {
                    "path": shapes_edge_ids[sid],
                    "model_name": model_name,
                    "weights": {"trip_time": 1.0},
                }
                for sid in shape_id_list
            ]

            # Run calculate path via CompassApp
            assert self.app is not None, "CompassApp must be loaded"
            results = self.app.run_calculate_path(queries)
            if isinstance(results, dict):
                results = [results]

            # Process results: extract energy from traversal_summary
            energy_records: list[dict[str, Any]] = []
            for sid, result in zip(shape_id_list, results):
                if "error" in result:
                    logger.warning(
                        f"run_calculate_path error for shape {sid}, "
                        f"model {model_name}: {result['error']}"
                    )
                    continue

                route = result.get("route", {})
                summary = route.get("traversal_summary", {})

                # Extract energy value from traversal summary
                energy_entry = summary.get(energy_field, {})
                energy_value = energy_entry.get("value", 0.0)

                # Extract distance
                distance_entry = summary.get("edge_distance", {})
                distance_value = distance_entry.get("value", 0.0)

                energy_records.append(
                    {
                        "shape_id": sid,
                        "energy_used": float(energy_value),
                        "miles": float(distance_value),
                        "vehicle": model_name,
                        "energy_unit": model_config["unit"],
                    }
                )

            if not energy_records:
                logger.warning(f"No energy results for model {model_name}")
                continue

            energy_by_shape = pd.DataFrame(energy_records)

            # Build link-level results (map-match data + vehicle label)
            model_link_results = link_results.copy()
            model_link_results["vehicle"] = model_name
            # Merge shape-level energy onto link results for per-link context
            model_link_results = model_link_results.merge(
                energy_by_shape[["shape_id", "energy_used", "energy_unit"]],
                on="shape_id",
                how="left",
            )

            # Aggregate to trip level
            trip_results = self._aggregate_predictions_by_trip(
                energy_by_shape, model_name
            )

            # Optionally add HVAC to trip-level results
            if add_hvac:
                logger.info("Adding HVAC energy impacts...")
                hvac_energy = add_HVAC_energy(self.feed, self.trips)
                trip_results = trip_results.merge(hvac_energy, on="trip_id", how="left")
                # Add HVAC energy to powertrain energy for electric vehicles
                kwh_mask = trip_results["energy_unit"] == "kWh"
                trip_results.loc[kwh_mask, "energy_used"] += trip_results.loc[
                    kwh_mask, "hvac_energy_kWh"
                ]
            else:
                trip_results = trip_results.merge(self.trips, on="trip_id")

            # Store results
            self.energy_predictions[f"{model_name}_link"] = model_link_results
            self.energy_predictions[f"{model_name}_trip"] = trip_results
            all_link_results.append(model_link_results)
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

    def _aggregate_predictions_by_trip(
        self, energy_by_shape: pd.DataFrame, vehicle_name: str
    ) -> pd.DataFrame:
        """
        Aggregate shape-level energy predictions to trip level.

        Args:
            energy_by_shape: DataFrame with shape-level energy predictions
                from run_calculate_path (columns: shape_id, energy_used,
                miles, vehicle, energy_unit)
            vehicle_name: Name of vehicle model

        Returns:
            DataFrame with trip-level aggregated results
        """
        # Map shapes to trips
        shape_to_trips = self.trips[["trip_id", "shape_id"]].drop_duplicates()
        energy_by_trip = energy_by_shape.merge(shape_to_trips, on="shape_id")

        # Clean up columns
        cols_to_drop = [c for c in ["shape_id"] if c in energy_by_trip.columns]
        return energy_by_trip.drop(columns=cols_to_drop)

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
        output_dir: str | Path | None = None,
        save_geometry: bool = True,
        save_inputs: bool = True,
    ) -> None:
        """
        Save prediction results to CSV files.

        Args:
            output_dir: Directory to save results. If None, uses self.output_dir,
                defaulting to the current working directory if that is also None.
            save_geometry: Whether to save link geometry separately
            save_inputs: Whether to save RouteE input features

        Raises:
            RuntimeError: If no predictions have been generated yet
        """
        if not self.energy_predictions:
            raise RuntimeError("No predictions to save. Call predict_energy() first.")

        if output_dir:
            output_path = Path(output_dir)
        elif self.output_dir:
            output_path = self.output_dir
        else:
            output_path = Path.cwd()

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
