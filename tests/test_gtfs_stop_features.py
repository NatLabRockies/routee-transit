"""Tests for GTFS stop feature functions in aggregate_speeds.py.

Uses a synthetic 4-edge straight route along lat=40.0 so all projections and
distance-weighted speed calculations can be verified analytically.

Route layout (edges west→east, each assigned link_length_m=1000):

  Edge 0        Edge 1        Edge 2        Edge 3
  [-105.00      [-105.01      [-105.02      [-105.03
        -105.01]      -105.02]      -105.03]      -105.04]
  cumul  0–1000   1000–2000   2000–3000   3000–4000   (m)

At lat=40 the actual geodesic width of 0.01° ≈ 854 m, but link_length_m is
set to an exact 1000 m so that cumul_dist_m arithmetic is clean.
"""

import math
import unittest

import geopandas as gpd
import pandas as pd
from aggregate_speeds import (
    aggregate_gtfs_features_by_edge,
    compute_scheduled_speeds_between_stops,
    project_stops_to_route,
)
from shapely.geometry import LineString

# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------

LAT = 40.0
LINK_LENGTH_M = 1000.0
# Longitude boundaries for each of the 4 edges
_LON_BOUNDS = [
    (-105.00, -105.01),
    (-105.01, -105.02),
    (-105.02, -105.03),
    (-105.03, -105.04),
]


def _make_edges_df() -> gpd.GeoDataFrame:
    """4-edge straight route at lat=40.0, each edge 1000 m."""
    geometries = [
        LineString([(lon_s, LAT), (lon_e, LAT)]) for lon_s, lon_e in _LON_BOUNDS
    ]
    return gpd.GeoDataFrame(
        {
            "edge_id": list(range(4)),
            "link_length_m": [LINK_LENGTH_M] * 4,
            "cumul_start_m": [i * LINK_LENGTH_M for i in range(4)],
        },
        geometry=geometries,
        crs="EPSG:4326",
    )


def _stops_df(stops: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Build stops.txt indexed by stop_id.  stops maps stop_id → (lat, lon)."""
    return pd.DataFrame(
        [
            {"stop_id": sid, "stop_lat": lat, "stop_lon": lon}
            for sid, (lat, lon) in stops.items()
        ]
    ).set_index("stop_id")


def _stop_times(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _stops_on_route(rows: list[dict]) -> pd.DataFrame:
    """Convenience constructor for pre-built stops_on_route DataFrames."""
    cols = [
        "stop_id",
        "stop_sequence",
        "edge_idx",
        "road_id",
        "cumul_dist_m",
        "departure_sec",
        "arrival_sec",
    ]
    return pd.DataFrame(rows, columns=cols)


def _sched_speeds(rows: list[dict]) -> pd.DataFrame:
    """Convenience constructor for pre-built sched_speeds DataFrames."""
    cols = [
        "stop_seq_from",
        "stop_seq_to",
        "edge_idx_from",
        "edge_idx_to",
        "cumul_dist_from",
        "cumul_dist_to",
        "dist_m",
        "time_sec",
        "scheduled_speed_mph",
    ]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# project_stops_to_route
# ---------------------------------------------------------------------------


class TestProjectStopsToRoute(unittest.TestCase):
    def test_stop_at_midpoint_of_edge(self) -> None:
        """Stop exactly at midpoint of edge 1 → edge_idx=1, cumul_dist_m=1500."""
        edges_df = _make_edges_df()
        stops_df = _stops_df({"S1": (LAT, -105.015)})
        stop_times = _stop_times(
            [
                {
                    "stop_id": "S1",
                    "stop_sequence": 1,
                    "arrival_time": "00:00:00",
                    "departure_time": "00:00:00",
                },
            ]
        )

        result = project_stops_to_route(stop_times, stops_df, edges_df)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["edge_idx"], 1)
        self.assertAlmostEqual(result.iloc[0]["cumul_dist_m"], 1500.0)

    def test_stop_at_end_of_edge(self) -> None:
        """Stop at the far endpoint of edge 1 → edge_idx=1, cumul_dist_m=2000."""
        edges_df = _make_edges_df()
        stops_df = _stops_df({"S1": (LAT, -105.02)})
        stop_times = _stop_times(
            [
                {
                    "stop_id": "S1",
                    "stop_sequence": 1,
                    "arrival_time": "00:00:30",
                    "departure_time": "00:00:30",
                },
            ]
        )

        result = project_stops_to_route(stop_times, stops_df, edges_df)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["edge_idx"], 1)
        self.assertAlmostEqual(result.iloc[0]["cumul_dist_m"], 2000.0)

    def test_two_stops_in_interior_of_same_edge(self) -> None:
        """Two stops both in the interior of edge 1 → both get edge_idx=1."""
        edges_df = _make_edges_df()
        # lon=-105.012 is frac=0.2 on edge 1; lon=-105.018 is frac=0.8
        stops_df = _stops_df({"S1": (LAT, -105.012), "S2": (LAT, -105.018)})
        stop_times = _stop_times(
            [
                {
                    "stop_id": "S1",
                    "stop_sequence": 1,
                    "arrival_time": "00:00:00",
                    "departure_time": "00:00:05",
                },
                {
                    "stop_id": "S2",
                    "stop_sequence": 2,
                    "arrival_time": "00:01:00",
                    "departure_time": "00:01:05",
                },
            ]
        )

        result = project_stops_to_route(stop_times, stops_df, edges_df)

        self.assertEqual(len(result), 2)
        self.assertTrue((result["edge_idx"] == 1).all())
        self.assertAlmostEqual(result.iloc[0]["cumul_dist_m"], 1200.0)
        self.assertAlmostEqual(result.iloc[1]["cumul_dist_m"], 1800.0)

    def test_two_stops_at_endpoints_of_edge(self) -> None:
        """Stops exactly at the two endpoints of edge 1.

        The greedy-forward algorithm assigns each endpoint to its *earlier*
        edge, so the start-of-edge-1 stop goes to edge 0 (frac=1.0) and the
        end-of-edge-1 stop goes to edge 1 (frac=1.0).
        """
        edges_df = _make_edges_df()
        # -105.01 is the shared boundary of edge 0 / edge 1
        # -105.02 is the shared boundary of edge 1 / edge 2
        stops_df = _stops_df({"S1": (LAT, -105.01), "S2": (LAT, -105.02)})
        stop_times = _stop_times(
            [
                {
                    "stop_id": "S1",
                    "stop_sequence": 1,
                    "arrival_time": "00:00:00",
                    "departure_time": "00:00:30",
                },
                {
                    "stop_id": "S2",
                    "stop_sequence": 2,
                    "arrival_time": "00:02:00",
                    "departure_time": "00:02:30",
                },
            ]
        )

        result = project_stops_to_route(stop_times, stops_df, edges_df)

        self.assertEqual(len(result), 2)
        # S1 at the start of edge 1 snaps to the end of edge 0 (greedy-forward picks the first zero-distance edge)
        self.assertEqual(result.iloc[0]["edge_idx"], 0)
        self.assertAlmostEqual(result.iloc[0]["cumul_dist_m"], 1000.0)
        # S2 at the end of edge 1 snaps to edge 1 (edge 0 is behind us now)
        self.assertEqual(result.iloc[1]["edge_idx"], 1)
        self.assertAlmostEqual(result.iloc[1]["cumul_dist_m"], 2000.0)

    def test_no_stops_on_middle_edge_reflected_in_n_stops(self) -> None:
        """Stops on edges 0, 2, 3 only — n_stops[1] must be 0 after aggregation."""
        edges_df = _make_edges_df()
        stops_df = _stops_df(
            {
                "S1": (LAT, -105.005),  # midpoint edge 0
                "S2": (LAT, -105.025),  # midpoint edge 2
                "S3": (LAT, -105.035),  # midpoint edge 3
            }
        )
        stop_times = _stop_times(
            [
                {
                    "stop_id": "S1",
                    "stop_sequence": 1,
                    "arrival_time": "00:00:00",
                    "departure_time": "00:00:30",
                },
                {
                    "stop_id": "S2",
                    "stop_sequence": 2,
                    "arrival_time": "00:03:00",
                    "departure_time": "00:03:30",
                },
                {
                    "stop_id": "S3",
                    "stop_sequence": 3,
                    "arrival_time": "00:06:00",
                    "departure_time": "00:06:30",
                },
            ]
        )

        stops_on_route = project_stops_to_route(stop_times, stops_df, edges_df)
        sched_speeds = compute_scheduled_speeds_between_stops(stops_on_route)
        feats = aggregate_gtfs_features_by_edge(stops_on_route, sched_speeds, edges_df)

        self.assertEqual(feats.loc[0, "n_stops"], 1)
        self.assertEqual(feats.loc[1, "n_stops"], 0)
        self.assertEqual(feats.loc[2, "n_stops"], 1)
        self.assertEqual(feats.loc[3, "n_stops"], 1)

    def test_stop_slightly_off_shape_projects_correctly(self) -> None:
        """Stop 0.002° north of the route (~220 m) still snaps to the correct edge."""
        edges_df = _make_edges_df()
        # lon=-105.015 is the midpoint of edge 1 in the longitude dimension;
        # lat=40.002 places the stop ~220 m north of the horizontal route.
        stops_df = _stops_df({"S1": (LAT + 0.002, -105.015)})
        stop_times = _stop_times(
            [
                {
                    "stop_id": "S1",
                    "stop_sequence": 1,
                    "arrival_time": "00:00:00",
                    "departure_time": "00:00:00",
                },
            ]
        )

        result = project_stops_to_route(stop_times, stops_df, edges_df)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["edge_idx"], 1)
        # Projection onto the horizontal edge is purely longitudinal → frac=0.5
        self.assertAlmostEqual(result.iloc[0]["cumul_dist_m"], 1500.0)

    def test_stop_far_from_shape_is_dropped(self) -> None:
        """Stop >1° away is excluded with the default max_snap_dist_deg=0.05."""
        edges_df = _make_edges_df()
        stops_df = _stops_df(
            {
                "S1": (LAT, -105.005),  # valid — midpoint of edge 0
                "S_FAR": (LAT + 1.0, -105.015),  # ~111 km north — should be dropped
            }
        )
        stop_times = _stop_times(
            [
                {
                    "stop_id": "S1",
                    "stop_sequence": 1,
                    "arrival_time": "00:00:00",
                    "departure_time": "00:00:30",
                },
                {
                    "stop_id": "S_FAR",
                    "stop_sequence": 2,
                    "arrival_time": "00:05:00",
                    "departure_time": "00:05:30",
                },
            ]
        )

        result = project_stops_to_route(stop_times, stops_df, edges_df)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["stop_id"], "S1")

    def test_stop_near_threshold_included(self) -> None:
        """Stop just inside max_snap_dist_deg is kept."""
        edges_df = _make_edges_df()
        # 0.04° north < default threshold 0.05°
        stops_df = _stops_df({"S1": (LAT + 0.04, -105.015)})
        stop_times = _stop_times(
            [
                {
                    "stop_id": "S1",
                    "stop_sequence": 1,
                    "arrival_time": "00:00:00",
                    "departure_time": "00:00:00",
                },
            ]
        )

        result = project_stops_to_route(stop_times, stops_df, edges_df)
        self.assertEqual(len(result), 1)

    def test_stop_beyond_threshold_excluded(self) -> None:
        """Stop just outside a custom max_snap_dist_deg is dropped."""
        edges_df = _make_edges_df()
        stops_df = _stops_df({"S1": (LAT + 0.015, -105.015)})
        stop_times = _stop_times(
            [
                {
                    "stop_id": "S1",
                    "stop_sequence": 1,
                    "arrival_time": "00:00:00",
                    "departure_time": "00:00:00",
                },
            ]
        )

        # Tight threshold: 0.01° — stop is 0.015° away so should be dropped
        result = project_stops_to_route(
            stop_times, stops_df, edges_df, max_snap_dist_deg=0.01
        )
        self.assertEqual(len(result), 0)


# ---------------------------------------------------------------------------
# compute_scheduled_speeds_between_stops
# ---------------------------------------------------------------------------


class TestComputeScheduledSpeeds(unittest.TestCase):
    def test_arrival_based_timing_excludes_dwell(self) -> None:
        """time_sec = B.arrival_sec - A.departure_sec, not departure-to-departure.

        Stop A departs at t=0.
        Stop B arrives at t=90 s, departs at t=120 s  (30 s dwell).
        Stop C arrives at t=200 s.

        A→B travel time should be 90 s (not 120 s which includes B's dwell).
        B→C travel time should be 80 s (200 - 120).
        """
        stops = _stops_on_route(
            [
                {
                    "stop_id": "A",
                    "stop_sequence": 1,
                    "edge_idx": 0,
                    "road_id": "0",
                    "cumul_dist_m": 0.0,
                    "departure_sec": 0.0,
                    "arrival_sec": 0.0,
                },
                {
                    "stop_id": "B",
                    "stop_sequence": 2,
                    "edge_idx": 1,
                    "road_id": "1",
                    "cumul_dist_m": 1500.0,
                    "departure_sec": 120.0,
                    "arrival_sec": 90.0,
                },
                {
                    "stop_id": "C",
                    "stop_sequence": 3,
                    "edge_idx": 3,
                    "road_id": "3",
                    "cumul_dist_m": 3500.0,
                    "departure_sec": 220.0,
                    "arrival_sec": 200.0,
                },
            ]
        )

        result = compute_scheduled_speeds_between_stops(stops)

        self.assertEqual(len(result), 2)
        # A→B
        self.assertAlmostEqual(result.iloc[0]["time_sec"], 90.0)
        expected_ab = (1500.0 / 1609.344) / (90.0 / 3600.0)
        self.assertAlmostEqual(
            result.iloc[0]["scheduled_speed_mph"], expected_ab, places=4
        )
        # B→C
        self.assertAlmostEqual(result.iloc[1]["time_sec"], 80.0)
        expected_bc = (2000.0 / 1609.344) / (80.0 / 3600.0)
        self.assertAlmostEqual(
            result.iloc[1]["scheduled_speed_mph"], expected_bc, places=4
        )

    def test_cumul_dist_columns_present(self) -> None:
        """Output includes cumul_dist_from and cumul_dist_to (needed for distance weighting)."""
        stops = _stops_on_route(
            [
                {
                    "stop_id": "A",
                    "stop_sequence": 1,
                    "edge_idx": 0,
                    "road_id": "0",
                    "cumul_dist_m": 200.0,
                    "departure_sec": 0.0,
                    "arrival_sec": 0.0,
                },
                {
                    "stop_id": "B",
                    "stop_sequence": 2,
                    "edge_idx": 2,
                    "road_id": "2",
                    "cumul_dist_m": 2800.0,
                    "departure_sec": 140.0,
                    "arrival_sec": 130.0,
                },
            ]
        )

        result = compute_scheduled_speeds_between_stops(stops)

        self.assertIn("cumul_dist_from", result.columns)
        self.assertIn("cumul_dist_to", result.columns)
        self.assertAlmostEqual(result.iloc[0]["cumul_dist_from"], 200.0)
        self.assertAlmostEqual(result.iloc[0]["cumul_dist_to"], 2800.0)

    def test_nan_speed_when_zero_travel_time(self) -> None:
        """Segment where B.arrival == A.departure produces NaN scheduled speed."""
        stops = _stops_on_route(
            [
                {
                    "stop_id": "A",
                    "stop_sequence": 1,
                    "edge_idx": 0,
                    "road_id": "0",
                    "cumul_dist_m": 0.0,
                    "departure_sec": 60.0,
                    "arrival_sec": 60.0,
                },
                {
                    "stop_id": "B",
                    "stop_sequence": 2,
                    "edge_idx": 1,
                    "road_id": "1",
                    "cumul_dist_m": 1000.0,
                    "departure_sec": 120.0,
                    "arrival_sec": 60.0,
                },
            ]
        )

        result = compute_scheduled_speeds_between_stops(stops)

        self.assertTrue(math.isnan(result.iloc[0]["scheduled_speed_mph"]))

    def test_fewer_than_two_stops_returns_empty(self) -> None:
        stops = _stops_on_route(
            [
                {
                    "stop_id": "A",
                    "stop_sequence": 1,
                    "edge_idx": 0,
                    "road_id": "0",
                    "cumul_dist_m": 0.0,
                    "departure_sec": 0.0,
                    "arrival_sec": 0.0,
                },
            ]
        )
        result = compute_scheduled_speeds_between_stops(stops)
        self.assertEqual(len(result), 0)


# ---------------------------------------------------------------------------
# aggregate_gtfs_features_by_edge
# ---------------------------------------------------------------------------


class TestAggregateGtfsFeaturesByEdge(unittest.TestCase):
    def test_n_stops_correct(self) -> None:
        """n_stops counts stops assigned to each edge, including zero for uncovered edges."""
        edges_df = _make_edges_df()
        # Stops on edges 0, 1, 3 — none on edge 2
        stops_on_route = _stops_on_route(
            [
                {
                    "stop_id": "S1",
                    "stop_sequence": 1,
                    "edge_idx": 0,
                    "road_id": "0",
                    "cumul_dist_m": 500.0,
                    "departure_sec": 30.0,
                    "arrival_sec": 0.0,
                },
                {
                    "stop_id": "S2",
                    "stop_sequence": 2,
                    "edge_idx": 1,
                    "road_id": "1",
                    "cumul_dist_m": 1500.0,
                    "departure_sec": 120.0,
                    "arrival_sec": 90.0,
                },
                {
                    "stop_id": "S3",
                    "stop_sequence": 3,
                    "edge_idx": 3,
                    "road_id": "3",
                    "cumul_dist_m": 3500.0,
                    "departure_sec": 360.0,
                    "arrival_sec": 330.0,
                },
            ]
        )
        sched_speeds = _sched_speeds([])

        result = aggregate_gtfs_features_by_edge(stops_on_route, sched_speeds, edges_df)

        self.assertEqual(result.loc[0, "n_stops"], 1)
        self.assertEqual(result.loc[1, "n_stops"], 1)
        self.assertEqual(result.loc[2, "n_stops"], 0)
        self.assertEqual(result.loc[3, "n_stops"], 1)

    def test_distance_weighted_speed_single_segment(self) -> None:
        """Single segment covering two edges assigns its speed to both."""
        edges_df = _make_edges_df()
        # Segment from 0 to 2000 m spans edge 0 (0–1000) and edge 1 (1000–2000) equally.
        sched_speeds = _sched_speeds(
            [
                {
                    "stop_seq_from": 1,
                    "stop_seq_to": 2,
                    "edge_idx_from": 0,
                    "edge_idx_to": 1,
                    "cumul_dist_from": 0.0,
                    "cumul_dist_to": 2000.0,
                    "dist_m": 2000.0,
                    "time_sec": 120.0,
                    "scheduled_speed_mph": 30.0,
                }
            ]
        )
        stops_on_route = _stops_on_route([])

        result = aggregate_gtfs_features_by_edge(stops_on_route, sched_speeds, edges_df)

        self.assertAlmostEqual(result.loc[0, "scheduled_speed_mph"], 30.0)
        self.assertAlmostEqual(result.loc[1, "scheduled_speed_mph"], 30.0)
        self.assertTrue(math.isnan(result.loc[2, "scheduled_speed_mph"]))
        self.assertTrue(math.isnan(result.loc[3, "scheduled_speed_mph"]))

    def test_distance_weighted_speed_differs_from_simple_mean(self) -> None:
        """Distance weighting gives a different result than a simple mean.

        Two segments both cover edge 1 (cumul 1000–2000):
          Segment A: cumul 800–1800 at 10 mph  → 800 m overlap on edge 1
          Segment B: cumul 1800–2200 at 40 mph → 200 m overlap on edge 1

        Distance-weighted mean = (10×800 + 40×200) / 1000 = 16 mph
        Simple mean            = (10 + 40) / 2              = 25 mph
        """
        edges_df = _make_edges_df()
        sched_speeds = _sched_speeds(
            [
                {
                    "stop_seq_from": 1,
                    "stop_seq_to": 2,
                    "edge_idx_from": 0,
                    "edge_idx_to": 1,
                    "cumul_dist_from": 800.0,
                    "cumul_dist_to": 1800.0,
                    "dist_m": 1000.0,
                    "time_sec": 100.0,
                    "scheduled_speed_mph": 10.0,
                },
                {
                    "stop_seq_from": 2,
                    "stop_seq_to": 3,
                    "edge_idx_from": 1,
                    "edge_idx_to": 2,
                    "cumul_dist_from": 1800.0,
                    "cumul_dist_to": 2200.0,
                    "dist_m": 400.0,
                    "time_sec": 40.0,
                    "scheduled_speed_mph": 40.0,
                },
            ]
        )
        stops_on_route = _stops_on_route([])

        result = aggregate_gtfs_features_by_edge(stops_on_route, sched_speeds, edges_df)

        # Edge 0: only segment A covers it (200 m overlap: min(1800,1000)-max(800,0))
        self.assertAlmostEqual(result.loc[0, "scheduled_speed_mph"], 10.0)
        # Edge 1: distance-weighted, NOT simple mean
        self.assertAlmostEqual(result.loc[1, "scheduled_speed_mph"], 16.0)
        # Edge 2: only segment B covers it (200 m overlap)
        self.assertAlmostEqual(result.loc[2, "scheduled_speed_mph"], 40.0)
        # Edge 3: no coverage
        self.assertTrue(math.isnan(result.loc[3, "scheduled_speed_mph"]))

    def test_edge_with_no_covering_segment_is_nan(self) -> None:
        """Edges not touched by any stop-pair segment have NaN scheduled_speed_mph."""
        edges_df = _make_edges_df()
        # Segment only covers edges 0 and 1
        sched_speeds = _sched_speeds(
            [
                {
                    "stop_seq_from": 1,
                    "stop_seq_to": 2,
                    "edge_idx_from": 0,
                    "edge_idx_to": 1,
                    "cumul_dist_from": 0.0,
                    "cumul_dist_to": 2000.0,
                    "dist_m": 2000.0,
                    "time_sec": 120.0,
                    "scheduled_speed_mph": 30.0,
                }
            ]
        )
        stops_on_route = _stops_on_route([])

        result = aggregate_gtfs_features_by_edge(stops_on_route, sched_speeds, edges_df)

        self.assertFalse(math.isnan(result.loc[0, "scheduled_speed_mph"]))
        self.assertFalse(math.isnan(result.loc[1, "scheduled_speed_mph"]))
        self.assertTrue(math.isnan(result.loc[2, "scheduled_speed_mph"]))
        self.assertTrue(math.isnan(result.loc[3, "scheduled_speed_mph"]))

    def test_two_stops_same_edge_n_stops_is_two(self) -> None:
        """Two stops projected onto the same edge increment n_stops by 2."""
        edges_df = _make_edges_df()
        stops_on_route = _stops_on_route(
            [
                {
                    "stop_id": "S1",
                    "stop_sequence": 1,
                    "edge_idx": 1,
                    "road_id": "1",
                    "cumul_dist_m": 1200.0,
                    "departure_sec": 5.0,
                    "arrival_sec": 0.0,
                },
                {
                    "stop_id": "S2",
                    "stop_sequence": 2,
                    "edge_idx": 1,
                    "road_id": "1",
                    "cumul_dist_m": 1800.0,
                    "departure_sec": 65.0,
                    "arrival_sec": 60.0,
                },
            ]
        )
        sched_speeds = _sched_speeds([])

        result = aggregate_gtfs_features_by_edge(stops_on_route, sched_speeds, edges_df)

        self.assertEqual(result.loc[1, "n_stops"], 2)
        self.assertEqual(result.loc[0, "n_stops"], 0)
        self.assertEqual(result.loc[2, "n_stops"], 0)


if __name__ == "__main__":
    unittest.main()
