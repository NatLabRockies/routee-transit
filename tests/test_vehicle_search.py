"""Every bundled vehicle model must support path search, not just path evaluation.

Map matching and deadhead routing run an A* search with the first requested
vehicle model, which exercises each energy model's ``estimate_traversal``
heuristic. ``run_calculate_path`` never calls it, so a model can evaluate
paths correctly while failing every search.
"""

import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import tomlkit

from routee.transit.compass_app import TransitCompassApp
from routee.transit.speed_model import SPEED_TRACE_MODEL_CONFIG

VEHICLE_MODEL_DIR = (
    Path(__file__).parents[1] / "routee" / "transit" / "resources" / "vehicle_models"
)
BUNDLED_MODELS = sorted(path.stem for path in VEHICLE_MODEL_DIR.glob("*.json"))


@pytest.fixture(scope="module")
def vehicle_app(tmp_path_factory: pytest.TempPathFactory) -> TransitCompassApp:
    """Two one-mile edges served by every bundled vehicle model."""
    tmp_path = tmp_path_factory.mktemp("vehicle_app")
    vehicles = tmp_path / "vehicles"
    vehicles.mkdir()
    for name in BUNDLED_MODELS:
        shutil.copy(VEHICLE_MODEL_DIR / f"{name}.json", vehicles)
        shutil.copy(VEHICLE_MODEL_DIR / f"{name}.bin", vehicles)

    (tmp_path / "vertices.csv").write_text(
        "vertex_id,x,y\n0,-105,40\n1,-104.99,40\n2,-104.98,40\n"
    )
    (tmp_path / "edges.csv").write_text(
        "edge_id,src_vertex_id,dst_vertex_id,distance\n0,0,1,1609.344\n1,1,2,1609.344\n"
    )
    (tmp_path / "geometry.txt").write_text(
        "LINESTRING (-105 40, -104.99 40)\nLINESTRING (-104.99 40, -104.98 40)\n"
    )
    (tmp_path / "grade.txt").write_text("0.0\n0.0\n")
    (tmp_path / "gtfs_stops.csv").write_text("trip_id,edge_id\n")
    np.array([0, 0], dtype="<u4").tofile(tmp_path / "transit_speed_edge_traces.bin")
    np.array([25.0] * 12, dtype="<f4").tofile(tmp_path / "transit_speed_traces.bin")

    config = {
        "graph": {
            "vertex_list_input_file": "vertices.csv",
            "edge_list": {"input_file": "edges.csv"},
        },
        "mapping": {
            "type": "edge",
            "spatial_index_type": "edge",
            "matching_type": ["vertex_id", "edge_id", "point"],
            "geometry": {
                "type": "from_linestrings",
                "geometry_input_file": "geometry.txt",
            },
        },
        "search": {
            "traversal": {
                "type": "combined",
                "models": [
                    {"type": "distance", "distance_unit": "miles"},
                    SPEED_TRACE_MODEL_CONFIG.copy(),
                    {"type": "time", "time_unit": "minutes"},
                    {
                        "type": "grade",
                        "grade_unit": "decimal",
                        "grade_input_file": "grade.txt",
                    },
                    {
                        "type": "transit_energy",
                        "vehicle_input_files": [
                            f"vehicles/{name}.json" for name in BUNDLED_MODELS
                        ],
                        "stop_edge_mapping_input_file": "gtfs_stops.csv",
                    },
                ],
            }
        },
        "cost": {
            "vehicle_rates": {
                "trip_time": {"type": "time", "factor": 1.0, "unit": "minutes"},
            }
        },
        "map_matching": {
            "type": "lcss",
            "distance_epsilon": 50.0,
            "similarity_cutoff": 0.9,
            "cutting_threshold": 10.0,
            "random_cuts": 0,
            "distance_threshold": 10000.0,
            "distance_unit": "meters",
        },
        "plugin": {
            "output_plugins": [
                {
                    "type": "traversal",
                    "route": "geo_json",
                    "geometry_input_file": "geometry.txt",
                },
            ]
        },
    }
    path = tmp_path / "config.toml"
    path.write_text(tomlkit.dumps(config))
    return cast(
        TransitCompassApp, TransitCompassApp.from_config_file(path, parallelism=1)
    )


def test_all_bundled_models_are_covered() -> None:
    assert len(BUNDLED_MODELS) == 8


@pytest.mark.parametrize("model_name", BUNDLED_MODELS)
def test_bundled_model_supports_routing(
    vehicle_app: TransitCompassApp, model_name: str
) -> None:
    result = vehicle_app.run(
        {
            "origin_vertex": 0,
            "destination_vertex": 2,
            "model_name": model_name,
            "start_time": "12:00:00",
            "start_weekday": "monday",
            "weights": {"trip_time": 1.0},
        }
    )
    assert isinstance(result, dict)
    assert "error" not in result, result["error"]


@pytest.mark.parametrize("model_name", BUNDLED_MODELS)
def test_bundled_model_supports_map_matching(
    vehicle_app: TransitCompassApp, model_name: str
) -> None:
    result = vehicle_app.map_match(
        {
            "trace": [{"x": -105.0 + i * 0.001, "y": 40.0} for i in range(21)],
            "search_parameters": {
                "model_name": model_name,
                "start_time": "12:00:00",
                "start_weekday": "monday",
                "weights": {"trip_time": 1.0},
            },
        }
    )
    assert isinstance(result, dict)
    assert "error" not in result, result["error"]
    assert len(result["matched_path"]["features"]) == 2
