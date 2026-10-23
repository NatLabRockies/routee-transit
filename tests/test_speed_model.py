import json
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest
import tomlkit

from routee.transit.compass_app import TransitCompassApp
from routee.transit.predictor import GTFSEnergyPredictor, _speed_model_cache_is_valid
from routee.transit.speed_model import (
    DEFAULT_SPEED_MODEL_DIR,
    SPEED_TRACE_MODEL_CONFIG,
    TEMPORAL_FEATURES,
    TIME_BIN_FEATURES,
    _write_speed_traces,
    insert_transit_speed_config,
    write_transit_speed_model,
)


def test_batched_traces_deduplicate_across_batches(tmp_path: Path) -> None:
    session = MagicMock()
    order = ["base"] + TEMPORAL_FEATURES

    def predict(_outputs: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        x = inputs["input"]
        assert len(x) <= 2
        assert x.dtype == np.float32
        period = x[:, 1:7] @ np.arange(6, dtype=np.float32)
        return [(x[:, 0] + period + x[:, 7] * 100).reshape(-1, 1)]

    session.run.side_effect = predict
    features = pd.DataFrame({"base": [2.0, 3.0, 2.0, 3.0, -200.0]})
    _write_speed_traces(features, order, session, tmp_path, batch_size=2)
    ids = np.fromfile(tmp_path / "transit_speed_edge_traces.bin", dtype="<u4")
    traces = np.fromfile(tmp_path / "transit_speed_traces.bin", dtype="<f4").reshape(
        -1, 12
    )
    np.testing.assert_array_equal(ids, [0, 1, 0, 1, 2])
    assert traces.shape == (3, 12)
    assert session.run.call_count == 3 * 12
    for day in range(2):
        for period in range(6):
            expected = 2 + period
            expected += 100 * (day == 0)
            assert traces[0, day * 6 + period] == expected
    np.testing.assert_array_equal(traces[2], np.ones(12))
    assert (tmp_path / "transit_speed_edge_traces.bin").stat().st_size == 4 * 5
    assert (tmp_path / "transit_speed_traces.bin").stat().st_size == 48 * 3


@pytest.mark.parametrize(
    "output", [np.array([[np.nan]]), np.ones((1, 2)), np.array([[np.inf]])]
)
def test_invalid_model_outputs_fail(tmp_path: Path, output: np.ndarray) -> None:
    session = MagicMock()
    session.run.return_value = [output]
    with pytest.raises(ValueError, match="Speed model returned"):
        _write_speed_traces(
            pd.DataFrame({"base": [1.0]}),
            ["base"] + TEMPORAL_FEATURES,
            session,
            tmp_path,
        )


def test_invalid_edge_order_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="edge_id order"):
        _write_speed_traces(
            pd.DataFrame({"base": [1.0]}, index=[1]),
            ["base"] + TEMPORAL_FEATURES,
            MagicMock(),
            tmp_path,
        )


def test_preprocessing_matches_bundled_onnx(tmp_path: Path) -> None:
    manifest = json.loads(
        (DEFAULT_SPEED_MODEL_DIR / "tuned_rf_speed_model_manifest.json").read_text()
    )
    order = manifest["input_order"]
    values = manifest["feature_medians"]
    features = pd.DataFrame(
        {
            name: [float(values.get(name, 0.0))] * 2
            for name in order
            if name not in TEMPORAL_FEATURES
        }
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(DEFAULT_SPEED_MODEL_DIR / "tuned_rf_speed_model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    _write_speed_traces(features, order, session, tmp_path, batch_size=2)
    traces = np.fromfile(tmp_path / "transit_speed_traces.bin", dtype="<f4")
    assert traces.shape == (12,)
    for bin_id in range(12):
        row = features.iloc[0].to_dict()
        row.update(
            {name: float(i == bin_id % 6) for i, name in enumerate(TIME_BIN_FEATURES)}
        )
        row["is_weekday"] = float(bin_id < 6)
        x = np.array([[row[name] for name in order]], dtype=np.float32)
        direct = np.asarray(session.run(None, {"input": x})[0]).item()
        assert traces[bin_id] == pytest.approx(round(max(direct, 1.0), 1), abs=1e-5)


def test_dataset_hook_runs_bundled_model(tmp_path: Path) -> None:
    from shapely.geometry import LineString

    params = MagicMock()
    params.output_directory = tmp_path
    params.edges = gpd.GeoDataFrame(
        {
            "edge_id": [0, 1],
            "maxspeed": ["25 mph", None],
            "lanes": ["2", None],
            "grade": [0.0, np.nan],
            "grade_abs": [0.0, np.nan],
            "distance": [200.0, 200.0],
            "highway": ["residential", "tertiary"],
        },
        geometry=[
            LineString([(-105, 40), (-104.99, 40)]),
            LineString([(-104.99, 40), (-104.98, 40)]),
        ],
        crs="EPSG:4326",
    )
    feed = MagicMock()
    feed.stops = pd.DataFrame({"stop_lon": [-104.995], "stop_lat": [40.0]})
    blocks = write_transit_speed_model(
        params,
        feed,
        DEFAULT_SPEED_MODEL_DIR / "tuned_rf_speed_model.onnx",
        DEFAULT_SPEED_MODEL_DIR / "tuned_rf_speed_model_manifest.json",
    )
    assert blocks == [SPEED_TRACE_MODEL_CONFIG]
    assert (
        np.fromfile(tmp_path / "transit_speed_edge_traces.bin", dtype="<u4").size == 2
    )
    speeds = np.fromfile(tmp_path / "transit_speed_traces.bin", dtype="<f4")
    assert speeds.size % 12 == 0
    assert np.isfinite(speeds).all()
    assert (speeds >= 1).all()


def test_speed_replaces_posted_speed_before_time(tmp_path: Path) -> None:
    config = Path(__file__).parents[1] / "routee/transit/resources/transit_energy.toml"
    output = tmp_path / "transit_energy.toml"
    output.write_text(config.read_text())
    insert_transit_speed_config(output, [SPEED_TRACE_MODEL_CONFIG.copy()])
    models = tomlkit.loads(output.read_text())["search"]["traversal"]["models"]
    types = [model["type"] for model in models]
    assert "speed" not in types
    assert "custom" not in types
    assert (
        types.index("transit_speed")
        < types.index("time")
        < types.index("transit_energy")
    )
    assert models[-1]["speed_feature_name"] == "transit_speed"


def test_trace_cache_validation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for source_name, cached_name in [
        ("tuned_rf_speed_model.onnx", "transit_speed_model.onnx"),
        ("tuned_rf_speed_model_manifest.json", "transit_speed_model_manifest.json"),
    ]:
        (source / source_name).write_bytes(b"source")
        (tmp_path / cached_name).write_bytes(b"source")
    config = tmp_path / "transit_energy.toml"
    config.write_text(
        tomlkit.dumps(
            {
                "search": {
                    "traversal": {
                        "models": [
                            SPEED_TRACE_MODEL_CONFIG.copy(),
                            {
                                "type": "transit_energy",
                                "speed_feature_name": "transit_speed",
                            },
                        ]
                    }
                },
            }
        )
    )
    mapping = tmp_path / "transit_speed_edge_traces.bin"
    mapping.write_bytes(np.array([0], dtype="<u4").tobytes())
    (tmp_path / "transit_speed_traces.bin").write_bytes(
        np.ones(12, dtype="<f4").tobytes()
    )
    assert _speed_model_cache_is_valid(tmp_path, source)
    mapping.write_bytes(b"x")
    assert not _speed_model_cache_is_valid(tmp_path, source)
    mapping.write_bytes(np.array([0], dtype="<u4").tobytes())
    (source / "tuned_rf_speed_model.onnx").write_bytes(b"changed")
    assert not _speed_model_cache_is_valid(tmp_path, source)


@pytest.fixture
def trace_app(tmp_path: Path) -> TransitCompassApp:
    (tmp_path / "vertices.csv").write_text(
        "vertex_id,x,y\n0,-105,40\n1,-104.99,40\n2,-104.98,40\n"
    )
    (tmp_path / "edges.csv").write_text(
        "edge_id,src_vertex_id,dst_vertex_id,distance\n0,0,1,1609.344\n1,1,2,1609.344\n"
    )
    (tmp_path / "geometry.txt").write_text(
        "LINESTRING (-105 40, -104.99 40)\nLINESTRING (-104.99 40, -104.98 40)\n"
    )
    np.array([0, 0], dtype="<u4").tofile(tmp_path / "transit_speed_edge_traces.bin")
    np.array([10.0] * 6 + [20.0] * 6, dtype="<f4").tofile(
        tmp_path / "transit_speed_traces.bin"
    )
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
                ],
            }
        },
        "cost": {
            "vehicle_rates": {
                "trip_time": {"type": "time", "factor": 1.0, "unit": "minutes"},
            }
        },
        "plugin": {
            "output_plugins": [
                {"type": "summary"},
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
        TransitCompassApp, TransitCompassApp.from_config_file(path, parallelism=2)
    )


def test_extension_uses_elapsed_time_and_shared_traces(
    trace_app: TransitCompassApp,
) -> None:
    queries = [
        {
            "path": [{"edge_id": 0}, {"edge_id": 1}],
            "start_time": "23:59:00",
            "start_weekday": "friday",
            "weights": {"trip_time": 1.0},
        }
    ] * 4
    results = trace_app.run_calculate_path(queries)
    assert isinstance(results, list)
    for result in results:
        assert "error" not in result, result
        route = result["route"]
        assert route["traversal_summary"]["trip_time"]["value"] == pytest.approx(9.0)
        states = [
            feature["properties"]["state"] for feature in route["path"]["features"]
        ]
        assert [state["edge_speed"] for state in states] == pytest.approx(
            [16.09344, 32.18688]
        )
        assert [state["transit_speed"] for state in states] == pytest.approx(
            [10.0, 20.0]
        )


def test_map_matching_uses_same_app() -> None:
    predictor = GTFSEnergyPredictor("/tmp/fake_gtfs")
    app = MagicMock()
    predictor.app = app
    shapes = [pd.DataFrame({"shape_id": ["s"]})]
    with (
        patch.object(predictor, "_create_map_match_query", return_value={}),
        patch.object(
            predictor, "_process_map_match_results", return_value=pd.DataFrame()
        ),
    ):
        predictor._match_shapes_to_network(shapes)
    app.map_match.assert_called_once()
    assert not hasattr(predictor, "routing_app")


@pytest.mark.parametrize("stale_cache", [False, True])
def test_new_dataset_constructs_only_one_app(tmp_path: Path, stale_cache: bool) -> None:
    from shapely.geometry import box

    predictor = GTFSEnergyPredictor(
        "/tmp/fake_gtfs", output_dir=tmp_path, overwrite=False
    )
    predictor.feed = MagicMock()
    predictor.shapes = pd.DataFrame({"shape_id": ["s"]})
    if stale_cache:
        cache_dir = tmp_path / "compass_app"
        cache_dir.mkdir()
        (cache_dir / "transit_energy.toml").write_text("")
    with (
        patch(
            "routee.transit.predictor.build_corridor_polygon",
            return_value=box(0, 0, 1, 1),
        ),
        patch("routee.transit.predictor.local_pbf_dir", return_value=None),
        patch("routee.transit.predictor.graph_from_polygon_with_retry"),
        patch.object(TransitCompassApp, "from_graph") as from_graph,
        patch.object(TransitCompassApp, "from_config_file") as from_config,
    ):
        predictor.load_compass_app()
    from_graph.assert_called_once()
    from_config.assert_not_called()
    assert from_graph.call_args.kwargs["overwrite"] == stale_cache
    assert predictor.app is from_graph.return_value


def test_extension_routing_uses_same_speeds(trace_app: TransitCompassApp) -> None:
    result = trace_app.run(
        {
            "origin_vertex": 0,
            "destination_vertex": 2,
            "start_time": "23:59:00",
            "start_weekday": "friday",
            "weights": {"trip_time": 1.0},
        }
    )
    assert isinstance(result, dict)
    assert "error" not in result, result
    assert result["route"]["traversal_summary"]["trip_time"]["value"] == pytest.approx(
        9.0
    )


def test_extension_map_matching_uses_same_speeds(trace_app: TransitCompassApp) -> None:
    result = trace_app.map_match(
        {
            "trace": [{"x": -105.0 + i * 0.001, "y": 40.0} for i in range(21)],
            "search_parameters": {
                "start_time": "23:59:00",
                "start_weekday": "friday",
                "weights": {"trip_time": 1.0},
            },
        }
    )
    assert isinstance(result, dict)
    assert "error" not in result, result
    features = result["matched_path"]["features"]
    assert len(features) == 2
    assert [
        feature["properties"]["state"]["transit_speed"] for feature in features
    ] == pytest.approx([10.0, 20.0])


def test_extension_speed_limit_and_invalid_queries(
    trace_app: TransitCompassApp,
) -> None:
    query = {
        "path": [{"edge_id": 0}, {"edge_id": 1}],
        "start_time": "12:00:00",
        "start_weekday": "saturday",
        "weights": {"trip_time": 1.0},
        "speed_limit": 5.0,
        "speed_limit_unit": "mph",
    }
    result = trace_app.run_calculate_path(query)
    assert isinstance(result, dict)
    assert "error" not in result, result
    assert result["route"]["traversal_summary"]["trip_time"]["value"] == pytest.approx(
        24.0
    )
    invalid_queries: list[dict[str, str | float]] = [
        {"start_time": "bad"},
        {"start_weekday": "holiday"},
        {"speed_limit": -1.0},
        {"speed_limit_unit": "bogus"},
    ]
    for changes in invalid_queries:
        result = trace_app.run_calculate_path({**query, **changes})
        assert isinstance(result, dict)
        assert "error" in result


def test_cached_vehicle_refresh_preserves_traces(tmp_path: Path) -> None:
    from shapely.geometry import box

    predictor = GTFSEnergyPredictor(
        "/tmp/fake_gtfs",
        output_dir=tmp_path,
        vehicle_models=["Transit_Bus_Diesel_40ft"],
        overwrite=False,
    )
    predictor.feed = MagicMock()
    predictor.shapes = pd.DataFrame({"shape_id": ["s"]})
    cache_dir = tmp_path / "compass_app"
    cache_dir.mkdir()
    config_path = cache_dir / "transit_energy.toml"
    template = (
        Path(__file__).parents[1] / "routee/transit/resources/transit_energy.toml"
    )
    config_path.write_text(template.read_text())
    with (
        patch(
            "routee.transit.predictor.build_corridor_polygon",
            return_value=box(0, 0, 1, 1),
        ),
        patch(
            "routee.transit.predictor._speed_model_cache_is_valid", return_value=True
        ),
        patch(
            "routee.transit.predictor.read_configured_vehicle_models",
            return_value=set(),
        ),
        patch("routee.transit.predictor.copy_transit_config"),
        patch("routee.transit.predictor.copy_custom_vehicle_models"),
        patch("routee.transit.predictor.patch_vehicle_speed_feature") as patch_vehicle,
        patch("routee.transit.predictor.write_transit_speed_model") as inference,
        patch("routee.transit.predictor.local_pbf_dir", return_value=None),
        patch("routee.transit.predictor.graph_from_polygon_with_retry") as download,
        patch.object(TransitCompassApp, "from_config_file") as load_app,
        patch.object(TransitCompassApp, "from_graph") as generate,
    ):
        predictor.load_compass_app()
    load_app.assert_called_once()
    generate.assert_not_called()
    download.assert_not_called()
    inference.assert_not_called()
    patch_vehicle.assert_called_once_with(
        cache_dir / "vehicles", ["Transit_Bus_Diesel_40ft"]
    )
    models = tomlkit.loads(config_path.read_text())["search"]["traversal"]["models"]
    assert any(model["type"] == "transit_speed" for model in models)
    assert not any(model["type"] == "speed" for model in models)
