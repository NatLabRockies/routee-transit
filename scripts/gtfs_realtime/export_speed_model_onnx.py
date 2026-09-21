"""Export a tuned speed model to ONNX for the energy-prediction pipeline.

Converts one of the tuned models persisted by
``fit_speed_models.fit_and_evaluate_models`` (``tuned_{model_key}_speed_model.joblib``,
default ``rf`` — ``RandomForestRegressor``, chosen for its ONNX support and
performance nearly matching the non-exportable tuned HGB) into an ONNX graph
that a Rust ``TransitSpeedModel`` traversal model can load and run per-edge,
per-query. The ONNX graph takes a single float vector input matching
``tuned_{model_key}_speed_model_manifest.json``'s ``input_order`` — the Rust
side is responsible for assembling that vector (static per-edge attributes +
one-hot ``functional_class`` from ``custom`` traversal models, plus
query-resolved ``hour``/``is_weekday``/``is_peak``) and does not need to know
about the sklearn/one-hot-encoding details, only the manifest's column order.

Note: ``hgb`` (HistGradientBoostingRegressor) cannot be exported with this
environment's skl2onnx/sklearn versions (boolean-attribute serialization bug
in its missing-value tree nodes); use ``rf`` or ``gbr`` instead.

Validates the exported graph against the original sklearn model on held-out
aggregated rows before declaring success.

Usage
-----
    python export_speed_model_onnx.py --results-dir reports/realtime_archive --model-key rf
"""

import argparse
import json
import logging
from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("reports/realtime_archive")
# Tolerances for sklearn-vs-ONNX prediction parity (mph).
MAX_ABS_DIFF_TOLERANCE = 0.05
MEAN_ABS_DIFF_TOLERANCE = 0.01


class _CategoricalFeatureManifest(TypedDict):
    name: str
    one_hot_columns: list[str]
    highway_to_functional_class: dict[str, str]
    default: str


class SpeedModelManifest(TypedDict):
    """Schema of ``tuned_{model_key}_speed_model_manifest.json`` (written by
    ``fit_speed_models._tune_and_persist_model``)."""

    target: str
    input_order: list[str]
    static_per_edge_features: list[str]
    per_query_temporal_features: list[str]
    categorical_feature: _CategoricalFeatureManifest
    feature_medians: dict[str, float] | None
    missing_indicator_features: list[str] | None


def build_inference_matrix(
    agg: pd.DataFrame, manifest: SpeedModelManifest
) -> npt.NDArray[np.float32]:
    """Assemble the model's input matrix from an aggregated training DataFrame.

    Mirrors what the Rust side will do per-edge/per-query: numeric + temporal
    columns pulled directly, and the categorical feature one-hot expanded in
    the exact column order recorded in the manifest. If the manifest carries
    ``feature_medians`` (models trained on imputed data, e.g. RF/GBR — unlike
    HGB, which handles NaN natively), missing values are filled with the same
    train-set medians used at fit time. If it also carries
    ``missing_indicator_features``, one trailing ``{feature}_was_missing``
    column per listed feature is appended (computed before imputation), so the
    model can still distinguish "originally missing" from "truly average".
    """
    cat = manifest["categorical_feature"]
    cat_name = cat["name"]
    numeric_cols = [c for c in manifest["static_per_edge_features"] if c != cat_name]
    temporal_cols = manifest["per_query_temporal_features"]
    one_hot_cols = cat["one_hot_columns"]

    X_num = agg[numeric_cols + temporal_cols].to_numpy(dtype=np.float32)

    missing_indicator_features = manifest.get("missing_indicator_features") or []
    if missing_indicator_features:
        indicator_idx = [numeric_cols.index(c) for c in missing_indicator_features]
        X_ind = np.isnan(X_num[:, indicator_idx]).astype(np.float32)
    else:
        X_ind = None

    feature_medians = manifest.get("feature_medians")
    if feature_medians is not None:
        medians = np.array(
            [feature_medians[c] for c in numeric_cols + temporal_cols],
            dtype=np.float32,
        )
        nan_mask = np.isnan(X_num)
        X_num[nan_mask] = np.broadcast_to(medians, X_num.shape)[nan_mask]
    dummies = pd.DataFrame(index=agg.index)
    for col in one_hot_cols:
        label = col[len(cat_name) + 1 :]  # strip "functional_class_" prefix
        dummies[col] = (agg[cat_name] == label).astype(np.float32)
    X_cat = dummies[one_hot_cols].to_numpy(dtype=np.float32)

    parts = [X_num, X_cat] + ([X_ind] if X_ind is not None else [])
    return np.hstack(parts).astype(np.float32)


def export_to_onnx(model_path: Path, manifest_path: Path, output_path: Path) -> None:
    bundle = joblib.load(model_path)
    manifest: SpeedModelManifest = json.loads(manifest_path.read_text())
    model = bundle["model"]
    n_features = len(manifest["input_order"])

    log.info("Converting %s (%d features) to ONNX…", type(model).__name__, n_features)
    onnx_model = convert_sklearn(
        model,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        target_opset=17,
    )
    output_path.write_bytes(onnx_model.SerializeToString())
    log.info("ONNX model saved → %s", output_path)


def validate_onnx(
    onnx_path: Path,
    model_path: Path,
    manifest_path: Path,
    agg_csv: Path,
    n_samples: int = 2000,
) -> None:
    bundle = joblib.load(model_path)
    manifest: SpeedModelManifest = json.loads(manifest_path.read_text())
    model = bundle["model"]

    agg = pd.read_csv(agg_csv)
    agg["functional_class"] = (
        agg["highway"]
        .map(manifest["categorical_feature"]["highway_to_functional_class"])
        .fillna(manifest["categorical_feature"]["default"])
    )
    if len(agg) > n_samples:
        agg = agg.sample(n_samples, random_state=42)

    X = build_inference_matrix(agg, manifest)

    sklearn_pred = model.predict(X.astype(np.float64))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_pred = session.run(None, {input_name: X})[0].ravel()

    abs_diff = np.abs(sklearn_pred - onnx_pred)
    log.info(
        "Validation on %d rows: max abs diff=%.4f mph, mean abs diff=%.4f mph",
        len(agg),
        abs_diff.max(),
        abs_diff.mean(),
    )
    if (
        abs_diff.max() > MAX_ABS_DIFF_TOLERANCE
        or abs_diff.mean() > MEAN_ABS_DIFF_TOLERANCE
    ):
        raise ValueError(
            f"ONNX predictions diverge from sklearn beyond tolerance "
            f"(max={abs_diff.max():.4f}, mean={abs_diff.mean():.4f}). "
            "Do not use this ONNX export."
        )
    log.info("Validation passed — ONNX predictions match sklearn within tolerance.")


def main(results_dir: Path, model_key: str) -> None:
    model_path = results_dir / f"tuned_{model_key}_speed_model.joblib"
    manifest_path = results_dir / f"tuned_{model_key}_speed_model_manifest.json"
    agg_csv = results_dir / "aggregated_training_data.csv"
    output_path = results_dir / f"tuned_{model_key}_speed_model.onnx"

    for path in (model_path, manifest_path, agg_csv):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run fit_speed_models.py / "
                "fit_archive_speed_models.py first (with --tune-hgb, the default)."
            )

    export_to_onnx(model_path, manifest_path, output_path)
    validate_onnx(output_path, model_path, manifest_path, agg_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory with fit_*_speed_models.py outputs "
        f"(default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--model-key",
        default="rf",
        choices=["rf", "gbr", "hgb"],
        help="Which tuned model bundle to export (default: rf; hgb will fail "
        "to convert with this environment's skl2onnx/sklearn versions).",
    )
    args = parser.parse_args()
    main(args.results_dir, args.model_key)
