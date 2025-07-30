import multiprocessing as mp
from functools import partial
from pathlib import Path

import nrel.routee.powertrain as pt
import pandas as pd

# Set constants
FEET_PER_KM = 3280.84
MI_PER_KM = 0.6213712


def predict_trip_energy(
    t_df: pd.DataFrame,
    routee_model_str: str | Path,
    feature_cols: list[str],
    distance_col: str,
) -> pd.DataFrame:
    """Predict energy consumption using a provided RouteE model and trip data.

    Args:
        t_df (pd.DataFrame): DataFrame containing trip link data, including geometry,
            distance, travel time, and elevation.
        routee_model: String specifying a nrel.routee.powertrain model for energy
            estimation. This could be the name of a model package with RouteE Powertrain
            or the path to a file hosting one.
        feature_cols (list of str): List of feature column names to use for prediction.
        distance_col (str): Name of the column representing distance in the features.

    Returns:
        pd.DataFrame: DataFrame with predicted energy consumption for each trip link.
    """
    routee_model = pt.load_model(routee_model_str)

    # Calculate speed and convert to mph
    t_df["gpsspeed"] = MI_PER_KM * t_df["kilometers"] / (t_df["travel_time_osm"] / 3600)
    t_df["miles"] = MI_PER_KM * t_df["kilometers"]
    t_df["grade"] = t_df["grade_dec_unfiltered"]

    pt_result = routee_model.predict(
        links_df=t_df, feature_columns=feature_cols, distance_column=distance_col
    )

    return pt_result


def predict_for_all_trips(
    routee_input_df: pd.DataFrame,
    routee_vehicle_model: str | Path,
    n_processes: int,
) -> pd.DataFrame:
    """Predict energy consumption for a set of trips in parallel."""
    links_df_by_trip = [
        routee_input_df[routee_input_df["trip_id"] == trip_id].copy()
        for trip_id in routee_input_df["trip_id"].unique()
    ]
    # Run RouteE energy prediction in parallel
    predict_partial = partial(
        predict_trip_energy,
        routee_model_str=routee_vehicle_model,
        feature_cols=["gpsspeed", "grade"],
        distance_col="miles",
    )
    with mp.Pool(n_processes) as pool:
        predictions_by_trip = pool.map(predict_partial, links_df_by_trip)

    all_predictions = pd.concat(predictions_by_trip)

    routee_results = pd.concat([routee_input_df, all_predictions], axis=1)
    cols_incl = [
        "trip_id",
        "shape_id",
        "road_id",
        "geom",
        "kilometers",
        "travel_time_osm",
        "grade_dec_unfiltered",
    ] + list(all_predictions.columns)
    return routee_results[cols_incl]
