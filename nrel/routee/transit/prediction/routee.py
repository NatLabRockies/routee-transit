import multiprocessing as mp
from functools import partial
from pathlib import Path

import pandas as pd

import nrel.routee.powertrain as pt

# Set constants
MI_PER_KM = 0.6213712


def predict_trip_energy(
    t_df: pd.DataFrame,
    routee_model_str: str | Path,
) -> pd.DataFrame:
    """Predict energy consumption using a provided RouteE model and trip data.

    Args:
        t_df (pd.DataFrame): DataFrame containing trip link data, including distance,
            travel time, grade, and elevation.
        routee_model_str (str): String specifying a nrel.routee.powertrain model for energy
            estimation. This could be the name of a model package with RouteE Powertrain
            or the path to a file hosting one.

    Returns:
        pd.DataFrame: DataFrame with predicted energy consumption for each trip link.
    """
    required_columns = ["kilometers", "travel_time_minutes", "grade"]
    missing = [col for col in required_columns if col not in t_df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in t_df: {', '.join(missing)}. "
            "Required columns are 'kilometers', 'travel_time_minutes', and 'grade'."
        )

    routee_model = pt.load_model(routee_model_str)

    # Calculate speed and convert to mph
    t_df["miles"] = MI_PER_KM * t_df["kilometers"]
    t_df["gpsspeed"] = t_df["miles"] / (t_df["travel_time_minutes"] / 60)

    return routee_model.predict(links_df=t_df)


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
    )
    with mp.Pool(n_processes) as pool:
        predictions_by_trip = pool.map(predict_partial, links_df_by_trip)

    all_predictions = pd.concat(predictions_by_trip)

    routee_results = pd.concat([routee_input_df, all_predictions], axis=1)
    cols_incl = [
        "trip_id",
        "shape_id",
        "road_id",
        "kilometers",
        "travel_time_minutes",
        "grade",
    ] + list(all_predictions.columns)
    return routee_results[cols_incl]


def aggregate_results_by_trip(
    routee_results: pd.DataFrame, vehicle_name: str
) -> pd.DataFrame:
    """Summarize predictions by aggregating from links to trips."""
    agg_cols = [c for c in ["gallons", "kWhs"] if c in routee_results.columns]
    energy_by_trip = routee_results.groupby("trip_id").agg(
        {"kilometers": "sum", **{c: "sum" for c in agg_cols}}
    )
    energy_by_trip["miles"] = MI_PER_KM * energy_by_trip["kilometers"]
    energy_by_trip["vehicle"] = vehicle_name

    return energy_by_trip.drop(columns="kilometers")
