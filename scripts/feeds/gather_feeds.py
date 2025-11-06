import argparse
import io
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests

from scripts.feeds.extract_static_gtfs import GtfsExtractor

GTFS_ROUTE_TYPE_BUS = 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather GTFS feeds and datasets.")
    parser.add_argument(
        "--db_root",
        type=str,
        default="reports/mdb",
        help="Root directory for storing GTFS datasets and feed info.",
    )
    parser.add_argument(
        "--states",
        type=str,
        nargs="*",
        default=[],
        help="State(s) from which to pull feeds (full name). Provide multiple states separated by spaces. Leave empty for all states.",
    )
    args = parser.parse_args()

    db_root = Path(args.db_root)
    extractor = GtfsExtractor()

    query = "?&status=active&country_code=US"

    # Gather feeds from specified states or all states
    active_feeds = []
    if args.states:
        # Query each state separately and combine results
        for state in args.states:
            state_query = query + f"&subdivision_name={state}"
            response = extractor.query_mdb_feeds(query=state_query)
            active_feeds.extend([r for r in response if r["status"] == "active"])
    else:
        # No states specified, get all US feeds
        response = extractor.query_mdb_feeds(query=query)
        active_feeds = [r for r in response if r["status"] == "active"]

    feed_info = list()
    for f in active_feeds:
        # Make sure latest dataset and bounding box are available
        bbox = f["bounding_box"]
        latest_data = f["latest_dataset"]

        if latest_data is None:
            print(
                f"Feed {f['id']} does not have a latest dataset identified. "
                "Skipping this feed."
            )
            continue

        if bbox is None:
            # This should only happen if latest_data is also None, but we'll check
            # the bounding box as well just in case.
            print(
                f"Feed {f['id']} is missing a service area bounding box. "
                "Skiping this feed."
            )
            continue

        try:
            feed_info.append(
                {
                    "id": f["id"],
                    "name": f["feed_name"],
                    "provider": f["provider"],
                    "status": f["status"],
                    "official": f["official"],
                    "latest_dataset_id": f["latest_dataset"]["id"],
                    "center_latitude": 0.5
                    * (
                        f["bounding_box"]["minimum_latitude"]
                        + f["bounding_box"]["maximum_latitude"]
                    ),
                    "center_longitude": 0.5
                    * (
                        f["bounding_box"]["minimum_longitude"]
                        + f["bounding_box"]["maximum_longitude"]
                    ),
                }
            )
        except KeyError as err:
            print(
                f"KeyError: Missing key '{err.args[0]}' in feed with id "
                f"'{f.get('id', 'unknown')}'. Skipping this feed."
            )
            continue

    feeds_df = pd.DataFrame(feed_info)

    # Build table summarizing datasets
    n_datasets = len(active_feeds)
    dataset_info = list()
    for d_id in feeds_df["latest_dataset_id"].tolist()[:n_datasets]:
        # Grab the dataset
        dataset_response = extractor.query_mdb_dataset(dataset_id=d_id)
        val_report = dataset_response["validation_report"]
        if val_report is None:
            # The validation report is sometimes missing. In that case, we'll mark
            # validation-related columns as NA.
            print(f"Validation report is missing for dataset {d_id}.")
            has_shapes = None
            has_errors = None
        else:
            has_shapes = True if "Shapes" in val_report["features"] else False
            has_errors = True if val_report["total_error"] > 0 else False

        this_dataset_summary = {
            "id": d_id,
            "has_shapes": has_shapes,
            "has_errors": has_errors,
            "service_date_range_start": dataset_response["service_date_range_start"],
            "service_date_range_end": dataset_response["service_date_range_end"],
            "hosted_url": dataset_response["hosted_url"],
        }

        if (
            this_dataset_summary["has_shapes"] is True
            and this_dataset_summary["has_errors"] is False
        ):
            # Unzip, check for bus trips and sufficient shapes
            # Download and extract
            download_zip = requests.get(this_dataset_summary["hosted_url"], timeout=60)
            download_zip.raise_for_status()

            extract_path = Path(db_root / d_id / "gtfs")
            os.makedirs(extract_path, exist_ok=True)

            with zipfile.ZipFile(io.BytesIO(download_zip.content)) as zip_ref:
                zip_ref.extractall(extract_path)
                print(f"Dataset extracted to {extract_path}")

            # Verify we have bus trips
            routes = pd.read_csv(extract_path / "routes.txt")
            trips = pd.read_csv(extract_path / "trips.txt")

            bus_routes = routes[routes["route_type"] == GTFS_ROUTE_TYPE_BUS][
                "route_id"
            ].tolist()
            bus_trips = trips[trips["route_id"].isin(bus_routes)]

            if len(bus_trips) >= 1:
                print("\tDataset includes bus trips")
                this_dataset_summary["includes_bus_trips"] = True
            else:
                print("\tNo bus trips in dataset")
                this_dataset_summary["includes_bus_trips"] = False

            if "shape_id" in bus_trips.columns:
                if bus_trips["shape_id"].isna().sum() == 0:
                    print("\tAll bus trips have shapes provided")
                    this_dataset_summary["includes_all_bus_shapes"] = True
                else:
                    print("\tSome bus trips are missing shapes")
                    this_dataset_summary["includes_all_bus_shapes"] = False
            else:
                print("\tNo shapes in dataset")
                this_dataset_summary["includes_all_bus_shapes"] = False

        dataset_info.append(this_dataset_summary)

    datasets_df = pd.DataFrame(dataset_info)

    # Write results summary tables
    datasets_df.to_csv(db_root / "datasets.csv", index=False)
    feeds_df.to_csv(db_root / "feeds.csv", index=False)
