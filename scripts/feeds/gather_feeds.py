import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from scripts.feeds.extract_static_gtfs import GtfsExtractor
    import pandas as pd
    import marimo as mo
    import zipfile
    import requests
    import io
    import os
    from pathlib import Path
    return GtfsExtractor, Path, io, mo, os, pd, requests, zipfile


@app.cell
def _(GtfsExtractor, Path):
    db_root = Path("reports/mdb")
    extractor = GtfsExtractor()
    return db_root, extractor


@app.cell
def _(extractor):
    response = extractor.query_mobility_db(path="gtfs_feeds", query="?&status=active&country_code=US&subdivision_name=Washington")
    return (response,)


@app.cell
def _(response):
    active_feeds = [r for r in response if r["status"] == "active"]
    return (active_feeds,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # What should be included in a feeds table?
    * Feed ID
    * Provider
    * Official
    * Status (should be active)
    * Latest dataset ID
    * Possibly some processed location info, e.g. center/centroid of service area, maybe entire coverage area


    # How about in a datasets table?
    * Dataset ID
    * Includes shapes?
    * Includes bus trips? Would have to analyze ourselves
    * Service date range start
    * Service date range end
    * Validation details -- TBD what we need to check for. To start, maybe just mark if there are any errors and don't analyze those feeds.
    """
    )
    return


@app.cell
def _(active_feeds):
    active_feeds[0]
    return


@app.cell
def _(active_feeds):
    feed_info = list()
    for f in active_feeds:
        feed_info.append(
            {
                "id": f["id"],
                "name": f["feed_name"],
                "provider": f["provider"],
                "status": f["status"],
                "official": f["official"],
                "latest_dataset_id": f["latest_dataset"]["id"],
                "center_latitude": 0.5 * (f["bounding_box"]["minimum_latitude"] + f["bounding_box"]["maximum_latitude"]),
                "center_longitude": 0.5 * (f["bounding_box"]["minimum_longitude"] + f["bounding_box"]["maximum_longitude"]),
            }
        )
    return (feed_info,)


@app.cell
def _(feed_info, pd):
    feeds_df = pd.DataFrame(feed_info)
    return (feeds_df,)


@app.cell
def _(feeds_df):
    feeds_df
    return


@app.cell
def _():
    import folium
    return (folium,)


@app.cell
def _(feeds_df, folium):
    m = folium.Map(location=[feeds_df["center_latitude"].mean(), feeds_df["center_longitude"].mean()], zoom_start=8, tiles="cartodb.positron")
    for idx, row in feeds_df.iterrows():
        marker_color = 'blue' if row.official == True else 'red'
        folium.Marker(
            location=[row['center_latitude'], row['center_longitude']],
            icon=folium.Icon(color=marker_color),
            popup=f"{row.provider} ({row.id})",  # Optional popup text
        ).add_to(m)
    return (m,)


@app.cell
def _(m):
    m
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Okay, we've compiled the feed info. How do we decide what to retain?
    * For each agency identified, get its most recent dataset. Then:
        * Confirm that the dataset includes bus trips.
        * Confirm that the dataset includes shapes and any other features our pipeline requires.
        * (maybe) compile some key features like number of bus trips and routes, anything else we might use to summarize service for filtering/searching/aggregation
    """
    )
    return


@app.cell
def _(feeds_df):
    feeds_df["latest_dataset_id"]
    return


@app.cell
def _(Path, db_root, extractor, feeds_df, io, os, pd, requests, zipfile):
    # Build table summarizing datasets
    n_datasets = 5  #len(active_feeds)
    dataset_info = list()
    for d_id in feeds_df["latest_dataset_id"].tolist()[:n_datasets]:
        # Grab the dataset
        dataset_response = extractor.query_mobility_db(path=f"datasets/gtfs/{d_id}", query="")
        val_report = dataset_response["validation_report"]
        this_dataset_summary = {
            "id": d_id,
            "has_shapes": True if "Shapes" in val_report["features"] else False,
            "has_errors": True if val_report["total_error"] > 0 else False,
            "service_date_range_start": dataset_response["service_date_range_start"],
            "service_date_range_end": dataset_response["service_date_range_end"],
            "hosted_url": dataset_response["hosted_url"],
        }

        if this_dataset_summary["has_shapes"] is True and this_dataset_summary["has_errors"] is False:
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

            bus_routes = routes[routes["route_type"] == 3]["route_id"].tolist()
            bus_trips = trips[trips["route_id"].isin(bus_routes)]

            if len(bus_trips) > 1:
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
    return (datasets_df,)


@app.cell
def _(datasets_df):
    datasets_df
    return


@app.cell
def _(datasets_df):
    datasets_df.to_csv("reports/mdb/datasets.csv", index=False)
    return


@app.cell
def _(feeds_df):
    feeds_df.to_csv("reports/mdb/feeds.csv", index=False)
    return


@app.cell
def _(datasets_df):
    valid_datasets = datasets_df[
        (~datasets_df["has_errors"]) & datasets_df["includes_bus_trips"] & datasets_df["includes_all_bus_shapes"]
    ]
    return (valid_datasets,)


@app.cell
def _(valid_datasets):
    valid_datasets
    return


if __name__ == "__main__":
    app.run()
