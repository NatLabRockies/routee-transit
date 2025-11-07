import argparse

import folium
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map GTFS feeds.")
    parser.add_argument("feeds_path", help="Path to the feeds.csv table")
    parser.add_argument("map_path", help="Destination path for the HTML map file")
    args = parser.parse_args()

    feeds_df = pd.read_csv(args.feeds_path)

    m = folium.Map(
        location=[
            feeds_df["center_latitude"].mean(),
            feeds_df["center_longitude"].mean(),
        ],
        zoom_start=8,
        tiles="cartodb.positron",
    )
    for idx, row in feeds_df.iterrows():
        marker_color = "blue" if row.official else "red"
        folium.Marker(
            location=[row["center_latitude"], row["center_longitude"]],
            icon=folium.Icon(color=marker_color),
            popup=f"{row.provider} ({row.id})",  # Optional popup text
        ).add_to(m)

    m.save(args.map_path)
