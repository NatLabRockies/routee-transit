from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from branca.colormap import linear


def plot_link_speeds(link_distrib, color_column):
    # Geometry is in EPSG:4326 (lon, lat) — no transformation needed
    centroid = gpd.GeoSeries(link_distrib.index).union_all().centroid
    c_lon, c_lat = centroid.x, centroid.y

    # Create a map centered on the trip
    m = folium.Map(location=[c_lat, c_lon], zoom_start=13, tiles="cartodb-positron")

    # Create a colormap for the speeds
    vmin = 0
    vmax = min(
        60, link_distrib[color_column].dropna().max()
    )  # Cap at 60 mph or max speed
    colormap = linear.RdYlGn_09.scale(vmin, vmax)
    colormap.caption = f"{color_column} Speed (MPH) on Road Link"

    # Plot each road link with color based on speed
    for geom, row in link_distrib.iterrows():
        speed = row[color_column]

        # Skip if geometry is missing or speed is invalid
        if geom is None or pd.isna(speed) or np.isinf(speed):
            continue

        # Geometry is already in EPSG:4326 — coords are (lon, lat)
        coords = shapely.get_coordinates(geom)
        path_points = [[coords[i, 1], coords[i, 0]] for i in range(len(coords))]

        # Add the road segment to the map
        folium.PolyLine(
            locations=path_points,
            popup=f"{color_column}: {row[color_column]:.2f} MPH<br>Observations: {row['count']}",
            color=colormap(speed),
            weight=3,
            opacity=0.6,
        ).add_to(m)

    # Add the colormap legend
    m.add_child(colormap)

    # Display the map
    m.show_in_browser()


if __name__ == "__main__":
    data_dir = Path("reports/realtime/greater_portland_me")
    link_summary = pd.read_csv(data_dir / "realtime_link_speeds_20251023.csv")

    # Convert the WKT column to a GeoSeries
    geometry = gpd.GeoSeries.from_wkt(link_summary["geom"])

    # Create a GeoDataFrame
    link_summary = gpd.GeoDataFrame(
        link_summary.drop(columns=["geom"]), geometry=geometry, crs="EPSG:4326"
    )
    link_summary_filt = link_summary.dropna(subset=["mph"])
    link_summary_filt = link_summary_filt[link_summary_filt["mph"] < 80]

    link_distrib = (
        link_summary_filt.groupby("geometry")["mph"].describe().sort_values(by="count")
    )
    # Only keep links with at least 10 observations
    link_distrib = link_distrib[link_distrib["count"] >= 10]
    # Plot median speed
    plot_link_speeds(link_distrib, "50%")
