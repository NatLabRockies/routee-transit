import folium
import numpy as np
from branca.colormap import linear
import shapely
from pyproj import Transformer
import pandas as pd
import geopandas as gpd
from pathlib import Path

def plot_link_speeds(link_distrib, color_column):
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    centroid = gpd.GeoSeries(link_distrib.index).union_all().centroid
    centroid_coords = shapely.get_coordinates(centroid)
    c_lon, c_lat = transformer.transform(centroid_coords[:, 0], centroid_coords[:, 1])

    # Create a map centered on the trip
    m = folium.Map(
        location=[c_lat, c_lon], zoom_start=13, tiles="cartodb-positron"
    )

    # Create a colormap for the speeds
    vmin = 0
    vmax = min(50, link_distrib[color_column].max())  # Cap at 50 mph or max speed
    colormap = linear.YlOrRd_09.scale(vmin, vmax)
    colormap.caption = f"{color_column} Speed (MPH) on Road Link"

    # Plot each road link with color based on speed
    for geom, row in link_distrib.iterrows():
        speed = row[color_column]
        
        # Skip if geometry is missing or speed is invalid
        if geom is None or pd.isna(speed) or np.isinf(speed):
            continue
        
        # Transform coordinates from EPSG:3857 to EPSG:4326 (lat/lon)
        coords = shapely.get_coordinates(shapely.LineString(geom))
        lon, lat = transformer.transform(coords[:, 0], coords[:, 1])
        path_points = [[lat[i], lon[i]] for i in range(len(lat))]
        
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
    data_dir = Path("scripts/gtfs_realtime/greater_portland_me")
    link_summary = pd.read_csv(data_dir / "realtime_link_speeds_20251023.csv")

    # Convert the WKT column to a GeoSeries
    geometry = gpd.GeoSeries.from_wkt(link_summary["geom"])

    # Create a GeoDataFrame
    link_summary = gpd.GeoDataFrame(link_summary.drop(columns=["geom"]), geometry=geometry, crs="EPSG:3857")
    # link_summary = gpd.read_csv(data_dir / "realtime_link_speeds_20251023.csv", crs="EPSG:3857")

    link_distrib = link_summary.groupby("geometry")["mph"].describe().sort_values(by="count")
    # Only keep links with at least 10 observations
    link_distrib = link_distrib[link_distrib["count"] >= 10]
    # Plot median speed
    plot_link_speeds(link_distrib, "50%")