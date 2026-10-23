"""Map a corridor polygon to the Geofabrik state extracts that cover it.

Geofabrik publishes per-state OSM extracts, which is the right granularity for
this pipeline. Cutting corridors out of a national PBF is not: osmium's
complete_ways strategy keeps a node-ID bitmap per output region, so memory grows
with the number of regions (measured: >350 GB for 150 corridors) regardless of
how small those corridors are. Reading a state file directly avoids that
entirely, and keeps the Overpass-style "polygon in, network out" flow.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

STATE_SHAPEFILE_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"
)
GEOFABRIK_US_BASE = "https://download.geofabrik.de/north-america/us"

# Geofabrik publishes these outside the us/ tree, so they need their own URLs.
_NON_STATE_SLUGS = {
    "puerto-rico": "https://download.geofabrik.de/north-america/us/puerto-rico-latest.osm.pbf",
}

# Territories with no Geofabrik US extract; datasets there fall back to Overpass.
UNSUPPORTED_SLUGS = frozenset(
    {
        "united-states-virgin-islands",
        "guam",
        "commonwealth-of-the-northern-mariana-islands",
        "american-samoa",
    }
)


def fetch_states_gdf(cache_dir: Path) -> gpd.GeoDataFrame:
    """Load US state boundaries, downloading the Census archive only once."""
    archive = cache_dir / STATE_SHAPEFILE_URL.rsplit("/", 1)[-1]
    if not archive.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading state boundaries from {STATE_SHAPEFILE_URL}")
        response = requests.get(STATE_SHAPEFILE_URL, timeout=120)
        response.raise_for_status()
        # Write aside and rename so an interrupted download is never cached.
        partial = archive.with_suffix(".part")
        partial.write_bytes(response.content)
        partial.replace(archive)

    states = gpd.read_file(archive)
    states["slug"] = states["NAME"].str.lower().str.replace(" ", "-", regex=False)
    return states.to_crs("EPSG:4326")


def states_for_polygon(polygon: BaseGeometry, cache_dir: Path) -> list[str]:
    """Return Geofabrik slugs for every state the polygon touches.

    Transit corridors near a border legitimately span states, so this returns
    all intersecting states rather than just the one containing the centroid.
    """
    states = fetch_states_gdf(cache_dir)
    hits = states[states.intersects(polygon)]
    slugs = sorted(set(hits["slug"].tolist()) - UNSUPPORTED_SLUGS)
    if not slugs:
        raise ValueError(
            "Corridor polygon does not intersect any US state extract "
            f"(bounds={tuple(round(v, 3) for v in polygon.bounds)})"
        )
    return slugs


def state_pbf_url(slug: str) -> str:
    return _NON_STATE_SLUGS.get(slug, f"{GEOFABRIK_US_BASE}/{slug}-latest.osm.pbf")


def state_pbf_path(slug: str, pbf_dir: Path) -> Path:
    return pbf_dir / f"{slug}-latest.osm.pbf"


def download_state_pbf(slug: str, pbf_dir: Path, timeout: int = 1800) -> Path:
    """Download one state extract if it isn't already present."""
    dest = state_pbf_path(slug, pbf_dir)
    if dest.is_file():
        return dest

    pbf_dir.mkdir(parents=True, exist_ok=True)
    url = state_pbf_url(slug)
    logger.info(f"Downloading {url}")
    partial = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    partial.replace(dest)
    return dest
