from __future__ import annotations

from pathlib import Path

import logging

import requests

from nrel.routee.transit.prediction.grade.tile_resolution import TileResolution

log = logging.getLogger(__name__)

CACHE_DIR = Path("cache")


# TODO: This logic might be better for inclusion in the gradit package
def _get_usgs_tiles(lat_lon_pairs: list[tuple[float, float]]) -> list[str]:
    def tile_index(lat: float, lon: float) -> str:
        if lat < 0 or lon > 0:
            raise ValueError(
                f"USGS Tiles are not available for point ({lat}, {lon}). "
                "Consider re-running with `grade=False`."
            )

        lat_deg = int(lat) + 1
        lon_deg = abs(int(lon)) + 1

        return f"n{lat_deg:02}w{lon_deg:03}"

    tiles = set()
    for lat, lon in lat_lon_pairs:
        tile = tile_index(lat, lon)
        tiles.add(tile)

    return list(tiles)


def _build_download_link(
    tile: str, resolution: TileResolution = TileResolution.ONE_THIRD_ARC_SECOND
) -> str:
    base_link_fragment = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/"
    resolution_link_fragment = f"{resolution.value}/TIFF/current/{tile}/"
    filename = f"USGS_{resolution.value}_{tile}.tif"
    link = base_link_fragment + resolution_link_fragment + filename

    return link


def _download_tile(
    tile: str,
    output_dir: Path = CACHE_DIR,
    resolution: TileResolution = TileResolution.ONE_THIRD_ARC_SECOND,
) -> Path:
    url = _build_download_link(tile, resolution)
    filename = url.split("/")[-1]
    destination = output_dir / tile / filename
    if not destination.parent.exists():
        destination.parent.mkdir(parents=True)

    if destination.is_file():
        log.info(f"{str(destination)} already exists, skipping")
        return destination

    with requests.get(url, stream=True) as r:
        log.info(f"downloading {tile}")
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise ValueError(
                f"Failed to download USGS tile {tile} from {url}. "
                "If this road network is outside of the US, consider re-running without "
                "GeneratePipelinePhase.GRADE in the `phases` argument."
            ) from e

        destination.parent.mkdir(exist_ok=True)

        # write to file in chunks
        with destination.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return destination


def download_usgs_tiles(
    lat_lon_pairs: list[tuple[float, float]],
    output_dir: Path = CACHE_DIR,
    resolution: TileResolution = TileResolution.ONE_THIRD_ARC_SECOND,
) -> list[Path]:
    """Download USGS elevation tiles for the given latitude and longitude pairs.

    Args:
        lat_lon_pairs (list[tuple[float, float]]): List of (latitude, longitude) pairs.
        output_dir (Path): Directory to save downloaded tiles.
        resolution (TileResolution): Resolution of the tiles to download.

    Returns:
        list[Path]: List of paths to downloaded tile files.
    """
    tiles = _get_usgs_tiles(lat_lon_pairs)
    log.info(f"Downloading {len(tiles)} USGS tiles at {resolution.name} resolution.")
    downloaded_tiles = [_download_tile(tile, output_dir, resolution) for tile in tiles]
    return downloaded_tiles
