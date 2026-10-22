"""Build OSM road graphs from a local PBF extract instead of the Overpass API.

Overpass is the wrong tool for bulk extraction -- its own documentation points
bulk users at planet dumps and says a national-scale run should not go through
the public instances. Reading a local extract removes the rate limits, IP bans,
and server errors entirely, and is dramatically faster.

The graph produced here is handed to the same ``osmnx.simplify_graph`` that
``osmnx.graph_from_polygon`` uses internally, so topology and attribute handling
match the Overpass path rather than merely resembling it.

Note that ``pyrosm.OSM.to_graph`` is deliberately NOT used: it re-aggregates
segments back into ways and collapses their tags into list-valued attributes,
which destroys ``oneway`` (observed: only 5.6% of edges retained a usable
value). The segment table from ``get_network(nodes=True)`` is already clean, so
the graph is assembled from that directly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from routee.transit.osm_states import (
    download_state_pbf,
    state_pbf_path,
    states_for_polygon,
)

logger = logging.getLogger(__name__)

# Edge tags the energy and speed models read. Anything not listed is dropped to
# keep the generated Compass dataset small.
EDGE_TAGS = (
    "highway",
    "maxspeed",
    "lanes",
    "name",
    "oneway",
    "bridge",
    "tunnel",
    "junction",
    "access",
    "ref",
    "width",
    "service",
)

_ONEWAY_TRUE = frozenset({"yes", "true", "1", "-1", "reversible"})
_IMPLICIT_ONEWAY_JUNCTIONS = frozenset({"roundabout", "circular"})


def local_pbf_dir() -> Path | None:
    """Return the directory of Geofabrik state extracts, or None for Overpass.

    Set ``$ROUTEE_OSM_PBF_DIR`` to enable the local path. Per-state extracts are
    the right granularity: the whole-US file is unusable here because pyrosm has
    no spatial index and its cost grows superlinearly with file size (0.1s for a
    44 MB state file vs >10 min for the 12 GB national one).
    """
    configured = os.environ.get("ROUTEE_OSM_PBF_DIR", "").strip()
    return Path(configured) if configured else None


def _is_oneway(row: dict[str, Any]) -> bool:
    value = row.get("oneway")
    if isinstance(value, str) and value.strip().lower() in _ONEWAY_TRUE:
        return True
    junction = row.get("junction")
    return (
        isinstance(junction, str)
        and junction.strip().lower() in _IMPLICIT_ONEWAY_JUNCTIONS
    )


def graph_from_local_pbf(
    polygon: BaseGeometry,
    pbf_dir: Path,
    network_type: str = "driving",
    download_missing: bool = True,
) -> Any:
    """Build a simplified, osmnx-compatible drive graph for *polygon*.

    Reads every Geofabrik state extract the polygon touches, so corridors that
    cross a state line are not silently truncated. ``network_type`` uses
    pyrosm's vocabulary ("driving"), not osmnx's ("drive").
    """
    from pyrosm import OSM

    slugs = states_for_polygon(polygon, pbf_dir / "_boundaries")
    bbox = list(polygon.bounds)

    node_frames: list[pd.DataFrame] = []
    way_frames: list[pd.DataFrame] = []
    for slug in slugs:
        pbf_path = state_pbf_path(slug, pbf_dir)
        if not pbf_path.is_file():
            if not download_missing:
                raise FileNotFoundError(f"Missing state extract: {pbf_path}")
            download_state_pbf(slug, pbf_dir)

        osm = OSM(str(pbf_path), bounding_box=bbox)
        nodes, ways = osm.get_network(network_type=network_type, nodes=True)
        if nodes is not None and ways is not None and not ways.empty:
            node_frames.append(nodes)
            way_frames.append(ways)

    if not way_frames:
        raise ValueError(
            f"No {network_type} network found in {slugs} for the requested area"
        )

    # State extracts overlap slightly at borders, so drop the duplicates a
    # multi-state corridor picks up.
    nodes = pd.concat(node_frames, ignore_index=True).drop_duplicates(subset="id")
    ways = pd.concat(way_frames, ignore_index=True).drop_duplicates(
        subset=["u", "v", "id"]
    )
    logger.info(
        f"Read {len(ways)} way segments from {len(slugs)} state extract(s): "
        f"{', '.join(slugs)}"
    )

    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "epsg:4326"

    for node in nodes.itertuples(index=False):
        # No osmid attribute: osmnx keeps it on the index, and Compass's
        # generate_compass_dataset reset_index()es that into a column.
        graph.add_node(node.id, x=float(node.lon), y=float(node.lat))

    present = [tag for tag in EDGE_TAGS if tag in ways.columns]
    for raw_row in ways.to_dict("records"):
        row: dict[str, Any] = {str(key): value for key, value in raw_row.items()}
        u, v = row["u"], row["v"]
        if u not in graph or v not in graph:
            continue
        attrs = {tag: row[tag] for tag in present if pd.notna(row.get(tag))}
        attrs["osmid"] = row.get("id")
        length = row.get("length")
        attrs["length"] = float(length) if pd.notna(length) else 0.0
        geometry = row.get("geometry")
        if geometry is not None:
            attrs["geometry"] = geometry
        attrs["oneway"] = _is_oneway(row)

        graph.add_edge(u, v, **attrs)
        if not attrs["oneway"]:
            reverse = dict(attrs)
            if isinstance(geometry, LineString):
                reverse["geometry"] = LineString(list(geometry.coords)[::-1])
            graph.add_edge(v, u, **reverse)

    if graph.number_of_edges() == 0:
        raise ValueError(f"Built an empty graph from {slugs} for the requested area")

    simplified = ox.simplify_graph(graph)
    logger.info(
        f"Built graph: {simplified.number_of_nodes()} nodes, "
        f"{simplified.number_of_edges()} edges"
    )
    return simplified
