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


def local_pbf_path(dataset_id: str | None = None) -> Path | None:
    """Return the local OSM extract to read, or None to use Overpass.

    A per-dataset extract from ``$ROUTEE_OSM_EXTRACT_DIR`` is strongly
    preferred: pyrosm has no spatial index and its cost scales superlinearly
    with file size, so reading a national PBF per dataset is unusably slow
    (0.1s for a 44 MB state file vs >10 min for the 12 GB US file).
    ``$ROUTEE_OSM_PBF`` names a single file and is only practical for
    region-sized extracts.
    """
    extract_dir = os.environ.get("ROUTEE_OSM_EXTRACT_DIR", "").strip()
    if extract_dir and dataset_id:
        candidate = Path(extract_dir) / f"{dataset_id}.osm.pbf"
        if candidate.is_file():
            return candidate
        logger.warning(
            f"No OSM extract for {dataset_id} in {extract_dir}; "
            f"falling back to $ROUTEE_OSM_PBF or Overpass"
        )

    configured = os.environ.get("ROUTEE_OSM_PBF", "").strip()
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
    pbf_path: Path,
    network_type: str = "driving",
) -> Any:
    """Build a simplified, osmnx-compatible drive graph for *polygon*.

    ``network_type`` uses pyrosm's vocabulary ("driving"), not osmnx's ("drive").
    """
    from pyrosm import OSM

    osm = OSM(str(pbf_path), bounding_box=polygon)
    nodes, ways = osm.get_network(network_type=network_type, nodes=True)
    if nodes is None or ways is None or ways.empty:
        raise ValueError(
            f"No {network_type} network found in {pbf_path} for the requested area"
        )

    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "epsg:4326"

    for node in nodes.itertuples(index=False):
        # No osmid attribute: osmnx keeps it on the index, and Compass's
        # generate_compass_dataset reset_index()es that into a column.
        graph.add_node(node.id, x=float(node.lon), y=float(node.lat))

    present = [tag for tag in EDGE_TAGS if tag in ways.columns]
    for row in ways.to_dict("records"):
        u, v = row["u"], row["v"]
        if u not in graph or v not in graph:
            continue
        attrs = {tag: row[tag] for tag in present if pd.notna(row.get(tag))}
        attrs["osmid"] = row.get("id")
        attrs["length"] = float(row.get("length") or 0.0)
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
        raise ValueError(f"Built an empty graph from {pbf_path} for the requested area")

    simplified = ox.simplify_graph(graph)
    logger.info(
        f"Built graph from {pbf_path.name}: {simplified.number_of_nodes()} nodes, "
        f"{simplified.number_of_edges()} edges"
    )
    return simplified
