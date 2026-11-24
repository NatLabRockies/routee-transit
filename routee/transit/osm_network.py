"""OSM network downloads that survive transient Overpass failures.

The public Overpass API hard-blocks an IP (``Errno 111 Connection refused``)
after sustained use, which otherwise turns a single hiccup into a failed
analysis run. The helpers here retry with backoff and rotate through any
additional endpoints listed in ``ROUTEE_OVERPASS_URLS``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import osmnx as ox
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

# Rotating across instances matters because Overpass limits concurrent slots per
# source IP, and sites behind a shared egress NAT exhaust the main instance's
# quota collectively. Override with ROUTEE_OVERPASS_URLS (comma-separated) where
# some of these are unreachable -- NREL's egress, for one, is refused outright
# by overpass-api.de.
DEFAULT_OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
    "https://overpass.monicz.dev/api",
)

_TRANSIENT_ERROR_MARKERS = (
    "overpass",
    "connection",
    "refused",
    "timed out",
    "timeout",
    "max retries",
    "429",
    "502",
    "503",
    "504",
    "rate",
    "temporarily",
    "remote end closed",
)


def overpass_endpoints() -> list[str]:
    """Return the Overpass endpoints to rotate through, in priority order."""
    configured = os.environ.get("ROUTEE_OVERPASS_URLS", "").strip()
    if configured:
        endpoints = [url.strip() for url in configured.split(",") if url.strip()]
        if endpoints:
            return endpoints
    return list(DEFAULT_OVERPASS_ENDPOINTS)


def is_transient_overpass_error(error: BaseException) -> bool:
    """Return whether *error* looks like a retryable Overpass/network failure."""
    message = str(error).lower()
    return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)


def graph_from_polygon_with_retry(
    polygon: BaseGeometry,
    network_type: str = "drive",
    cycles: int = 3,
    base_wait: float = 60.0,
) -> Any:
    """Download a drivable OSM graph for *polygon*, retrying transient failures.

    Each cycle tries every endpoint once with no delay, so a mirror that refuses
    outright costs one immediate retry rather than a full backoff; the backoff
    only applies once every endpoint has failed.

    Successful responses are served from osmnx's on-disk cache, so retries and
    re-runs only hit Overpass for polygons that have never been fetched.
    """
    endpoints = overpass_endpoints()
    total_attempts = cycles * len(endpoints)
    original_url = ox.settings.overpass_url
    attempt = 0

    # ox.settings.overpass_url is global, so concurrent callers can land on the
    # same endpoint; harmless as long as every endpoint in rotation works.
    try:
        for cycle in range(cycles):
            for endpoint in endpoints:
                attempt += 1
                ox.settings.overpass_url = endpoint
                try:
                    return ox.graph_from_polygon(polygon, network_type=network_type)
                except Exception as error:
                    if attempt == total_attempts or not is_transient_overpass_error(
                        error
                    ):
                        raise
                    logger.warning(
                        "Overpass download failed on %s (attempt %d/%d): %s: %s",
                        endpoint,
                        attempt,
                        total_attempts,
                        type(error).__name__,
                        error,
                    )
            wait = base_wait * (cycle + 1)
            logger.warning(
                "All %d Overpass endpoints failed; retrying in %.0fs",
                len(endpoints),
                wait,
            )
            time.sleep(wait)
    finally:
        ox.settings.overpass_url = original_url

    raise RuntimeError("Overpass download retries exhausted")
