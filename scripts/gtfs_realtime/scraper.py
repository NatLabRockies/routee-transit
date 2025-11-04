#!/usr/bin/env python3
"""
Continuous GTFS Realtime scraper that polls feeds at regular intervals
and logs complete records to files.
"""

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from google.protobuf.json_format import MessageToDict
from google.transit import gtfs_realtime_pb2

parser = argparse.ArgumentParser("GTFS scraper. runs until interrupted (ctl-c)")
parser.add_argument("feed", help="URL to scrape GTFS data from")
parser.add_argument(
    "--log-dir", type=Path, default=Path("gtfs_logs"), help="directory to write results"
)
parser.add_argument(
    "--retries",
    type=int,
    default=3,
    help="URL connection retry policy before terminating",
)
parser.add_argument(
    "--interval",
    type=int,
    default=30,
    help="time, in seconds, between polling the feed",
)
parser.add_argument(
    "--timeout",
    type=int,
    default=10,
    help="time, in seconds, to allow the HTTP connection to run for a given polling request",
)


class GTFSRealtimeScraper:
    """Continuous scraper for GTFS Realtime feeds with logging capabilities."""

    def __init__(
        self,
        feed_url: str,
        poll_interval: int = 30,
        log_dir: str = "logs",
        max_retries: int = 3,
        timeout: int = 10,
    ):
        """
        Initialize the scraper.

        Args:
            feed_url: URL of the GTFS Realtime feed
            poll_interval: Time between requests in seconds (default: 60)
            log_dir: Directory to store log files
            max_retries: Maximum number of retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        self.feed_url = feed_url
        self.poll_interval = poll_interval
        self.log_dir = Path(log_dir)
        self.max_retries = max_retries
        self.timeout = timeout
        self.running = False

        # Create log directory
        self.log_dir.mkdir(exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> None:
        """Configure logging to both file and console."""
        log_filename = (
            self.log_dir / f"scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Scraper initialized. Logs will be written to {log_filename}")

    # type ignore: handler signature set by the signal package
    def _signal_handler(self, signum, frame) -> None:  # type: ignore
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.running = False

    def _fetch_feed_data(self) -> Optional[gtfs_realtime_pb2.FeedMessage]:
        """
        Fetch and parse GTFS Realtime feed data with retry logic.

        Returns:
            Parsed FeedMessage or None if failed
        """
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(
                    f"Fetching data from {self.feed_url} (attempt {attempt + 1})"
                )

                response = requests.get(self.feed_url, timeout=self.timeout)
                response.raise_for_status()

                feed = gtfs_realtime_pb2.FeedMessage()
                feed.ParseFromString(response.content)

                self.logger.debug(
                    f"Successfully parsed feed with {len(feed.entity)} entities"
                )
                return feed

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
            except Exception as e:
                self.logger.error(f"Parsing failed (attempt {attempt + 1}): {e}")

            if attempt < self.max_retries:
                wait_time = 2**attempt  # Exponential backoff
                self.logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        self.logger.error(f"Failed to fetch data after {self.max_retries + 1} attempts")
        return None

    def _extract_vehicle_data(
        self, feed: gtfs_realtime_pb2.FeedMessage
    ) -> List[Dict[str, Any]]:
        """
        Extract vehicle position data from the feed.

        Args:
            feed: Parsed GTFS Realtime feed

        Returns:
            List of vehicle data dictionaries
        """
        vehicles = []

        for entity in feed.entity:
            if entity.HasField("vehicle"):
                vehicles.append(MessageToDict(entity))
        return vehicles

    def _write_records(self, vehicles: List[Dict[str, Any]]) -> None:
        """
        Write vehicle records to log files.

        Args:
            vehicles: List of vehicle data dictionaries
        """
        if not vehicles:
            return

        # Create filename with current date
        date_str = datetime.now().strftime("%Y%m%d")
        records_file = self.log_dir / f"gtfs_realtime_records_{date_str}.jsonl"

        try:
            with open(records_file, "a", encoding="utf-8") as f:
                for vehicle in vehicles:
                    f.write(json.dumps(vehicle) + "\n")

            self.logger.info(f"Wrote {len(vehicles)} vehicle records to {records_file}")

        except Exception as e:
            self.logger.error(f"Failed to write records: {e}")

    def run(self) -> None:
        """Start the continuous scraping process."""
        self.logger.info(f"Starting continuous scraper for {self.feed_url}")
        self.logger.info(f"Poll interval: {self.poll_interval} seconds")
        self.logger.info(f"Log directory: {self.log_dir.absolute()}")

        self.running = True

        while self.running:
            try:
                # Fetch and process data
                feed = self._fetch_feed_data()

                if feed:
                    vehicles = self._extract_vehicle_data(feed)
                    self._write_records(vehicles)
                    self.logger.info(f"Processed {len(vehicles)} vehicles")
                else:
                    self.logger.warning("No data retrieved this cycle")

                # Wait for next poll
                if self.running:  # Check if we should still be running
                    self.logger.debug(
                        f"Waiting {self.poll_interval} seconds until next poll..."
                    )
                    time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in main loop: {e}")
                if self.running:
                    time.sleep(self.poll_interval)

        self.logger.info("Scraper stopped")


def main() -> None:
    """Main entry point for the scraper."""

    args = parser.parse_args()

    # Create and run scraper
    scraper = GTFSRealtimeScraper(
        feed_url=args.feed,
        poll_interval=args.interval,
        log_dir=args.log_dir,
        timeout=args.timeout,
        max_retries=args.retries,
    )

    try:
        scraper.run()
    except Exception as e:
        logging.error(f"Scraper failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
