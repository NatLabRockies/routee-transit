import json
import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv


class GtfsExtractor:
    def __init__(self) -> None:
        load_dotenv()
        try:
            self.REFRESH_TOKEN = os.environ["MOBILITY_DATA_REFRESH_TOKEN"]
        except KeyError:
            raise ValueError(
                "Environment variable 'MOBILITY_DATA_REFRESH_TOKEN' is not set. "
                "If you do not have a Mobility Database refresh token, obtain one "
                "from https://mobilitydatabase.org/"
            )
        self.api_key: str | None = None

    # Use the refresh token to get a new API key
    def get_api_key(self) -> str:
        """Get a fresh API key using the refresh token

        Returns:
            str: A valid API access token
        """
        headers = {"Content-Type": "application/json"}
        json_data = {"refresh_token": self.REFRESH_TOKEN}

        response = requests.post(
            "https://api.mobilitydatabase.org/v1/tokens",
            headers=headers,
            json=json_data,
            timeout=15,
        )
        response.raise_for_status()
        print("Refreshing api key...  ", response.status_code)
        api_key = str(json.loads(response.text)["access_token"])
        return api_key

    # Checks for 200 from Mobility DB
    def check_api_status(self, api_key: str) -> int:
        headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
        response = requests.get(
            "https://api.mobilitydatabase.org/v1/metadata", headers=headers
        )
        return response.status_code

    def query_mobility_db(
        self, path: str, query: str
    ) -> List[Dict[str, Any]] | Dict[str, Any]:
        """Make a request to the Mobility Database API with the given query

        Args:
            path: The API path, e.g. "gtfs_feeds" or "datasets/gtfs"
            query: The query string to append to the base URL

        Returns:
            List of dictionaries containing the API response data
        """
        if self.api_key is None:
            self.api_key = self.get_api_key()
        base_url = f"https://api.mobilitydatabase.org/v1/{path}"
        query_url = f"{base_url}{query}"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = self.make_api_request(query_url, headers)
        result: List[Dict[str, Any]] | Dict[str, Any] = json.loads(response.text)
        return result

    def query_mdb_feeds(self, query: str = "") -> List[Dict[str, Any]]:
        """Query Mobility Database feeds endpoint for feed information"""
        result = self.query_mobility_db(
            path="gtfs_feeds",
            query=query,
        )
        if not isinstance(result, list):
            raise TypeError(f"Expected list from feeds endpoint, got {type(result)}")
        return result

    def query_mdb_dataset(self, dataset_id: str, query: str = "") -> Dict[str, Any]:
        "Query Mobility Database datasets endpoint for dataset details"
        result = self.query_mobility_db(path=f"datasets/gtfs/{dataset_id}", query=query)
        if not isinstance(result, dict):
            raise TypeError(f"Expected dict from dataset endpoint, got {type(result)}")
        return result

    def make_api_request(
        self, url: str, headers: Dict[str, str], timeout: int = 15
    ) -> requests.Response:
        """Make an API request

        Args:
            url: The API endpoint URL
            headers: Request headers including authentication
            timeout: Request timeout in seconds

        Returns:
            Response object from successful request
        """
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response
