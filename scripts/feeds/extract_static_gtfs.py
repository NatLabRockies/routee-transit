import io
import json
import os
import pathlib
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table


class GtfsExtractor:
    def __init__(self):
        load_dotenv()
        try:
            self.REFRESH_TOKEN = os.environ["MOBILITY_DATA_REFRESH_TOKEN"]
        except KeyError:
            raise ValueError(
                "Environment variable 'MOBILITY_DATA_REFRESH_TOKEN' is not set. "
                "If you do not have a Mobility Database refresh token, obtain one "
                "from https://mobilitydatabase.org/"
            )
        self.api_key = None

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
        api_key = json.loads(response.text)["access_token"]
        return api_key

    # Checks for 200 from Mobility DB
    def check_api_status(self, api_key: str) -> int:
        headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
        response = requests.get(
            "https://api.mobilitydatabase.org/v1/metadata", headers=headers
        )
        return response.status_code

    def query_mobility_db(self, query: str) -> List[Dict[str, Any]]:
        """Make a request to the Mobility Database API with the given query

        Args:
            query: The query string to append to the base URL

        Returns:
            List of dictionaries containing the API response data
        """
        if self.api_key is None:
            self.api_key = self.get_api_key()
        base_url = "https://api.mobilitydatabase.org/v1/gtfs_feeds"
        query_url = f"{base_url}{query}"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = self.make_api_request(query_url, headers)
        return json.loads(response.text)

    def get_gtfs_by_agency(self, agency: str, state: str, municipality: str) -> str:
        query = f"?&status=active&subdivision_name={state}&municipality={municipality}&q={agency}%"

        results = self.query_mobility_db(query)
        last_path = ""

        if len(results) > 1:
            print(
                f"{len(results)} agencies found matching {agency} in {municipality}, {state}. "
            )
            for i, result in enumerate(results):
                zip_url = results[i]["source_info"]["producer_url"]
                mdb_id = results[i]["id"]

                # Use consistent naming format with underscores for multiple results
                agency_id = f"{agency}_{mdb_id}"
                self.unzip_from_url(zip_url, agency_id, mdb_id)
                last_path = f"data/gtfs/{agency_id}"

        elif len(results) == 1:
            zip_url = results[0]["source_info"]["producer_url"]
            mdb_id = results[0]["id"]
            self.unzip_from_url(zip_url, agency, mdb_id)
            last_path = f"data/gtfs/{agency}"

        else:
            print(
                f"No matching transit agencies found for {agency}. Please add more information and try again."
            )

        return last_path

    # Displays a list of active GTFS datasets matching the input. User chooses which dataset to download by index number.
    def get_gtfs_query(self, user_query: str) -> str:
        api_query = f"?offset=0&status=active&subdivision_name={user_query}%"

        results = self.query_mobility_db(api_query)

        if len(results) > 1:
            tmp_list = []
            # We have to loop to keep the transit agency name 1:1 with the location
            for i, agency in enumerate(results):
                locations = results[i]["locations"][0]
                locations["transit provider"] = results[i]["provider"]
                locations["mdb_id"] = results[i]["id"]
                locations["zip files"] = results[i]["source_info"]["producer_url"]
                tmp_list.append(locations)
            results_df_rough = pd.DataFrame(
                tmp_list,
                index=pd.RangeIndex(start=1, stop=len(tmp_list) + 1, name="index"),
            )
            results_df = results_df_rough.drop(columns="country_code")
            results_df = results_df_rough.drop(columns="zip files")
            console = Console()
            title = f"GTFS Datasets matching '{user_query}'"
            table = Table(title)
            table.add_row(
                results_df.to_string(float_format=lambda _: "{:.4f}".format(_))
            )
            console.print(table)
            x = input(
                "Please select from above options by typing your selection's index number and pressing return: "
            )
            agency_name = results_df.loc[int(x), "transit provider"]
            print("Selection: ", agency_name)
            zip_url = str(results_df_rough.loc[int(x), "zip files"])
            mdb_id = results_df_rough.loc[int(x), "mdb_id"]
            self.unzip_from_url(zip_url, agency_name, mdb_id)
            unzipped_path = f"data/gtfs/{agency_name}"
            print(f"Dataset for {agency_name} downloaded at {unzipped_path}.")

        elif len(results) == 1:
            agency = results[0]["provider"]
            print("Getting data for ", agency)
            zip_url = results[0]["source_info"]["producer_url"]
            mdb_id = results[0]["id"]
            self.unzip_from_url(zip_url, agency, mdb_id)
            unzipped_path = f"data/gtfs/{agency}"
        else:
            print(
                "No matching transit agencies found. Please add more information and try again."
            )

        return unzipped_path

    failed_agencies: list[Any] = []

    def get_gtfs_by_state(self, state: str, api_key: str) -> str:
        query = f"?&status=active&subdivision_name={state}"
        try:
            results = self.query_mobility_db(query)

            if len(results) > 1:
                print(len(results), f"results for {state}")
                for i, result in enumerate(results):
                    zip_url = results[i]["source_info"]["producer_url"]
                    mdb_id = results[i]["id"]
                    agency = results[i]["provider"]
                    status = results[i]["status"]
                    if status == "active" and os.path.exists(
                        "data/gtfs/" + state + "/" + agency + "_" + mdb_id
                    ):
                        pass

                    elif status == "deprecated":
                        print(f"{agency} deprecated")
                        failed_agency_dict = {
                            "agency": agency + "_" + mdb_id,
                            "state": state,
                            "error code": status,
                        }
                        self.failed_agencies.append(failed_agency_dict)

                    else:
                        print(f"Unzipping for number {i}, {agency}")
                        self.unzip_from_url_state(zip_url, agency, mdb_id, state)
                        unzipped_path = f"data/gtfs/{state}/{agency}_{mdb_id}"

            elif len(results) == 1:
                print(f"One result for {state}")
                zip_url = results[0]["source_info"]["producer_url"]
                mdb_id = results[0]["id"]
                agency = results[0]["provider"]
                status = results[0]["status"]
                if status == "active" and os.path.exists(
                    "data/gtfs/" + state + "/" + agency + "_" + mdb_id
                ):
                    pass
                elif status == "deprecated":
                    print(f"{agency} deprecated")
                    failed_agency_dict = {
                        "agency": agency,
                        "state": state,
                        "error code": "deprecated",
                    }
                    self.failed_agencies.append(failed_agency_dict)
                else:
                    print(f"Unzipping for number {i}, {agency}")
                    unzipped_path = self.unzip_from_url_state(
                        zip_url, agency, mdb_id, state
                    )

            else:
                print(
                    f"No matching transit agencies found for {state}. Please add more information and try again."
                )

            return unzipped_path

        except requests.exceptions.HTTPError as e:
            raise Exception(f"API error from api call: {str(e)}")

    # For unzipping files based on zip URL given by Mobility DB
    def unzip_from_url(self, zip_url: str, agency_name: Any, mdb_id: Any) -> None:
        download_zip = requests.get(zip_url, timeout=15)
        download_zip.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(download_zip.content)) as zip_ref:
            agency_dir = os.path.join("GTFS_Data", agency_name)
            agency_with_id_dir = os.path.join("GTFS_Data", f"{agency_name}_{mdb_id}")

            # Check if already exists in routee-transit directory structure
            routee_agency_dir = os.path.join("routee-transit", "data/gtfs", agency_name)
            routee_agency_with_id_dir = os.path.join(
                "routee-transit", "data/gtfs", f"{agency_name}_{mdb_id}"
            )

            if os.path.exists(routee_agency_dir) and os.path.exists(
                routee_agency_with_id_dir
            ):
                print(f"{agency_name}_{mdb_id} already downloaded.")
            elif os.path.exists(routee_agency_dir):
                os.makedirs(agency_with_id_dir, exist_ok=True)
                zip_ref.extractall(agency_with_id_dir)
                print(f"Dataset extracted to {agency_with_id_dir}")
            else:
                os.makedirs(agency_dir, exist_ok=True)
                zip_ref.extractall(agency_dir)
                print(f"Dataset extracted to {agency_dir}")

    # For unzipping local files
    def unzip_file_local(self, gtfs_zip: str) -> None:
        zip_name = pathlib.PurePosixPath(gtfs_zip).stem
        output_dir = os.path.join("data/gtfs", zip_name)

        with zipfile.ZipFile(gtfs_zip, "r") as zip_ref:
            routee_output_dir = os.path.join("routee-transit", "data/gtfs", zip_name)
            if os.path.exists(routee_output_dir):
                duplicate_dir = os.path.join("data/gtfs", f"{zip_name}_duplicate")
                os.makedirs(duplicate_dir, exist_ok=True)
                zip_ref.extractall(duplicate_dir)
                print(f"File extracted at {duplicate_dir}")
            else:
                os.makedirs(output_dir, exist_ok=True)
                zip_ref.extractall(output_dir)
                print(f"File extracted at {output_dir}")

    def unzip_from_url_state(
        self, zip_url: str, agency_name: str, mdb_id: str, state: str
    ) -> Optional[str]:
        """Download and extract GTFS data for a specific agency in a state

        Args:
            zip_url: URL to download the GTFS zip file
            agency_name: Name of the transit agency
            mdb_id: Mobility Database ID
            state: State where the agency operates

        Returns:
            str: Path to the extracted data directory, or None if extraction failed
        """
        # Create the state directory if it doesn't exist
        state_dir = os.path.join("data/gtfs", state)
        os.makedirs(state_dir, exist_ok=True)

        # Path for this agency's data
        zip_path = os.path.join(state_dir, f"{agency_name}_{mdb_id}")

        # Skip if already downloaded
        if os.path.exists(zip_path):
            print(f"{agency_name}_{mdb_id} already exists in {state}")
            return zip_path

        try:
            # Download and extract
            download_zip = requests.get(zip_url, timeout=15)
            download_zip.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(download_zip.content)) as zip_ref:
                os.makedirs(zip_path, exist_ok=True)
                zip_ref.extractall(zip_path)
                print(f"Dataset extracted to {zip_path}")
            return zip_path

        except Exception as e:
            # Just log the error and add to failed agencies
            print(f"Error downloading/extracting {agency_name}: {str(e)}")
            error_code = "Unknown"
            if isinstance(e, requests.exceptions.HTTPError) and hasattr(e, "response"):
                error_code = e.response.status_code
            elif isinstance(e, requests.exceptions.ConnectionError):
                error_code = "Connection Error"
            elif isinstance(e, requests.exceptions.Timeout):
                error_code = "Timeout"
            elif isinstance(e, zipfile.BadZipFile):
                error_code = "Bad Zip File"

            self.failed_agencies.append(
                {
                    "agency": f"{agency_name}_{mdb_id}",
                    "state": state,
                    "error code": error_code,
                }
            )

            return None

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

    def run(self, api_key=None) -> None:
        while True:
            print(
                "Welcome to the GTFS data extractor. What are you looking for?\n "
                "1: Search for a GTFS dataset.\n "
                "2: Extract a specific GTFS dataset \n "
                "3: Extract a local GTFS dataset \n "
                "4: Extract GTFS datasets for all states"
            )
            x = input(
                "Please select from above options by typing your selection number and pressing return: "
            )

            if x == "1":
                state = input(
                    "Please type the full name of the state that the agency of interest is in and press return:"
                )
                self.get_gtfs_query(state)

            elif x == "2":
                agency = input("Transit agency name?:")
                state = input("What state is the agency in?:")
                city = input("What city is the agency in?:")
                print(f"Looking for {agency} in {city}, {state}...")
                self.get_gtfs_by_agency(agency, state, city)

            elif x == "3":
                epsilon = input(
                    "Please provide the full path to the dataset zipfile and press return:"
                )
                self.unzip_file_local(epsilon)

            elif x == "4":
                with open(Path("scripts/feeds/states.json"), "r") as f:
                    states = json.load(f)
                    states_list = list(states.values())
                    for i in states_list:
                        try:
                            print("State: ", i)
                            if api_key is None or self.check_api_status(api_key) != 200:
                                api_key = self.get_api_key()
                                print("Fetched new API key")
                            x = self.get_gtfs_by_state(i, api_key)
                            print(x)
                        except Exception as e:
                            print(f"Error on {i}: {str(e)}")

            exit = input(
                "To start over, press return. Otherwise, type 'exit' and press return:"
            )
            if exit.lower() in ("exit", "e"):
                return  # Exit the program


if __name__ == "__main__":
    extractor = GtfsExtractor()
    api_key = extractor.get_api_key()
    extractor.run(api_key)
