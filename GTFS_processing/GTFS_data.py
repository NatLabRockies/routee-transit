import os
import requests
import json
import zipfile
import pathlib
import io
import pandas as pd
from rich.console import Console
from rich.table import Table

# States dictionary is used to convert between two letter and full state name. 
# This could be saved as a json file and opened instead of being placed directly in the file. 
states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}
REFRESH_TOKEN = os.environ.get("REFRESH_TOKEN")

# Use the refresh token to get a new API key
def get_api_key():
    headers = {'Content-Type': 'application/json'}
    json_data = {'refresh_token': REFRESH_TOKEN}
    response = requests.post('https://api.mobilitydatabase.org/v1/tokens', headers=headers, json=json_data)
    print("Refreshing api key...  ", response.status_code)
    api_key = json.loads(response.text)["access_token"]
    return api_key

# Test a small API query.
def GTFS_testquery(query, api_key):
    url = f'https://api.mobilitydatabase.org/v1/gtfs_feeds?&status=active&subdivision_name={query}%'
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}"
        }
    try:
            res = requests.get(url, headers=headers)
            res.raise_for_status()    
            results = json.loads(res.text)

    except requests.exceptions.HTTPError as e:
        raise Exception(f"API error from api call: {str(e)}")
    
    return len(results)
    
# Loop extract data for multiple agencies at the same time based on search results
failed_agencies = []
def getGTFS_byagency(agency, state, municipality, api_key):
    url = f'https://api.mobilitydatabase.org/v1/gtfs_feeds?&status=active&subdivision_name={state}&municipality={municipality}&q={agency}%'
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}"
        }
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()    
        results = json.loads(res.text)
       
        if len(results) > 1: 
            for i, state in enumerate(results):
                zip_url = (results[i]["source_info"]["producer_url"])
                mdb_id = results[i]["id"]
                unzip_from_url(zip_url, agency, mdb_id)
                unzipped_path = f"GTFS_Data/{agency}+{mdb_id}"
        
        elif len(results) == 1:
            zip_url = (results[0]["source_info"]["producer_url"])
            mdb_id = results[0]["id"]
            unzip_from_url(zip_url, agency, mdb_id)
            unzipped_path = f"GTFS_Data/{agency}"

        else:
            print(f"No matching transit agencies found for {agency}. Please add more information and try again.")
            failed_agencies.append(agency)


        return unzipped_path
    
    except requests.exceptions.HTTPError as e:
        raise Exception(f"API error from api call: {str(e)}")
    

# Displays a list of active GTFS datasets matching the input. User chooses which dataset to download by index number. 
def getGTFS(query):
    api_key = get_api_key()
    url = f'https://api.mobilitydatabase.org/v1/gtfs_feeds?offset=0&status=active&subdivision_name={query}%'
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}"
        }
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()    
        results = json.loads(res.text)
        if len(results) > 1: 
            tmp_list = []
            #We have to loop to keep the transit agency name 1:1 with the location
            for i, agency in enumerate(results):
                locations = results[i]['locations'][0]
                locations["transit provider"] = results[i]["provider"]
                locations["zip files"] = (results[i]["source_info"]["producer_url"])
                tmp_list.append(locations)
            results_df_rough = pd.DataFrame(tmp_list, index=pd.RangeIndex(start=1, stop=len(tmp_list)+1, name='index'))
            results_df = results_df_rough.drop(columns='country_code')
            results_df = results_df_rough.drop(columns='zip files')
            console = Console()
            title = f"GTFS Datasets matching '{query}'"
            table = Table(title)
            table.add_row(results_df.to_string(float_format=lambda _: '{:.4f}'.format(_)))
            console.print(table)
            x = input("Please select from above options by typing your selection's index number and pressing return: ")
            agency_name = results_df.loc[int(x), 'transit provider']
            print("Selection: ", agency_name )
            zip_url = results_df_rough.loc[int(x), 'zip files']
            unzip_from_url(zip_url, agency_name )
            unzipped_path = f"GTFS_Data/{agency_name}"
        elif len(results) == 1:
            print("Getting data for [name of transit agency]")
            zip_url = (results[0]["source_info"]["producer_url"])
            mdb_id = results[0]["id"]
            unzip_from_url(zip_url, agency, mdb_id)
            unzipped_path = f"GTFS_Data/{agency}"
        else:
            print("No matching transit agencies found. Please add more information and try again.")

        return unzipped_path
    
    except requests.exceptions.HTTPError as e:
        raise Exception(f"API error from api call: {str(e)}")

# For unzipping files based on zip URL given by Mobility DB
def unzip_from_url(zip_url, agency_name, mdb_id):
    download_zip = requests.get(zip_url)
    download_zip.raise_for_status()
    try:
        with zipfile.ZipFile(io.BytesIO(download_zip.content)) as zip_ref:
            if os.path.exists("routee-transit/GTFS_Data/" + agency_name) and os.path.exists("routee-transit/GTFS_Data/" + agency_name + "_" + mdb_id):
                print(f"{agency_name} + '_' + {mdb_id} already downloaded.")
            elif os.path.exists("routee-transit/GTFS_Data/" + agency_name):
                zip_ref.extractall("GTFS_Data/" + agency_name + "_" + mdb_id) 
            else:
                zip_ref.extractall("GTFS_Data/" + agency_name)
            zip_ref.extractall("GTFS_Data/" + agency_name)
    except:
        raise Exception("File not found.")

# For unzipping local files
def unzip_file(GTFS_zip):
    try:
        zip_name = pathlib.PurePosixPath(GTFS_zip).stem
        with zipfile.ZipFile(GTFS_zip,"r") as zip_ref:
            if os.path.exists("routee-transit/GTFS_Data" + zip_name):
                zip_ref.extractall("GTFS_Data/" + zip_name + "_duplicate") 
            else:
                zip_ref.extractall("GTFS_Data/" + zip_name)
    except:
        raise Exception("File not found.")

#Checks for 200 from Mobility DB    
def check_api_status(api_key):
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}'}
    response = requests.get('https://api.mobilitydatabase.org/v1/metadata', headers=headers)
    return response.status_code

# Main function used to locally mass extract data. Will either turn this into a separate function or get rid of it altogether 
if __name__=='__main__':
    df = pd.read_csv('routee-transit/GTFS_processing/2022_NTD_Annual_Data_Vehicles_Type_Count_by_Agency_2023.csv', sep=',', header=0)
    df['State'] = df['State'].replace(to_replace=states)
    df = df[df.Bus != 0]
    states_list = df['State']
    agencies_list = df['Agency']
    municipality_list = df['City']

    api_key = get_api_key()
    for i, agency in enumerate(agencies_list):
        try:
            if check_api_status(api_key) == 200:
                getGTFS_byagency(agency, states_list[i], municipality_list[i], api_key)

            else:
                api_key = get_api_key()
                getGTFS_byagency(states_list[i], api_key)
        except:
            print(f"error on {agency}")
    print("Failed searches: ", len(failed_agencies), failed_agencies)
