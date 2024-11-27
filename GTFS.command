#!/bin/bash
source utils/.env
python utils/_init_.py
pip install -r utils/requirements.txt
echo "Welcome to the GTFS module. Type 's' to search for datasets or 'l' to use a local file and press return: "
read input 
if [[ "$input" == "s" ]]; then
   echo "What would you like to search for? Please type the name of the state:"
   read query
   echo "Looking for $query GTFS datasets..."
   python GTFS_processing/GTFS_processing.py -s "$query"
elif [[ "$input" == "l" ]]; then
   echo "Please provide the complete path to your local GTFS dataset:"
   read path
   echo "Analyzing data from $path..."
   python GTFS_processing/GTFS_processing.py -f $path
else
   echo "User error. Please try again."
fi
