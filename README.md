# routee-transit
Application of RouteE Powertrain to transit bus energy prediction using GFTS data as inputs.

This repo has two notebooks: _00_GTFS.ipynb and _01_Route_Predict.ipynb
" _00_GTFS.ipynb" processes GTFS data. We have two ways for speed information: (1) using the stop information from GTFS, which specifies the arrival time and accumulative driving distance for each stop. (2) using the mapmatching package to map the GTFS shapefile to openstreetmap and then extract speed information from openstreetmap (we might only have free flow speed).

"_01_Route_Predict.ipynb" use a pre-trained RouteE model to predict energy consumption for a trip from GTFS. Currently we assume 0 road grade and only use the driving speed information from GTFS.


# Packages needed

## Map Matching Package "mappymatch": https://mappymatch.readthedocs.io/en/latest/

pip install mappymatch

## Vehicle Energy Consumption Prediction Package "nrel.routee.powertrain": https://nrel.github.io/routee-powertrain/intro.html

pip install nrel.routee.powertrain

## Geopandas: https://geopandas.org/en/stable/getting_started/install.html

conda install geopandas

## Pandas

pip install pandas

## matplotlib for figure plotting

pip install matplotlib