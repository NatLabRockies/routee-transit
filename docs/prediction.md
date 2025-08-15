# RouteE-Transit Prediction Pipeline
RouteE-Transit builds on [RouteE-Powertrain](https://github.com/NREL/routee-powertrain) to predict the energy consumption of bus trips given a static GTFS feed. Its key role is to convert GTFS features (such as trip and shape data) into RouteE features (such as vehicle speed, road grade, and distance along road links), so that a RouteE-Powertrain model can be used to predict energy consumption. The full prediction pipeline is summarized by the following figure:

![Prediction Pipeline Overview](images/PredictionOverview.png)

RouteE-Transit moves from static GTFS files to energy predictions in three steps:

## 1) Specify GTFS Trip Data
First, users need to specify the scope of predictions by supplying data from a static GTFS feed. See [](data:gtfs-reqs) for details on what GTFS data must be available. RouteE-Transit needs a set of trips along with their shape traces, stop locations, and stop times in order to proce estimated distance, road grade, and speed for RouteE-Powertrain models.

Users can supply an entire GTFS feed as input or filter down trips (e.g., only trips on a certain day or serving a certain route).

## 2) Prepare RouteE-Powertrain Features
Next, RouteE-Transit transforms the input GTFS data into link-level features on the road network so a RouteE-Powertrain model can be applied. The first step in this process is to upsample the shape traces from `shapes.txt` to approximately 1 Hz resolution for better map matching accuracy, and then match the shapes to OpenStreetMap road links using NREL's `mappymatch` package.

The map-matched shapes are then used to calculate link distances and NREL's `gradeit` package appends road grade information to each link based on USGS National Map elevation data.

Finally, the estimated distances are used along with the time intervals between stops from`stop_times.txt` to estimate bus average speed along each road link.

## 3) Predict Energy Consumption with RouteE-Powertrain
In the last step, a trained RouteE-Powertrain model is run to predict energy consumption for each trip. RouteE-Powertrain version 1.3.2 introduced two initial transit bus models (`Transit_Bus_Diesel` and `Transit_Bus_Battery_Electric`) and additional models for different bus styles and manufacturers will be rolled out over time.

# Assumptions and Limitations
This initial version of RouteE-Transit has the following limitations:
* Weather impacts are not currently accounted for. Note that HVAC loads have a major impact on electric bus energy consumption, especially when temperatures are very low or very high.
* Deadhead trips (including pull-out and pull-in trips from/to the depot as well as deadhead in between service trips) are not currently modeled.