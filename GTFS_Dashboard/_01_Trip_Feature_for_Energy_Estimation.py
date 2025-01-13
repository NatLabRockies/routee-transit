## Authors: Zhaocai Liu, Phoebe Ho
## Purpose: This notebook is used to process GTFS data for the RouteE-BEAT Tool (Get Energy Consumption Estimation for Each Trip based on Mapmatching and RouteE Prediction)

##------------------------------------------------------------------------##
## Create dataframe with trips and corresponding shapes (get GPS traces)
##------------------------------------------------------------------------##


import pandas as pd
import datetime
from geopy.distance import great_circle
import geopandas as gpd

# select city for analysis
city = 'richmond'#'RTD' #'saltlake'

# load data
df_shape = pd.read_csv(f'./GTFS_Data/{city}/shapes.txt', sep=',', header=0)
df_route = pd.read_csv(f'./GTFS_Data/{city}/routes.txt', sep=',', header=0)
df_trips = pd.read_csv(f'./GTFS_Data/{city}/trips.txt', sep=',', header=0)
df_stops = pd.read_csv(f'./GTFS_Data/{city}/stops.txt', sep=',', header=0)
df_stops_times = pd.read_csv(f'./GTFS_Data/{city}/stop_times.txt', sep=',', header=0)

trip_route = pd.merge(df_trips, df_route, how='left', on='route_id')
trip_route = trip_route[trip_route['route_type'] == 3]  # select only bus services (route_type = 3)

### Only consider bus shapes
df_shape = df_shape[df_shape.shape_id.isin(trip_route.shape_id.unique())]
df_shape = df_shape.sort_values(by = ['shape_id','shape_pt_sequence'])
df_shape_list = [group for _,group in df_shape.groupby('shape_id')]


#####---------------------#####
## Define a function to upsample shape GPS to 1s trace and get timestamp for each point
#####---------------------#####
import datetime
def upsample_shape(df_tmp):
    """
    This function is used to upsample a shape file to 1s (8m)
    input: df_tmp for one individual shape
    output: upsampled shape 
    """
    # Shift latitude and longitude to get previous point
    df_tmp['prev_latitude'] = df_tmp['shape_pt_lat'].shift()
    df_tmp['prev_longitude'] = df_tmp['shape_pt_lon'].shift()
    
    # Calculate the distance between consecutive points using great_circle
    df_tmp['distance_km'] = df_tmp.apply(lambda row: great_circle(
        (row['prev_latitude'], row['prev_longitude']),  # Previous point
        (row['shape_pt_lat'], row['shape_pt_lon'])  # Current point
    ).kilometers if pd.notnull(row['prev_latitude']) else 0, axis=1)
    
    # Calculate total distance
    total_distance_km = df_tmp['distance_km'].sum()

    ## Define a random date 
    date_tmp = datetime.datetime(2023, 9, 3)  

    ## Use calculated total distance instead of shape_dist_traveled because sometimes it is NAN
    df_tmp['shape_dist_traveled'] = df_tmp['distance_km'].cumsum()
    
    ## Speed is assumed to be 30 km/h, which is about 10 (8.33) m per second/node
    df_tmp['segment_duration_delta'] = df_tmp['shape_dist_traveled']/df_tmp['shape_dist_traveled'].max() * datetime.timedelta(seconds=round(total_distance_km / 30 * 3600))
    df_tmp['segment_duration_delta'] =  df_tmp['segment_duration_delta'].apply(lambda x: datetime.timedelta(seconds=round(x.total_seconds())))
    df_tmp['timestamp'] = datetime.timedelta(seconds=0) + df_tmp['segment_duration_delta'] + date_tmp
    
    ## Upsample to 1s 
    shape_id_tmp = df_tmp.shape_id.iloc[0]
    df_tmp = df_tmp[['shape_pt_lat','shape_pt_lon','timestamp','shape_dist_traveled']].drop_duplicates(subset = ['timestamp']).set_index('timestamp').resample('1s').interpolate(method='linear')
    
    ## Now we have the 1 HZ gps trace for each trip with timestamp
    
    df_tmp = df_tmp.reset_index(drop = True)
    
    df_tmp['shape_id'] = shape_id_tmp

    return df_tmp

#####---------------------#####
### Now mapmatching 
#####---------------------#####
import sqlalchemy as sql
from mappymatch.constructs.coordinate import Coordinate
from mappymatch.constructs.geofence import Geofence
from mappymatch.constructs.trace import Trace
from mappymatch.matchers.lcss.lcss import LCSSMatcher
from nrel.mappymatch.readers.tomtom import read_tomtom_nxmap_from_sql
from nrel.mappymatch.readers.tomtom_config import TomTomConfig 

### Engine for mapmatching
user = "zliu2"
password = "NRELisgr8!"
engine = sql.create_engine(
    f"postgresql://{user}:{password}@trolley.nrel.gov:5432/master"
)

### This is the usgs_tiles file I downloaded
raster_path = '/kfs2/projects/mbap/Zhaocai/Conda_Pack/gradeit/scripts/usgs_tiles' #'/projects/mbap/data/NED_13'

KM_TO_METERS = 1000
FT_TO_METERS = 0.3048
FT_TO_MILES = 0.000189394

### MapMatching should be conducted for each individual shapes to avoid duplicate mapmatching for each trip
def get_matches(df_tmp):
    df_tmp = df_tmp.drop_duplicates(
      subset = ['shape_pt_lat', 'shape_pt_lon'],
      keep = 'first').reset_index(drop = True)
    trace = Trace.from_dataframe(df_tmp, lat_column ='shape_pt_lat', lon_column='shape_pt_lon')
    geofence = Geofence.from_trace(trace, padding=1e3)
    config = TomTomConfig(include_display_class=True,include_direction = True)
    nxmap = read_tomtom_nxmap_from_sql(engine, geofence, tomtom_config=config)
    matcher = LCSSMatcher(nxmap)
    matches = matcher.match_trace(trace).matches_to_dataframe()
    df_result = pd.concat([df_tmp, matches], axis=1)
    return df_result

#####-------------
## Define a function to upsample shape GPS to 1s trace and get timestamp for each point
def attach_timestamp(df_tmp):
    ## Calculate the travel time
    df_tmp['segment_duration_delta'] = df_tmp['shape_dist_traveled']/df_tmp['shape_dist_traveled'].max() * (df_tmp['d_time'] - df_tmp['o_time'])
    df_tmp['segment_duration_delta'] =  df_tmp['segment_duration_delta'].apply(lambda x: datetime.timedelta(seconds=round(x.total_seconds())))
    df_tmp['timestamp'] = df_tmp['o_time'] + df_tmp['segment_duration_delta']
    
    ## get hour and minute of gps timestamp
    df_tmp["Datetime_nearest5"] = df_tmp['timestamp'].dt.round("5min")
    df_tmp['hour'] = df_tmp['Datetime_nearest5'].dt.components['hours']  
    df_tmp['minute'] = df_tmp['Datetime_nearest5'].dt.components['minutes'] 

    return df_tmp


#######----------------------
## For each GPS trace, get the corresponding link speed, grade, and traffic signal info
# Define functions

def add_speed_profile(df_match):
    # get profile ids for road links
    road_key_list = df_match.road_key.unique().tolist()
    speed_profile_query = f"""
    select * from network_w_speed_profiles
    where netw_id in {tuple(road_key_list)}  
    """
    speed_profiles = pd.read_sql(speed_profile_query, con=engine)

    speed_profiles = speed_profiles.dropna(subset = ['link_direction','monday_profile_id', 'tuesday_profile_id','wednesday_profile_id',
                        'thursday_profile_id', 'friday_profile_id',
                        'saturday_profile_id', 'sunday_profile_id'])
    speed_profiles['link_direction'] = speed_profiles['link_direction'].astype('int64')
    speed_profiles['netw_id'] = speed_profiles['netw_id'].astype('str')
    df_merge = pd.merge(df_match, speed_profiles, how='left', 
                        left_on=['road_key','direction'], 
                        right_on=['netw_id','link_direction'])
    return df_merge


def get_stacked_profiles(df):
    stacked_profiles = pd.melt(df[['index','monday_profile_id', 'tuesday_profile_id','wednesday_profile_id',
                    'thursday_profile_id', 'friday_profile_id',
                    'saturday_profile_id', 'sunday_profile_id',]],
    id_vars=['index']).sort_values('index').reset_index(drop=True)
    stacked_profiles.columns = ['index','day_of_week','profile_id']
    stacked_profiles['day_of_week'] = stacked_profiles['day_of_week'].str.extract('(.+)_profile_id', expand=True)
    return stacked_profiles


def extract_traffic_signals(junction_id_list):
    # get traffic signals for specified road links
    signal_query = f"""
    SELECT *
    FROM tomtom_multinet_current.MNR_Traffic_Light nt2tl
    WHERE nt2tl.JUNCTION_ID IN {tuple(junction_id_list)}
    """
    signals = pd.read_sql(signal_query, con=engine)
    signals_agg = signals.groupby(['junction_id'])['feat_id'].count().reset_index()
    signals_agg = signals_agg.rename(columns={"feat_id": "traffic_signal",})
    
    # join signal count to main df
    signals_agg['junction_id'] = signals_agg['junction_id'].astype('str')  # Convert from UUID   
    return signals_agg

def add_signal_features_to_dataframe(df):
    junction_id_list = pd.concat([df.junction_id_from, df.junction_id_to], 
                                 ignore_index=True).dropna()
    junction_id_list = junction_id_list.unique().tolist()
    signals = extract_traffic_signals(junction_id_list)
    signals.junction_id = signals.junction_id.astype('str')
    
    # merge with main dataframe 
    df.junction_id_from = df.junction_id_from.astype('str')
    df.junction_id_to = df.junction_id_to.astype('str')
    df = pd.merge(df, signals, how='left', left_on='junction_id_from', right_on='junction_id')
    df = pd.merge(df, signals, how='left', left_on='junction_id_to', right_on='junction_id')
    df = df.rename(columns={"traffic_signal_x": "traffic_signal_from",
                                          "traffic_signal_y": "traffic_signal_to",})
    df[['traffic_signal_from', 'traffic_signal_to']] = df[['traffic_signal_from', 'traffic_signal_to']].fillna(value=0)  # fill NA with 0 (0 signals on link)
    df = df.drop(columns=['junction_id_x','junction_id_y'])
    df['signal_count'] = df.traffic_signal_from + df.traffic_signal_to
    df['signal_binary'] = np.where((df['signal_count'] > 0), 1, 0)
    return df

#######----------------------

def process_gtfs(df):

    ### We might further put "add_speed_profile" and "add_signal_features_to_dataframe" into a seperate function because they are limited by the server parallel user limit.
    
    try:
        # add grade
        df_result = gradeit(df=df, lat_col='shape_pt_lat', lon_col='shape_pt_lon',
                               filtering=True, source='usgs-local',usgs_db_path=raster_path)
            
        # join speed and grade information
        df_result['miles_new'] = df_result['distances_ft'] * FT_TO_MILES  # column necessary for RouteE .predict() method 'distance_ft_filtered'
        df_result['gpsspeed_new'] = df_result['kilometers'] / df_result['minutes']*60*0.621371  # column necessary for RouteE .predict() method
        df_result['grade_new'] = df_result['grade_dec_filtered']  # GRADE
        
        df_result = add_speed_profile(df_result)
        points_no_profile = df_result[df_result['netw_id'].isnull()].shape[0]
        #print(f'no speed profile available for {points_no_profile} GPS points (out of {len(df_result)})')
        
        df_result['index'] = df_result.index.values
        stacked_profiles = get_stacked_profiles(df_result) # create a df with index and the day of week profiles stacked (cols to rows)
        df_result = pd.merge(df_result, stacked_profiles, how='right', on='index')
        
        # get relative speed % by profile id and hour
        df_result['profile_id'] = df_result['profile_id'].fillna(0)
        df_result['profile_id'] = df_result['profile_id'].astype('str')  # Convert from UUID    
        df_result = pd.merge(df_result, profile_ids, how='left',
                            on=['profile_id','hour','minute'])
        
        # calculate speed for timestamp
        df_result['new_speed_kph'] = df_result['free_flow_speed'] * (df_result['relative_speed']/1000)
        df_result['new_speed_mph'] = df_result['new_speed_kph'] * 0.6213711922  # kph to mph 
        df_result['new_speed_mph'] = df_result['new_speed_mph'].fillna(df_result.dropna(subset = ['new_speed_mph'])['new_speed_mph'].median())
        df_result = add_signal_features_to_dataframe(df_result)
    
        ## Fill missing time-dependent speed values with the non-time-dependent values
        df_result['new_speed_mph'] = df_result['new_speed_mph'].fillna(df_result['gpsspeed_new'])
        
        df_result = df_result[['trip_id',
               'road_key',
               'miles_new', 'grade_new', 
               'day_of_week','new_speed_mph',
               'signal_count', 
                'with_stop',
                'hour'
                ]]
    
        ## Link level aggregation for the purpose of simplicity and saving space
        df_result_agg = df_result.groupby(['trip_id','day_of_week','road_key']).agg({'miles_new':"sum",'grade_new':'mean','new_speed_mph':'mean','signal_count':"max",'with_stop':'max','hour':'mean'})
        df_result_agg = df_result_agg.reset_index()
        
        return df_result_agg
    except: ## Deal with bug/outliers
        trip_id_tmp = df.trip_id.iloc[0]
        df_result_agg = pd.DataFrame({'trip_id':[trip_id_tmp]})
        return df_result_agg

#####-------------
## Define a function to add stops to each trip
def attach_stop(df_tmp):
    """
    Attache stop to nearest GPS points based on spatial join
    """
    trip_id_tmp = df_tmp['trip_id'].iloc[0]
    df_stop_tmp = df_stops_trip[df_stops_trip.trip_id == trip_id_tmp]
    
    gdf_tmp = gpd.GeoDataFrame(df_tmp, geometry=gpd.points_from_xy(df_tmp.shape_pt_lon, df_tmp.shape_pt_lat))
    gdf_stop_tmp = gpd.GeoDataFrame(df_stop_tmp, geometry=gpd.points_from_xy(df_stop_tmp.stop_lon, df_stop_tmp.stop_lat))
    gdf_stop_tmp = gdf_stop_tmp.sjoin_nearest(gdf_tmp[['geometry','coordinate_id']])
    gdf_tmp['with_stop'] = 0
    gdf_tmp.loc[gdf_tmp.coordinate_id.isin(gdf_stop_tmp.coordinate_id.to_list()),'with_stop'] = 1 

    df_tmp = gdf_tmp.drop(['geometry'],axis = 1)
    return df_tmp


### Define a dict to track the time consumption for each step
dict_time = {}

##---------------------------------------------------------------------------------------------------------------------
## Main function
if __name__ == '__main__': 

    ### Conduct upsampling and mapmatching using multiprocessing
    # use multiprocessing to process all the files
    import multiprocessing as mp
    import time

    t1 = time.time()
    
    ## Upsampling
    pool = mp.Pool(mp.cpu_count())
    output = pool.map(upsample_shape, df_shape_list)
    pool.close()

    t2 = time.time()
    dict_time['upsample_shape'] = [t2 - t1]
    
    ### Map matching
    pool = mp.Pool(15)
    df_match_list = pool.map(get_matches, output)
    pool.close()
    output_df = pd.concat(df_match_list)
    
    t3 = time.time()
    dict_time['get_matches'] = [t3 - t2]

    #####---------------------#####
    # Now mergeg trip with shape
    #####---------------------#####
    df_stops_times_group = df_stops_times.groupby('trip_id').agg({'arrival_time':'first', 'departure_time':'last',\
                                                                   'stop_id':['first','last']}).reset_index()    
    df_stops_times_group.columns = ['trip_id', 'o_time','d_time','o_stop_id','d_stop_id'] 
    
    ### Merge trip_route with df_stops_times_group 
    trip_route = pd.merge(trip_route, df_stops_times_group, how='left', on='trip_id')
    trip_route['o_time'] = pd.to_timedelta(trip_route['o_time'])
    trip_route['d_time'] = pd.to_timedelta(trip_route['d_time'])   

    trip_route['trip_duration'] = trip_route['d_time'] - trip_route['o_time']
    # merge with shapes to get GPS traces for each trip
    print(trip_route.columns)
    print(output_df.columns)
    trip_shape = pd.merge(trip_route[['trip_id','shape_id','o_time','d_time']], output_df, how='left', on='shape_id')    
    trip_shape = trip_shape.sort_values(by=['trip_id', 'shape_dist_traveled']).reset_index(drop=True)
    ### Merge stop times with stops to obtain the stops for each trip
    df_stops_trip = df_stops_times[['trip_id','stop_sequence','stop_id']].merge(df_stops[['stop_id','stop_lat','stop_lon']],on='stop_id')

    # calculate approximate timestamps for each GPS trace
    list_trip_shape = [item for _,item in trip_shape.groupby('trip_id')] 

    ### Attache stop to points (spatil join nearest)
    pool = mp.Pool(mp.cpu_count())
    list_trip_shape = pool.map(attach_stop, list_trip_shape)
    pool.close()
    
    # Attach timestamp
    pool = mp.Pool(mp.cpu_count())
    output_trip = pool.map(attach_timestamp, list_trip_shape)
    pool.close()    
    
    #####---------------------#####
    # Now we do gradeit and attaching time-dependent speed/time information
    #####---------------------#####
    import pandas as pd
    import geopandas as gpd
    import time
    import os 
    import numpy as np
    
    from gradeit.gradeit import gradeit
    from shapely.geometry import Polygon, Point 

    # Pre-process the speed profiles
    profile_dir = '/projects/mbap/data/amazon-eco/us_network/profile_id_mapping.csv'
    profile_ids = pd.read_csv(profile_dir)
    profile_ids['datetime'] = pd.to_datetime(profile_ids['time_slot'], unit='s')
    profile_ids['hour'] = profile_ids.datetime.dt.hour
    profile_ids['minute'] = profile_ids.datetime.dt.minute
    profile_ids['profile_id'] = profile_ids['profile_id'].astype('str')  # Convert from UUID   


    # use multiprocessing to process all the files
    pool = mp.Pool(15)
    output = pool.map(process_gtfs, output_trip)
    pool.close()
    output_df = pd.concat(output)
    output_df.to_csv('./GTFS_Extracted_Features/gtfs_features_agg_{city}.csv', index=False)
    t4 = time.time()
    dict_time['process_gtfs'] = [t4 - t3]

    df_time = pd.DataFrame(dict_time)
    df_time.to_csv('./GTFS_Feature_Extraction_Time_Log/time_track_{}.csv'.format(city),index = False)
