import numpy as np
import pandas as pd
import pickle as pickle
from datetime import datetime, timedelta, date

import requests

def ui_transformer(start_city, end_city, user_date, ice_name):
    """
    This function transforms user input into a DataFrame that's readable for the model
    to make the prediction.
    """

    # variables for model
    #['trip','mean_delay','weekday','sin_time','cos_time','sin_day','cos_day','public_holiday','covid_lockdown','
    # temp_max_combined', 'temp_min_combined','prcp_max_combined', 'snow_max_combined',
    # 'wspd_max_combined','wpgt_max_combined', 'coco_max_combined']

    start_city =  special_characters(start_city)
    end_city = special_characters(end_city)
    arrival_date = pd.to_datetime(user_date) ###KEEP user_date as STRING ! datetime object called arrival_date

    #1: trip
    trip = start_city + '-' + end_city

    #2: mean_delay
    mean_delay_df = pd.read_csv('api/data/mean_delay_ices.csv')
    mean_delay = mean_delay_df.mean_delay[mean_delay_df.zugnr == ice_name]

    #3: weekday
    weekday = arrival_date.day_name()

    #4:sin_time
    seconds_in_day = 24*60*60
    time = user_date.split(" ")[1]
    seconds = sum(x * int(t) for x, t in zip([3600, 60, 1], time.split(":")))
    sin_time = np.sin(2*np.pi*seconds/seconds_in_day)
    cos_time = np.cos(2*np.pi*seconds/seconds_in_day)

    #6:sin_day
    #7:cos_day
    days_in_year = 366
    day_of_year = int(arrival_date.strftime('%j'))
    sin_day = np.sin(2*np.pi*day_of_year/days_in_year)
    cos_day = np.cos(2*np.pi*day_of_year/days_in_year)

    #8:public_holiday
    public_holiday = 0

    #9:covid_lockdown
    covid_lockdown = 0

    #10:weather_forecast
    weather_df = weather_forecast(start_city, end_city, user_date)

    #11:build_final_dataframe
    X = pd.DataFrame({
        'trip': '-'.join([start_city, end_city]),
        'mean_delay': mean_delay,
        'weekday': weekday,
        'sin_time': sin_time,
        'cos_time': cos_time,
        'sin_day': sin_day,
        'cos_day': cos_day,
        'public_holiday': public_holiday,
        'covid_lockdown': covid_lockdown,
        'temp_max_combined': weather_df.loc[0, 'temp_max_combined'],
        'temp_min_combined': weather_df.loc[0, 'temp_min_combined'],
        'prcp_max_combined': weather_df.loc[0, 'prcp_max_combined'],
        'snow_max_combined': weather_df.loc[0, 'snow_max_combined'],
        'wspd_max_combined': weather_df.loc[0, 'wspd_max_combined'],
        'wpgt_max_combined': weather_df.loc[0, 'wpgt_max_combined'],
        'coco_max_combined': weather_df.loc[0, 'coco_max_combined'],
    })

    return X


def weather_forecast(start_city, end_city, user_date):
    # get weather forecast
    key = '7DYDYYY5GVYHQA52HXFQV5A5Y' #Add here your own key from https://www.visualcrossing.com

    start_date = pd.Timestamp(user_date).round('H')
    start_date_00 = pd.Timestamp(start_date.year, start_date.month, start_date.day)
    end_date = datetime.now() + timedelta(14)
    end_date_00 = pd.Timestamp(end_date.year, end_date.month, end_date.day) + timedelta(1) - timedelta(hours=1)

    start_date_str = datetime.strftime(start_date, '%Y-%m-%d')
    end_date_str = datetime.strftime(end_date, '%Y-%m-%d')

    stations_lat_lon = pd.read_csv('api/data/Deutsche_Bahn_Haltestellen.csv', usecols=['X', 'Y', 'NAME'])
    stations_name = ['Köln Hbf',
                'München Hbf',
                'Mannheim Hbf',
                'Stuttgart Hbf',
                'Würzburg Hbf',
                'Frankfurt(Main)Hbf',
                'Nürnberg Hbf',
                'Essen Hbf',
                'Hamburg Hbf',
                'Berlin Hbf',
                'Hannover Hbf',
                'Hagen Hbf',
                'Erfurt Hbf',
                'Göttingen'
                ]

    station_sc = [station for station in stations_name if start_city in station]
    station_ec = [station for station in stations_name if end_city in station]
    stations = station_sc + station_ec

    nodes = ['origin_city', 'destination_city']
    trip = {node: station for node, station in zip(nodes, stations)}
    hours = [6, 12]
    c_variables = ['temp', 'prcp', 'snow', 'wspd', 'wpgt', 'coco']

    coco_forecast = pd.read_csv('api/data/weather_coco_forecast.csv', sep=';')
    coco_forecast.set_index('Code', inplace=True)
    coco_forecast = coco_forecast.to_dict()['Weather Condition']

    new_classes_forecast = {
        1: [29, 42, 43],
        2: [2, 8, 9, 19, 20, 21, 24, 27, 28, 30, 31, 32, 33, 36, 38, 39, 40, 41],
        3: [1, 4, 6, 11, 12, 14, 18, 23, 26, 35, 37],
        4: [3, 5, 7, 10, 13, 15, 16, 17, 22, 25, 34]
    }

    #weather_response = {}
    weather = {}
    weather_dict = {}
    df = pd.DataFrame()
    for n, s in trip.items():
        lat = stations_lat_lon[stations_lat_lon['NAME'] == s]['Y'].mean()
        lon = stations_lat_lon[stations_lat_lon['NAME'] == s]['X'].mean()

        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{str(lat)},{str(lon)}/{start_date_str}/{end_date_str}"
        params = {'key': key,
                'unitGroup': 'metric'}
        response = requests.get(url, params=params).json()

        #weather_response[s] = response
        #response = weather_response[s] ### THIS CHANGED

        number_of_days = (end_date - start_date).days + 1
        variables = ['datetime', 'temp', 'precip', 'snow', 'windspeed', 'windgust', 'conditions']
        sw_f = pd.DataFrame(columns=variables)
        sw_f['datetime'] = pd.date_range(start_date_00, end_date_00, freq='H')
        sw_f.set_index('datetime', inplace=True)

        for d in range(number_of_days):
            for i, r in sw_f.iterrows():
                for v in variables[1:]:
                    sw_f.loc[i, v] = response['days'][d]['hours'][i.hour][v]

        sw_f.rename(columns={'precip': 'prcp',
                            'windspeed': 'wspd',
                            'windgust': 'wpgt',
                            'conditions': 'coco'}, inplace=True)

        sw_f.index.names = ['time']

        weather[s] = sw_f.copy()

        weather[s]['coco'] = weather[s]['coco'].apply(lambda x: coco_func(x, coco_forecast, new_classes_forecast))
        weather[s].reset_index(inplace=True)
        weather[s]['time']  = pd.to_datetime(weather[s]['time'])

        if n == 'origin_city':
            nn = 'oc'
        elif n == 'destination_city':
            nn = 'dc'
        for h in hours:
            weather_dict['time_' + nn + ' -' + str(h) + 'H'] = weather[s][weather[s]['time'] == start_date - timedelta(hours=h)].rename(
                columns={'temp': 'temp_' + nn + '_' + str(h),
                            'prcp': 'prcp_' + nn + '_' + str(h),
                            'snow': 'snow_' + nn + '_' + str(h),
                            'wspd': 'wspd_' + nn + '_' + str(h),
                            'wpgt': 'wpgt_' + nn + '_' + str(h),
                            'coco': 'coco_' + nn + '_' + str(h)})

    total_df = pd.DataFrame()
    for k, v in weather_dict.items():
        v.reset_index(inplace=True)
        total_df[list(v.columns)[1:]] = v[list(v.columns)[1:]]

    for v in c_variables:
        total_df[v + '_max_combined'] = total_df[[v + '_oc_6', v + '_oc_12', v + '_dc_6', v + '_dc_12']].max(axis=1)

    total_df['temp_min_combined'] = total_df[['temp_oc_6', 'temp_oc_12', 'temp_dc_6', 'temp_dc_12']].min(axis=1)

    return total_df

def special_characters(city):
    if city == 'Munchen':
        return 'München'
    elif city == 'Koln':
        return 'Köln'
    elif city == 'Wurzburg':
        return 'Würzburg'
    elif city == 'Nurnberg':
        return 'Nürnberg'
    elif city == 'Gottingen':
        return 'Göttingen'
    else:
        return city

def day_categories(day):
    time_of_day_cat = ['night', 'morning', 'afternoon', 'evening']
    if 0 < day.hour < 6:
        return time_of_day_cat[0]
    elif 6 < day.hour < 12:
        return time_of_day_cat[1]
    elif 12 < day.hour < 18:
        return time_of_day_cat[2]
    elif 18 < day.hour < 24:
        return time_of_day_cat[3]

def coco_func(x, coco_forecast, new_classes_forecast):
    x_list = x.split(', ')
    for k, v in coco_forecast.items():
        if v in x_list:
            for c, i in new_classes_forecast.items():
                if k in i:
                    return c
