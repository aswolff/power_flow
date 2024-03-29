import requests
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
from astral.sun import sun
from astral import LocationInfo
import numpy as np
import os
from config import *

class DataProcessor:
    def __init__(self):
        self.electricity_data = self.get_electricity_data()
        self.weather_data = self.get_weather_data()
        self.combined_data = self.combine_data(self.electricity_data, self.weather_data)
        self.exog_features = self.get_exog_features()
        self.data_with_exog = self.combine_exog_data(self.combined_data, self.exog_features)
        #self.download_data(self.data_with_exog)

    def get_electricity_data(self, subba: str = SUBBA_DEFAULT) -> pd.DataFrame:
        offset = 0
        api_url = f"https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/?frequency=hourly&data[0]=value&facets[subba][]={subba}&sort[0][column]=period&sort[0][direction]=asc&api_key={API_KEY}"
        records = []
        
        while True:
            params = {
                'offset': offset,
                'length': MAX_PAGE_SIZE,
                'end': f'{START_DATE}T00',
                'end': f'{END_DATE}T00'
            }

            response = requests.get(api_url, params=params)

            if response.status_code == 200:
                json_response = response.json()
                new_records = json_response.get('response', {}).get('data', [])
                
                records.extend(new_records)
                
                # Check if we have received the last page
                if len(new_records) < MAX_PAGE_SIZE:
                    break
                
                # Increase offset to get the next page
                offset += MAX_PAGE_SIZE
            else:
                print(f"Failed to fetch data: {response.status_code}")
                break

        # Convert records to dataframe, sort index, and drop null values
        data = pd.DataFrame(records)
        data['period'] = pd.to_datetime(data['period'], format='%Y-%m-%dT%H')
        data['value'] = pd.to_numeric(data['value'])
        data = data[['period', 'value']]
        data.rename(columns = {'period': 'time'}, inplace=True)
        data = data.set_index('time')
        data = data.sort_index()
        data = data.asfreq('1h')
        data = data.dropna()
        return data
    
    
    def get_weather_data(self, lat: int = LAT_DEFAULT, lon: int = LON_DEFAULT) -> pd.DataFrame:
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        #Parameters
        url = "https://archive-api.open-meteo.com/v1/archive"
        params_weather = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "temperature_2m"
        }

        responses_weather = openmeteo.weather_api(url, params=params_weather)

        response_weather = responses_weather[0]

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response_weather.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s"),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}
        hourly_data["temperature_2m"] = hourly_temperature_2m

        hourly_dataframe = pd.DataFrame(data = hourly_data)
        hourly_dataframe.rename(columns = {'temperature_2m': 'temperature'}, inplace=True)
        hourly_dataframe = hourly_dataframe.set_index('date')
        hourly_dataframe = hourly_dataframe.sort_index()
        hourly_dataframe = hourly_dataframe.asfreq('1h')
        return hourly_dataframe
    
    def combine_data(self, energy_data: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
        data = energy_data.join(weather_data, how='left')
        data = data.sort_index()
        data = data.asfreq('1h', method='ffill')
        return data
    
    def download_data(self, data: pd.DataFrame) -> None:
        if not os.path.exists('./Data'):
            os.makedirs('./Data')

        data.to_csv(f'./Data/demand_data_{SUBBA_DEFAULT}_{END_DATE}.csv')

    def verify_complete_data(self, data: pd.DataFrame) -> bool:
        return data.isnull().any(axis=1).mean() == 0.0


    def get_exog_features(self) -> pd.DataFrame:
        """
        Creates exogenous variables such as calendar features, sunlight features, and temperature features
        Code taken and modified from: https://www.cienciadedatos.net/documentos/py29-forecasting-electricity-power-demand-python.html
        Returns
        -------
        exogenous_features : pd.DataFrame
            Dataframe with exogenous variables
        """
        calendar_features = pd.DataFrame(index=self.combined_data.index)
        calendar_features['month'] = calendar_features.index.month
        calendar_features['week_of_year'] = calendar_features.index.isocalendar().week
        calendar_features['week_day'] = calendar_features.index.day_of_week + 1
        calendar_features['hour_day'] = calendar_features.index.hour + 1

        location = LocationInfo(
            CITY_DEFAULT,
            COUNTRY_DEFAULT,
            latitude=LAT_DEFAULT,
            longitude=LON_DEFAULT,
            timezone=TIMEZONE_DEFAULT
        )
        # Prepare the sunlight features
        sunrise_hour = []
        sunset_hour = []
        for date in self.combined_data.index:
            try:
                sunrise = sun(location.observer, date, tzinfo=location.timezone)['sunrise'].hour
                sunset = sun(location.observer, date, tzinfo=location.timezone)['sunset'].hour
            except ValueError:
                # Handle days with no sunrise/sunset
                sunrise = 6  # Default or could use NaN or another placeholder
                sunset = 18  # Default or could use NaN or another placeholder
            sunrise_hour.append(sunrise)
            sunset_hour.append(sunset)
            
        sun_light_features = pd.DataFrame({
                                'sunrise_hour': sunrise_hour,
                                'sunset_hour': sunset_hour}, 
                                index = self.combined_data.index
                            )
        sun_light_features['daylight_hours'] = (
            sun_light_features['sunset_hour'] - sun_light_features['sunrise_hour']
        )
        sun_light_features['is_daylight'] = np.where(
                                                (self.combined_data.index.hour >= sun_light_features['sunrise_hour']) & \
                                                (self.combined_data.index.hour < sun_light_features['sunset_hour']),
                                                1,
                                                0
                                            )

        # Temperature features
        temp_features = self.combined_data[['temperature']].copy()
        temp_features['temp_roll_mean_1_day'] = temp_features['temperature'].rolling(window=96, min_periods=1, closed='left').mean()
        temp_features['temp_roll_mean_7_day'] = temp_features['temperature'].rolling(window=96*7, min_periods=1, closed='left').mean()
        temp_features['temp_roll_max_1_day'] = temp_features['temperature'].rolling(window=96, min_periods=1, closed='left').max()
        temp_features['temp_roll_min_1_day'] = temp_features['temperature'].rolling(window=96, min_periods=1, closed='left').min()
        temp_features['temp_roll_max_7_day'] = temp_features['temperature'].rolling(window=96*7, min_periods=1, closed='left').max()
        temp_features['temp_roll_min_7_day'] = temp_features['temperature'].rolling(window=96*7, min_periods=1, closed='left').min()

        # Merge all exogenous variables
        exogenous_features = pd.concat([
                                calendar_features,
                                sun_light_features,
                                temp_features
                            ], axis=1)
        
        # Create cyclical features
        month_encoded = self.cyclical_encoding(exogenous_features['month'], 12)
        week_of_year_encoded = self.cyclical_encoding(exogenous_features['week_of_year'], 52)
        week_day_encoded = self.cyclical_encoding(exogenous_features['week_day'], 7)
        hour_day_encoded = self.cyclical_encoding(exogenous_features['hour_day'], 24)
        sunrise_hour_encoded = self.cyclical_encoding(exogenous_features['sunrise_hour'], 24)
        sunset_hour_encoded = self.cyclical_encoding(exogenous_features['sunset_hour'], 24)

        cyclical_features = pd.concat([
                        month_encoded,
                        week_of_year_encoded,
                        week_day_encoded,
                        hour_day_encoded,
                        sunrise_hour_encoded,
                        sunset_hour_encoded
                    ], axis=1)
        
        exogenous_features = pd.concat([exogenous_features, cyclical_features], axis=1)

        exogenous_features = exogenous_features.dropna()
        exogenous_features.to_csv(f'./Data/exogenous_features.csv')
        return exogenous_features
    
    def cyclical_encoding(self, data: pd.Series, cycle_length: int) -> pd.DataFrame:
        """
        Encode a cyclical feature with two new features sine and cosine.
        The minimum value of the feature is assumed to be 0. The maximum value
        of the feature is passed as an argument.
        Code taken from: https://www.cienciadedatos.net/documentos/py29-forecasting-electricity-power-demand-python.html
        
        Parameters
        ----------
        data : pd.Series
            Series with the feature to encode.
        cycle_length : int
            The length of the cycle. For example, 12 for months, 24 for hours, etc.
            This value is used to calculate the angle of the sin and cos.

        Returns
        -------
        result : pd.DataFrame
            Dataframe with the two new features sin and cos.
        """

        sin = np.sin(2 * np.pi * data/cycle_length)
        cos = np.cos(2 * np.pi * data/cycle_length)
        result =  pd.DataFrame({
                    f"{data.name}_sin": sin,
                    f"{data.name}_cos": cos
                })

        return result
    
    def combine_exog_data(self, data: pd.DataFrame, exogenous_features: pd.DataFrame) -> pd.DataFrame:
        exog_features = []
        # Columns that ends with _sin or _cos are selected
        exog_features.extend(exogenous_features.filter(regex='_sin$|_cos$').columns.tolist())

        # Columns that start with temp_ are selected
        exog_features.extend(exogenous_features.filter(regex='^temp_.*').columns.tolist())

        # Include original features
        exog_features.extend(['temperature'])

        data = data[['value']].merge(
            exogenous_features[exog_features],
            left_index=True,
            right_index=True,
            how='left'
        )

        data = data.dropna()
        data = data.astype('float32')

        return data