from datetime import date, timedelta

"""Register for eia.gov API key here: https://www.eia.gov/opendata/register.php
Find available suregions here: https://www.eia.gov/opendata/browser/electricity/rto/region-data
"""

# Default Data Settings
MAX_PAGE_SIZE = 5000
YEARS = 6
END_DATE = date.today() - timedelta(days=2)
START_DATE = date.today() - timedelta(days=(365*YEARS))
API_KEY = ""

# Default Region settings
LAT_DEFAULT = 32.71
LON_DEFAULT = -117.16
SUBBA_DEFAULT = "SDGE"
CITY_DEFAULT = "San Diego"
COUNTRY_DEFAULT = "America"
TIMEZONE_DEFAULT = 'America/Los_Angeles'

# Default Data Path for Modeling
DATA_PATH = f'./Data/demand_data_{SUBBA_DEFAULT}_{END_DATE}.csv'
EXO_PATH = f'./Data/exogenous_features.csv'
