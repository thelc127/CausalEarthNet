"""
Author: Rashila Lamichhane
Email: rashila.lamichhane01@gmail.com
Description: Script to download comprehensive monthly mean ERA5 reanalysis data from the Copernicus
    Climate Data Store (CDS). This step acquires the raw, global, multi-level time series
    data required for subsequent feature engineering and causal analysis.
"""
# data/download_era5.py

import cdsapi
import os

# 1. Initialize the CDS API client
c = cdsapi.Client()

# Define the years to download (e.g., 20 years for robust climatology)
years = [str(y) for y in range(2000, 2020)] # 20 full years

# Variables needed for specific regional proxies (e.g., ENSO SST, E. Africa rainfall)
surface_variables = [
    '2m_temperature',          # t_ENSO, t_Midlat (as low-level/SST proxy)
    'total_precipitation',     # tp_E_Africa (accumulated rainfall)
    'sea_surface_temperature', # Primary variable for calculating ENSO index (t_ENSO)
]

# Variables requiring a pressure level dimension (atmospheric dynamics)
pressure_variables = [
    'temperature',          # t (used for mid-atmosphere analysis, though 1000hPa is used here)
    'geopotential',         # z (Used for z500_Arctic for Polar Vortex/AO dynamics)
    'specific_humidity'     # q (Used for q850_E_Africa for moisture transport)
]

# Pressure levels needed to extract specific climate mechanisms
pressure_levels = [
    '1000', '925', '850', '700', '500' # Covers boundary layer (850hPa) to mid-troposphere (500hPa)
]
print("Starting download of required ERA5 monthly mean data (2000-2019)...")

# ----- API calls ----

# Request 1: Single-Level Variables (Surface Data)
# Downloads surface and accumulated variables (tp, sst)
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': surface_variables,
        'year': years,
        'month': [
            '01', '02', '03', '04', '05', '06',
            '07', '08', '09', '10', '11', '12',
        ],
        'time': '00:00',
        'format': 'netcdf',
    },
    'era5_surface_monthly.nc'
)

# Request 2: Pressure-Level Variables (Atmospheric Data)
# Downloads atmospheric dynamics variables (t, z, q) at specified levels
c.retrieve(
    'reanalysis-era5-pressure-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': pressure_variables,
        'pressure_level': pressure_levels,
        'year': years,
        'month': [
            '01', '02', '03', '04', '05', '06',
            '07', '08', '09', '10', '11', '12',
        ],
        'time': '00:00',
        'format': 'netcdf',
    },
    'era5_pressure_monthly.nc'
)

print("\nDownload finished. Files saved as 'era5_surface_monthly.nc' and 'era5_pressure_monthly.nc'.")
print("Next step: Run 'data/aggregate.py' to process these files.")