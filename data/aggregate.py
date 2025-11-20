"""
Author: Rashila Lamichhane
Email: rashila.lamichhane01@gmail.com
Description: Processes raw, global, monthly ERA5 NetCDF files by performing vertical level selection,
    spatial averaging over predefined regional boxes, and converting the output into a single,
    univariate time series CSV file.
"""
# data/aggregate_regional_data.py
# Step 2: This code converts hourly data into meaningful climate timescales:

import xarray as xr
import pandas as pd
import zipfile
import os
import numpy as np

# --- 1. Load and Merge Data ---
# 1.1 Load Pressure Level Data (e.g., z, q, t at 500hPa, 850hPa, etc.)
try:
    ds1 = xr.open_dataset('era5_pressure_monthly.nc', engine='netcdf4')
except FileNotFoundError:
    print("Error: 'era5_pressure_monthly.nc' not found. Run download_era5.py first.")
    exit()

# 1.2 Load and Extract Surface Level Data (e.g., tp)
try:
    extracted_dir = 'extracted_surface_data'
    with zipfile.ZipFile('era5_surface_monthly.nc', 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)
    nc_file = os.path.join(extracted_dir, os.listdir(extracted_dir)[0])
    # Use h5netcdf engine for robustness
    ds2 = xr.open_dataset(nc_file, engine='h5netcdf')
except Exception as e:
    # Catches FileNotFoundError (if the zip isn't there) or IO errors (if the file is corrupt/unreadable)
    print(f"FATAL ERROR: Could not process era5_surface_monthly.nc even with ZIP handler. Error: {e}")
    exit()

# Merge pressure and surface level data
ds_combined = xr.merge([ds1, ds2])

# --- 2. Define Regions and FINALIZED Level Strategy ---
# Defines the boundaries (lat/lon) for key climate phenomena (teleconnection nodes)
regions = {
    "ENSO": {"lat": (-5, 5), "lon": (190, 240)},
    "E_Africa": {"lat": (-10, 10), "lon": (30, 50)},
    "Arctic": {"lat": (70, 90), "lon": (0, 360)},
    "Midlat": {"lat": (40, 60), "lon": (-100, 100)},
}

# Defines the specific variable and pressure level for each required climate proxy
analysis_vars = {
    "t": {"level": 1000, "regions": ["ENSO", "Midlat"]},  # t1000 acts as an SST/low-level proxy
    "z": {"level": 500, "regions": ["Arctic"]},          # z500 captures mid-tropospheric flow (Polar Vortex)
    "q": {"level": 850, "regions": ["E_Africa"]},        # q850 captures low-level moisture transport
    "tp": {"level": None, "regions": ["E_Africa"]},      # tp is surface-level precipitation
}

regional_ts = {}

# --- 3. Aggregation Loop ---
print("\n--- Aggregating Regional Time Series ---")
for var, var_config in analysis_vars.items():

    if var not in ds_combined:
        print(f"Skipping variable {var}: not found in combined dataset.")
        continue

    data = ds_combined[var]

    # 3.1. Vertical Selection
    # Selects the specific pressure level only if the dimension exists and a level is specified
    if "pressure_level" in data.dims and var_config["level"] is not None:
        data = data.sel(pressure_level=var_config["level"], method='nearest')

    if "pressure_level" in data.dims and var_config["level"] is None:
        data = data.mean(dim='pressure_level')

    for name in var_config["regions"]:
        region = regions[name]

        # --- 3.2. Spatial Aggregation & Subsetting ---
        # Latitude slice is reversed (high to low) for xarray's indexing convention
        lat_slice = slice(region["lat"][1], region["lat"][0])
        lon_min = region["lon"][0] % 360
        lon_max = region["lon"][1] % 360

        # Handles regions that cross the Prime Meridian (Longitude Wrapping)
        if lon_max < lon_min:
            subset = xr.concat([
                data.sel(latitude=lat_slice, longitude=slice(lon_min, 360)),
                data.sel(latitude=lat_slice, longitude=slice(0, lon_max))
            ], dim='longitude')
        else:
            subset = data.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))

        # Calculate the final spatial mean across the selected region
        spatial_mean_ts = subset.mean(dim=["latitude", "longitude"])

        # Convert to a Pandas Series and store
        ts_name = f"{var}_{var_config['level'] or ''}_{name}"
        regional_ts[ts_name] = spatial_mean_ts.to_pandas()
        print(f"  - Aggregated: {ts_name}")

# --- 4. Final Concatenation and Save ---
df = pd.concat(regional_ts, axis=1)
df.index.name = 'valid_time'

# Rename columns to a clean format
df.columns = [
    col.replace('_1000_', '1000_').replace('_500_', '500_').replace('_850_', '850_')
    for col in df.columns
]
# Clean up the 'tp' column name
df.columns = [col.replace('tp__E_Africa', 'tp_E_Africa') for col in df.columns]

df.to_csv("regional_timeseries_final.csv")
print("\n--- Final Aggregated Time Series Head ---")
print(df.head())
print(f"\nFinal DataFrame Shape: {df.shape}")
print("\nData saved to: regional_timeseries_final.csv")