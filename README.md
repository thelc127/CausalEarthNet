# CausalEarthNet
Learning the set-wise causal structure of the Earth system using hypergraphs that outperforms pairwise models in identifying climate tipping points and enabling transportable counterfactual forecasts

## Setup and Dependencies 
**Step 1: Prerequisities and Required libraries** <br>
Before running the data acquisition step, you must set up the necessary environment. All requirements are listed in requirements.txt file <br> 
``` 
pip install -r requirements.txt
```
or copy paste below:
```
# Core Data Handling and Scientific Computing
pandas
numpy
xarray
matplotlib

# Causal Inference and Modeling
tigramite
scikit-learn

# Data Acquisition and I/O Backends (Crucial for ERA5 formats)
cdsapi
cfgrib
netcdf4
h5netcdf
```
**Step 2: Data Acquisition** <br>
To access the data from Climate Data Store(CDS), follow the steps below: 

1. Setup the CDS API key <br>
   Create or register an account at https://cds.climate.copernicus.eu
   
2. Install the CDS API <br>
      The CDS API client is a Python based library. It provides support for Python 3.
   ```
   pip install cdsapi
    ```
3. Use the CDS API client for data access <br> 
   Once the CDS API client is installed, it can be used to request data from the datasets listed in the CDS, ADS, ECDS, XDS and CEMS Early Warning DS catalogues. The bottom of each dataset download form has "Show API    request code" button to display a Python code snippet that can be used to run the manually built request. 
   For example: 
   ``` 
   import cdsapi
   client = cdsapi.Client()
   dataset = "<DATASET-SHORT-NAME>"
   request = {
       <SELECTION-REQUEST>
   }
   target = "<TARGET-FILE>"
   client.retrieve(dataset, request, target)
   ```
## Analysis Pipeline
**Step 1: Data Download and Preprocessing**
This stage generates the essential input file, ```regional_timeseries_final.csv```

***1.1.Download raw ERA5 data*** <br>
The script below downloads 20 years (2000-2019) of monthly mean ERA5 data for the variables and pressure levels needed for our regional proxies.
```
python data/download_era5.py 
# Downloads 'era5_surface_monthly.nc' and 'era5_pressure_monthly.nc' 
```
***1.2 Aggregate Regional Time Series***

This script takes the raw, multi-dimensional NetCDF files, merges them ```(xr.merge)``` and process them into the clean, univariate regional time series by performing vertical level selection (e.g., z500​) and spatial averaging over defined regions (e.g., ENSO, Arctic, E_Africa).
```
python data/aggregate_data.py
# Generates 'regional_timeseries_final.csv'
```

A. Defining Regional Proxies and Levels <br>

| Region Name | Latitude (°) | Longitude (°) | Relevant Variable | Pressure Level | Final Column Name |
|-------------|--------------|---------------|--------------------|----------------|--------------------|
| ENSO        | -5 to 5      | 190 to 240    | Temperature (t)    | 1000 hPa       | t1000 ENSO         |
| Midlat      | 40 to 60     | -100 to 100   | Temperature (t)    | 1000 hPa       | t1000 Midlat       |
| Arctic      | 70 to 90     | 0 to 360      | Geopotential (z)   | 500 hPa        | z500 Arctic        |
| E_Africa    | -10 to 10    | 30 to 50      | Specific Humidity (q) | 850 hPa     | q850 E_Africa      |
| E_Africa    | -10 to 10    | 30 to 50      | Total Precipitation (tp) | Surface | tp E_Africa       |

References: 
https://www.ncei.noaa.gov/access/monitoring/enso/sst <br>
https://www.ncei.noaa.gov/access/monitoring/ao/ <br>
https://www.cpc.ncep.noaa.gov/products/CDB/Extratropics/fige7.shtml <br>
https://charlie.weathertogether.net/2012/09/el-ninosouthern-oscillation-enso-for/ <br>
https://pure.iiasa.ac.at/id/eprint/15033/1/Moon%20J.%20et%20al_SJFS_NO6.pdf <br>
https://www.aoml.noaa.gov/phod/docs/lopez_kirtman_climate_dynamics_2018.pdf <br>



