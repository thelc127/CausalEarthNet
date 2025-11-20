# CausalEarthNet
Learning the set-wise causal structure of the Earth system using hypergraphs that outperforms pairwise models in identifying climate tipping points and enabling transportable counterfactual forecasts

## Code Structure
```
CausalEarthNet
├── data
│   ├── download_era5.py
│   ├── preprocess.py
│   ├── aggregate.py
│   └── era5_pressure_monthly.nc
│   └── era5_surface_monthly.nc
│   └── feature_set.csv
│   └── regional_timeseries_final.csv
├── src
│   ├── config.py
│   ├── hypergraph_discovery.py
│   └── baseline.py
└── main.py
└── requirements.txt

```
#### Setup and Dependencies 

## Step 1 : Prerequisities and Required libraries <br>

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
#### Analysis Pipeline

## Step 3 : Data Download and Preprocessing

This stage generates the essential input file, ```regional_timeseries_final.csv```

***3.1.Download raw ERA5 data*** <br>

The script below downloads 20 years (2000-2019) of monthly mean ERA5 data for the variables and pressure levels needed for our regional proxies.
```
python data/download_era5.py 
# Downloads 'era5_surface_monthly.nc' and 'era5_pressure_monthly.nc' 
```
***3.2 Aggregate Regional Time Series***

This script loads the raw, multi-dimensional NetCDF files, for example:
```
ds1 = xr.open_dataset('era5_pressure_monthly.nc', engine='netcdf4')
 ```
merges them ```(xr.merge)``` and process them into the clean, univariate regional time series by performing vertical level selection (e.g., z500​) and spatial averaging over defined regions (e.g., ENSO, Arctic, E_Africa).
```
python data/aggregate_data.py
# Generates 'regional_timeseries_final.csv'
```

3.2.1. Defining Regional Proxies and Levels <br>

| Region Name | Latitude (°) | Longitude (°) | Relevant Variable | Pressure Level | Final Column Name |
|-------------|--------------|---------------|--------------------|----------------|--------------------|
| ENSO        | -5 to 5      | 190 to 240    | Temperature (t)    | 1000 hPa       | t1000 ENSO         |
| Midlat      | 40 to 60     | -100 to 100   | Temperature (t)    | 1000 hPa       | t1000 Midlat       |
| Arctic      | 70 to 90     | 0 to 360      | Geopotential (z)   | 500 hPa        | z500 Arctic        |
| E_Africa    | -10 to 10    | 30 to 50      | Specific Humidity (q) | 850 hPa     | q850 E_Africa      |
| E_Africa    | -10 to 10    | 30 to 50      | Total Precipitation (tp) | Surface | tp E_Africa       |

References: <br>
https://www.ncei.noaa.gov/access/monitoring/enso/sst <br>
https://www.ncei.noaa.gov/access/monitoring/ao/ <br>
https://www.cpc.ncep.noaa.gov/products/CDB/Extratropics/fige7.shtml <br>
https://charlie.weathertogether.net/2012/09/el-ninosouthern-oscillation-enso-for/ <br>
https://pure.iiasa.ac.at/id/eprint/15033/1/Moon%20J.%20et%20al_SJFS_NO6.pdf <br>
https://www.aoml.noaa.gov/phod/docs/lopez_kirtman_climate_dynamics_2018.pdf <br>

3.2.2. Aggregation Step <br>
Summarize multi-dimensional climate datasets into interpretable (uni-dimensional)signals for analysis, like climate indices or teleconnections proxies. 

   *A. Vertical Selection*: <br> 
   Example: <br>
   ``` data = data.sel(pressure_level = 500, method='nearest') ``` <br>
   
   This isolates the signal at the layer most relevant to the physical process. This is not needed for surface-only fields (eg. precipitation) 
   
   *B. Spatial Aggregation*:<br>
   
   The code calculates the mean value across all latitude and longitude points within the defined box. 
   ```(.mean(dim = [“latitude”, “longitude”]))```

All resulting univariate time-series data are concatenated into a single pandas DataFrame, indexed by time, and saved as, ```regional_timeseries_final.csv```, which is then ready for the next step. 

*3.3 Feature Engineering* <br>

Transforms the aggregated climate time series data ```(regional_timeseries_final.csv)``` into the standardized formats required for your causal inference and predictive modeling steps. <br> 

3.3.1 Anomaly Detection: <br>

```load_clean_data()``` : outputs a clean, stationary time series of anamoloies by removing the strong seasonal change, ensures the data relects unpredictable anomalies, and isolates non-seasonal physical teleconnections. 

3.3.2. Handling Missing Values: <br>
Original csv files has a lot of missing values. Uses forward-fill ```ffill``` and backward fill ```bfill```, and then drops remaining NaN rows. <br> 
Ensures the resulting time series is complete and continuous

3.3.3 Conversion to tigramite dataframe: <br> 

```df_anomaly_clean``` is converted into a tigramite dataframe to prepare it specifically for baseline analysis

3.3.4 Handling Time Lag: **Additional step**  <br>

```create_lagged_dataframe()``` transforms the timeseries data to a feature matrix, and explicitly creates separate columns for every lagged time step up to ```max_lag``` values. 
For example: If max_lag = 4, for t_{1000_Midlat}, it creates t_{1000_Midlat_t-1}, t_{1000_Midlat_t-2}, t_{1000_Midlat_t-3}, and t_{1000_Midlat_t-4}

The resulting dataframe is used to train Ridge regression models, and used as input features (**X** matrix) for target variable at time t (**Y_t**). Output: ```feature_set.csv``` <br>

## Step 4 : Causal Analysis and Comparison <br> 

This step uses the aggregated time series data to perform the two analyses: 1) the PCMCI+ baseline 2) Hypergraph method. The goal is to compare their predictive power (R^2) <br>

**4.1 PCMCI+ Baseline Comparison** <br> 

```src/pcmciplus_baseline.py``` 
has the baseline model. <br>

PCMCI+(Partial Conditional Mutual Information & Conditional Independence) <br>

Independence Test: CMIknn (Conditional Mutual Information, based on k-nearest neighbours) to capture non-linear relationships. <br>

Output: List of most statistically significant (P-value > alpha) **pairwise links** (eg. t_1000 ENSO at (t-3) -> t_pE Africa). Used as input for the baseline model in the final comparison. 

**4.2 Hypergraph Discovery** <br>
```
src/hypergraph_discovery.py
```
Implements the hypergraph causal discovery method

4.2.1 Conditional Mutual Information : ```conditional_mutual_information(X,Y,Z)``` estimates CMI **(I(X_s;Y | Z)**, where Xs is set of lagged dricers, Y is the target variable at the present time t and Z is set of all other system variables <br>

**Significance:**  Higher CMI values indicates a stronger dependency between the set X and the target Y

4.2.2 Independence Test : ```test_independence(X, Y, Z)``` determines if the observed CMI value is significantly significant. <br>
 **If is below the alpha value(significance level), the hyperlink is considered significant** <br>

4.2.3 Search for hyperedges: Function ```discover_hypergraph()``` <br>
Records all set Xs that are sigificant as the discovered hyperedges  <br>

**4.3 Performance Comparison** <br>

The pairwise links obtained from baseline model (PCMCI+) are used to identify the single best pairwise driver for the target variable, based on lowest p-value and highest CMI. Next, a ridge regression model is trained on this driver, and R2 value(r2_pair) is recorded as performance baseline. <br>

Likwise, for hyperedge discovered, regression model is trained using all the features within this best hyperedge and r2_hyper value is recorded.  <br>

The claim is made by reporting the relative R2 improvement. The result quantitavely demonstrates that the set-wise interactions(hyperedges) significantly improve the predictive power over the strongest single-link baseline. 










