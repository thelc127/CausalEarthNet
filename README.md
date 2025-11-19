# CausalEarthNet
Learning the set-wise causal structure of the Earth system using hypergraphs that outperforms pairwise models in identifying climate tipping points and enabling transportable counterfactual forecasts

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

   
