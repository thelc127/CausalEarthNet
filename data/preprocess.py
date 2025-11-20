"""
Author: Rashila Lamichhane
Email: rashila.lamichhane01@gmail.com
Description: Performs essential data standardization and feature engineering for the causal analysis.
    This includes calculating climatological anomalies (removing the seasonal cycle)
    and creating the time-lagged feature matrix (feature_set.csv) required for predictive modeling
"""

# data/preprocess.py
import os
import pandas as pd
from tigramite import data_processing as dp


def load_clean_data(file_path='data/regional_timeseries_final.csv'):
    """
    Loads raw time series data, calculates climatological anomalies,
    and performs basic imputation/cleaning.
    """
    print("--- Step 1: Loading and Cleaning Data ---")

    # Load data
    df_monthly = pd.read_csv(file_path, index_col='valid_time', parse_dates=True)

    # Calculate monthly climatology
    monthly_climatology = df_monthly.groupby(df_monthly.index.month).mean()

    # Calculate anomalies (subtract climatology)
    df_anomaly = df_monthly.groupby(df_monthly.index.month).apply(
        lambda x: x - monthly_climatology.loc[x.name]
    ).reset_index(level=0, drop=True)

    # Impute missing data (Forward fill then Backward fill)
    df_anomaly_imputed = df_anomaly.ffill().bfill()

    # Final cleanup (drops any remaining NaN rows, if the series start/end with NaNs)
    df_anomaly_clean = df_anomaly_imputed.dropna()

    # Prepare data for Tigramite (PCMCI)
    var_names = list(df_anomaly_clean.columns)
    T = df_anomaly_clean.values
    data_tigramite = dp.DataFrame(T, var_names=var_names)

    print(f"Cleaned Time Series Shape: {df_anomaly_clean.shape}")

    return df_anomaly_clean, data_tigramite, var_names


def create_lagged_data_frame(df, max_lag):
    """
    Creates a DataFrame with explicit lagged variables for the causal search
    (required for the Hypergraph class's regression models).
    """
    df_lagged = pd.DataFrame(df)

    for var in df.columns:
        for lag in range(1, max_lag + 1):
            df_lagged[f'{var}_t-{lag}'] = df[var].shift(lag)

    return df_lagged.dropna()

if __name__ == "__main__":
    df_anomaly, _, _ = load_clean_data(file_path='regional_timeseries_final.csv')
    df_lagged = create_lagged_data_frame(df_anomaly, max_lag = 4)
    df_lagged.to_csv("feature_set.csv")