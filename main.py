"""
Author: Rashila Lamichhane
Email: rashila.lamichhane01@gmail.com
Description: The main execution script for the CausalEarthNet project. This file orchestrates
    the entire analysis pipeline, from loading the clean climate time series to running
    the PCMCI+ baseline and validating the predictive superiority of the novel
    Hypergraph Causal Discovery method.
"""
# main.py

# Import modules
from src.config import TAU_MAX, PC_ALPHA, ALPHA_LEVEL, MAX_HYPEREDGE_SIZE, RANDOM_STATE, RIDGE_ALPHA, TEST_SIZE, \
    KNN_NEIGHBORS, N_PERMUTATIONS
from data.preprocess import load_clean_data, create_lagged_data_frame
from src.baseline import run_pcmciplus_baseline
from src.hypergraph_discovery import HypergraphCausalDiscovery

def main():
    # --- 1. Data Loading and Cleaning ---
    df_anomaly_clean, data_tigramite, var_names = load_clean_data()
    print(df_anomaly_clean.shape)

    # --- 2. PCMCI+ Baseline (CMIknn Test) ---
    pcmci_links = run_pcmciplus_baseline(
        data=data_tigramite,
        tau_max=TAU_MAX,
        pc_alpha=PC_ALPHA,
        alpha_level=ALPHA_LEVEL,
        var_names=var_names,
        knn_neighbors=KNN_NEIGHBORS
    )

    # --- 3. Prepare Lagged Data for Hypergraph ---
    lagged_data_df = create_lagged_data_frame(df_anomaly_clean, TAU_MAX)
    system_vars = df_anomaly_clean.columns.tolist()
    drivers_lagged_cols = [col for col in lagged_data_df.columns if col not in system_vars]

    # --- 4. Hypergraph Causal Discovery (Our Contribution) ---
    # Define the primary targets for the analysis (based on original script)
    target_ea_final = 'tp_E_Africa'
    target_midlat_final = 't1000_Midlat'

    discoverer = HypergraphCausalDiscovery(
        max_hyperedge_size=MAX_HYPEREDGE_SIZE,
        significance_level=ALPHA_LEVEL,
        ridge_alpha=RIDGE_ALPHA,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        n_permutations=N_PERMUTATIONS
    )

    # Set the PCMCI+ links as the known competition (Baseline)
    discoverer.set_pairwise_baseline(pcmci_links)

    print("\n" + "=" * 50)
    print("STARTING HYPERGRAPH DISCOVERY")
    print("=" * 50)

    # Discover Hyperedges for Primary Targets
    discoverer.discover_hypergraph(lagged_data_df, target_ea_final, drivers_lagged_cols, system_vars)
    discoverer.discover_hypergraph(lagged_data_df, target_midlat_final, drivers_lagged_cols, system_vars)

    # --- 5. Compare Model Performance (Proof of Claim II) ---
    comparison_ea = discoverer.compare_pairwise_vs_hypergraph(lagged_data_df, target_ea_final)
    comparison_midlat = discoverer.compare_pairwise_vs_hypergraph(lagged_data_df, target_midlat_final)

    print("\nEast Africa Rainfall (tp_E_Africa) Outperformance:")
    print(f"Pairwise R²: {comparison_ea['pairwise_r2']:.4f}")
    print(f"Hypergraph R²: {comparison_ea['hypergraph_r2']:.4f}")
    # print(f"Best Pairwise CMI P-value: {comparison_ea['best_pairwise_p']:.4f}")

    print("\nMidlatitude Temperature (t_1000_Midlat) Outperformance:")
    print(f"Pairwise R²: {comparison_midlat['pairwise_r2']:.4f}")
    print(f"Hypergraph R²: {comparison_midlat['hypergraph_r2']:.4f}")
    # print(f"Best Pairwise CMI P-value: {comparison_midlat['best_pairwise_p']:.4f}")


if __name__ == "__main__":
    main()