"""
Author: Rashila Lamichhane
Email: rashila.lamichhane01@gmail.com
Description: Implements the non-linear PCMCI+ algorithm using the CMIknn test. This script
    establishes the causal discovery baseline by finding the strongest statistically
    significant pairwise (single-driver) temporal links in the climate time series.
    The resulting links serve as the competitive reference for the Hypergraph model's
    predictive validation (R² comparison).
"""

# src/baseline.py

import numpy as np
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.pcmci import PCMCI


def run_pcmciplus_baseline(data, tau_max, pc_alpha, alpha_level, var_names, knn_neighbors):
    """
    Implements the PCMCI+ (Non-Linear) Baseline using the CMIknn test
    to find significant pairwise temporal links.
    """
    ci_test = CMIknn(knn=knn_neighbors)
    pcmci = PCMCI(dataframe=data, cond_ind_test=ci_test, verbosity=0)

    print("\n--- Step 2: Running PCMCI+ (Non-Linear CMIknn Baseline) ---")
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)

    try:
        p_matrix = results['p_matrix']
    except KeyError:
        print("Fatal Error: Could not retrieve 'p_matrix'. Cannot proceed.")
        return []

    pairwise_links = []

    # Deducing the best lag from the p_matrix (shape is [N, N, tau_max + 1])
    # Index 0 is lag 0 (contemporaneous), we search from index 1 (lag 1 to tau_max).

    N = len(var_names)
    for j in range(N):  # target index
        for i in range(N):  # driver index
            if p_matrix.shape[2] > 1:
                # Find the minimum P-value across all non-contemporaneous lags
                min_p_index = np.argmin(p_matrix[i, j, 1:])  # Index of the best lag (0 to tau_max-1)
                min_p_value = p_matrix[i, j, min_p_index + 1]  # Value of the best lag

                # The actual lag is the index + 1
                lag = min_p_index + 1

                if min_p_value < alpha_level:
                    pairwise_links.append((var_names[i], var_names[j], lag, min_p_value))

    print(f"\nDiscovered Pairwise Links (P-value < {alpha_level}):")
    if not pairwise_links:
        print("No significant temporal links found by CMIknn.")
    else:
        for driver, target, lag, p_value in sorted(pairwise_links, key=lambda x: x[3]):
            print(f"  {driver} (t-{lag} months)  ->  {target}  |  P-value: {p_value:.4f}")

    return pairwise_links