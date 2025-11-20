"""
Author: Rashila Lamichhane
Email: rashila.lamichhane01@gmail.com
Description: Implements the HypergraphCausalDiscovery class, which performs two main tasks:
    1) Discovers higher-order (set-wise) causal interactions using Conditional Mutual
       Information (CMI) and permutation testing.
    2) Validates the novelty by comparing the predictive skill (R²) of the best hyperedge
       against the best single-link (pairwise) driver found by PCMCI+.
"""

# src/hypergraph_discovery.py
import numpy as np
import itertools
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class HypergraphCausalDiscovery:
    """
    Hypergraph-based causal discovery for Earth system dynamics.
    Implements the novel method to discover higher-order (set-wise) interactions.
    """

    def __init__(self, max_hyperedge_size, significance_level, ridge_alpha, test_size, random_state, n_permutations):
        self.max_hyperedge_size = max_hyperedge_size
        self.significance_level = significance_level
        self.ridge_alpha = ridge_alpha
        self.test_size = test_size
        self.random_state = random_state
        self.n_permutations = n_permutations
        self.hypergraph = {'nodes': set(), 'edges': []}
        self.pairwise_baseline = {}

    def conditional_mutual_information(self, X, Y, Z=None):
        """
        Compute conditional mutual information I(X;Y|Z) using the non-linear residual approximation.
        """
        Y_1d = Y.ravel()

        if Z is None or Z.shape[1] == 0:
            mi_values = mutual_info_regression(X, Y_1d, random_state=self.random_state)
            mi = np.mean(mi_values) if mi_values.size > 0 else 0.0
        else:
            # Conditional MI: I(X;Y|Z) ~ I(res_X|Z ; res_Y|Z)
            X_2d = X.reshape(-1, X.shape[1]) if X.ndim == 1 else X

            # Regress X on Z
            model_x = Ridge(alpha=0.1).fit(Z, X_2d)
            residual_x = X_2d - model_x.predict(Z)

            # Regress Y on Z
            model_y = Ridge(alpha=0.1).fit(Z, Y_1d)
            residual_y = Y_1d - model_y.predict(Z)

            # MI between residuals
            mi_values = mutual_info_regression(residual_x, residual_y, random_state=self.random_state)
            mi = np.mean(mi_values) if mi_values.size > 0 else 0.0
        return mi

    def test_independence(self, X, Y, Z=None):
        """Test conditional independence (I(Y; X_S | Z)) using permutation test (for p-value)."""
        Y_1d = Y.ravel()
        observed_cmi = self.conditional_mutual_information(X, Y_1d, Z)

        null_distribution = []
        for _ in range(self.n_permutations):
            Y_perm = np.random.permutation(Y_1d)
            null_cmi = self.conditional_mutual_information(X, Y_perm, Z)
            null_distribution.append(null_cmi)

        p_value = np.mean(np.array(null_distribution) >= observed_cmi)
        is_independent = p_value > self.significance_level
        return is_independent, p_value, observed_cmi

    def set_pairwise_baseline(self, pcmci_links):
        """Sets the PCMCI+ links as the strongest pairwise baseline for comparison."""
        self.pairwise_baseline = {}
        for driver, target, lag, p_value in pcmci_links:
            source_col = f'{driver}_t-{lag}'
            if target not in self.pairwise_baseline:
                self.pairwise_baseline[target] = []
            self.pairwise_baseline[target].append({'source': source_col, 'cmi_p': p_value})

    def discover_hypergraph(self, data_df, target_var_t, all_drivers_lagged_cols, system_vars_t):
        """Discover hypergraph structure for a specific target variable Y_t."""
        print(f"\n=== Discovering hypergraph for {target_var_t} ===")

        Y = data_df[target_var_t].values.ravel()
        hyperedges = []

        # Z: Conditioning set includes all other contemporaneous system variables
        Z_cols = [v for v in system_vars_t if v != target_var_t]
        Z_data = data_df[Z_cols].values

        for size in range(2, self.max_hyperedge_size + 1):
            for source_set_cols in itertools.combinations(all_drivers_lagged_cols, size):
                source_set_cols = list(source_set_cols)
                X_set = data_df[source_set_cols].values

                is_indep, p_value, cmi = self.test_independence(X_set, Y, Z_data)

                if not is_indep:
                    hyperedge = {'sources': source_set_cols, 'target': target_var_t, 'cmi': cmi, 'p_value': p_value,
                                 'order': size}
                    hyperedges.append(hyperedge)
                    source_str = " ∩ ".join(source_set_cols)
                    # print(f"  {{{source_str}}} -> {target_var_t}: CMI={cmi:.4f}, p={p_value:.4f} ✓")
                    print(f"  {{{source_str}}} -> {target_var_t}: CMI={cmi:.4f}")

        self.hypergraph['edges'].extend(hyperedges)
        return hyperedges

    def compare_pairwise_vs_hypergraph(self, data_df, target_var_t):
        """Compares R² performance of the best pairwise link vs. the best hyperedge."""

        # 1. Identify Best Pairwise Model (Baseline)
        if target_var_t not in self.pairwise_baseline or not self.pairwise_baseline[target_var_t]:
            print(f"Skipping comparison for {target_var_t}: No significant pairwise baseline found.")
            return {'pairwise_r2': 0.0, 'hypergraph_r2': 0.0, 'best_pairwise_p': 1.0}

        best_link = min(self.pairwise_baseline[target_var_t], key=lambda x: x['cmi_p'])
        best_pairwise_driver = [best_link['source']]
        best_pairwise_p = best_link['cmi_p']
        Y = data_df[target_var_t].values.ravel()

        # 2. Identify Best Hypergraph Model (Novelty)
        relevant_hyperedges = [e for e in self.hypergraph['edges'] if e['target'] == target_var_t]

        # Default to the best pairwise driver if no significant hyperedge is found
        best_hyperedge_drivers = best_pairwise_driver
        if relevant_hyperedges:
            best_edge = max(relevant_hyperedges, key=lambda x: x['cmi'])
            best_hyperedge_drivers = best_edge['sources']


        # # --- 3. Fit Models and Calculate R² ---
        #
        X_pair = data_df[best_pairwise_driver].values
        X_hyper = data_df[best_hyperedge_drivers].values

        # Standardize Y
        Y_norm = (Y - np.mean(Y)) / np.std(Y)

        # Split data consistently for both models
        X_train_p, X_test_p, Y_train, Y_test = train_test_split(
            X_pair, Y_norm, test_size=self.test_size, random_state=self.random_state
        )
        X_train_h, X_test_h, _, _ = train_test_split(
            X_hyper, Y_norm, test_size=self.test_size, random_state=self.random_state
        )

        # 📢 CRITICAL FIX: Initialize and Fit Scaler on X_train only 📢
        scaler_p = StandardScaler()
        X_train_p_scaled = scaler_p.fit_transform(X_train_p)
        X_test_p_scaled = scaler_p.transform(X_test_p)

        scaler_h = StandardScaler()
        X_train_h_scaled = scaler_h.fit_transform(X_train_h)
        X_test_h_scaled = scaler_h.transform(X_test_h)

        # Fit and Score Pairwise (using scaled data)
        model_pair = Ridge(alpha=self.ridge_alpha).fit(X_train_p_scaled, Y_train)
        r2_pair = r2_score(Y_test, model_pair.predict(X_test_p_scaled))

        # Fit and Score Hypergraph (using scaled data)
        model_hyper = Ridge(alpha=self.ridge_alpha).fit(X_train_h_scaled, Y_train)
        r2_hyper = r2_score(Y_test, model_hyper.predict(X_test_h_scaled))

        #4. Report Comparison
        print(f"\nPairwise Baseline (Best CMI Link: {best_pairwise_driver[0]}): R²={r2_pair:.4f}")
        print(
            f"Hypergraph Novelty (Best Hyperedge: {' ∩ '.join(best_hyperedge_drivers) if best_hyperedge_drivers else 'N/A'}): R²={r2_hyper:.4f}")

        if r2_pair > 0.01:
            improvement = (r2_hyper - r2_pair) / r2_pair * 100
            print(f"Relative R² Improvement: {improvement:+.1f}%")

        return {'pairwise_r2': r2_pair, 'hypergraph_r2': r2_hyper, 'best_pairwise_p': best_pairwise_p}
