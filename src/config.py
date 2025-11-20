"""
Author: Rashila Lamichhane
Email: rashila.lamichhane01@gmail.com
Description: Configuration file for the causal discovery pipeline.
Defines hyperparameters for data processing and causal inference methods.
"""

# src/config.py

# Max lag (in months) to consider for causal links (tau_max in PCMCI)
TAU_MAX = 4

# Alpha level for the PC (Pruning Conditional) step in PCMCI
PC_ALPHA = 0.05

# P-value threshold for considering a link as statistically significant
ALPHA_LEVEL = 0.01

# --- Configuration for Regression Comparison ---
TEST_SIZE = 0.3  # Proportion of data to use for the test set
RANDOM_STATE = 42 # Seed for reproducibility
RIDGE_ALPHA = 1.0 # Regularization strength for Ridge Regression
MAX_HYPEREDGE_SIZE = 3 # Maximum size of a hyperedge (set of drivers)

# --- Configuration for CMI Test ---
KNN_NEIGHBORS = 5 # Number of nearest neighbors for CMIknn test
N_PERMUTATIONS = 20 # Number of permutations for the CMI independence test