import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu, pearsonr, wasserstein_distance
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu, pearsonr, wasserstein_distance
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import jensenshannon

try:
    import ot  # POT (Python Optimal Transport) library for multivariate Wasserstein
    pot_available = True
except ImportError:
    pot_available = False
    print("POT library is not available. Install with 'pip install pot' for multivariate Wasserstein.")

# Function to handle NaN values (Impute with mean for simplicity)
def handle_nans(data, strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)
    return imputer.fit_transform(data)

# Multivariate KS Test by taking max KS statistic across each dimension
def multivariate_ks_test(data1, data2):
    ks_stats = []
    p_values = []
    for i in range(data1.shape[1]):  # Loop over each feature
        ks_stat, p_value = ks_2samp(data1[:, i], data2[:, i])
        ks_stats.append(ks_stat)
        p_values.append(p_value)
    return np.mean(ks_stats), np.mean(p_values)

# Multivariate Jensen-Shannon Divergence
def js_divergence(data1, data2, bins=20):
    js_divs = []
    for i in range(data1.shape[1]):  # Loop over each feature
        hist1, _ = np.histogram(data1[:, i], bins=bins, density=True)
        hist2, _ = np.histogram(data2[:, i], bins=bins, density=True)
        js_div = jensenshannon(hist1 + 1e-10, hist2 + 1e-10)
        js_divs.append(js_div)
    return np.mean(js_divs)

# Mann-Whitney U Test for each dimension
def multivariate_mann_whitney(data1, data2):
    stats = []
    p_values = []
    for i in range(data1.shape[1]):  # Loop over each feature
        stat, p_value = mannwhitneyu(data1[:, i], data2[:, i])
        stats.append(stat)
        p_values.append(p_value)
    return np.mean(stats), np.mean(p_values)

# High-Dimensional Pearson Correlation using pairwise correlations across dimensions
def high_dimensional_pearson(data1, data2):
    pearson_corrs = [pearsonr(data1[:, i], data2[:, i])[0] for i in range(data1.shape[1])]
    return np.mean(pearson_corrs)

# High-dimensional Wasserstein Distance
def high_dimensional_wasserstein(data1, data2):
    if pot_available:
        # Using POT for true multivariate Wasserstein distance
        M = ot.dist(data1, data2, metric='euclidean')  # Cost matrix using Euclidean distance
        wasserstein_dist = ot.emd2([], [], M)  # Solves for the Wasserstein distance
    else:
        # Average Wasserstein distance per feature if POT is unavailable
        wasserstein_dists = [wasserstein_distance(data1[:, i], data2[:, i]) for i in range(data1.shape[1])]
        wasserstein_dist = np.mean(wasserstein_dists)
    return wasserstein_dist

# Example usage with your datasets:
def compute_all_metrics(data_clean1, data_clean2):
    # Handle NaN values (impute them)
    data_clean1 = handle_nans(data_clean1)
    data_clean2 = handle_nans(data_clean2)
    
    # Ensure the data is not empty after imputation
    if data_clean1.size == 0 or data_clean2.size == 0:
        raise ValueError("After handling NaN values, one of the datasets is empty.")
    
    # Multivariate KS Test
    ks_stat, ks_p_value = multivariate_ks_test(data_clean1, data_clean2)
    
    # Multivariate Jensen-Shannon Divergence
    js_div = js_divergence(data_clean1, data_clean2)
    
    # Mann-Whitney U Test
    mw_stat, mw_p_value = multivariate_mann_whitney(data_clean1, data_clean2)
    
    # High-Dimensional Pearson Correlation
    corr = high_dimensional_pearson(data_clean1, data_clean2)
    
    # High-Dimensional Wasserstein Distance
    wasserstein_dist = high_dimensional_wasserstein(data_clean1, data_clean2)

    # Displaying the results for the whole dataset comparison
    print("Comparison of Entire Datasets:")
    print(f"KS Statistic (averaged across dimensions) = {ks_stat}, p-value = {ks_p_value}")
    print(f"Jensen-Shannon Divergence (averaged across dimensions) = {js_div}")
    print(f"Mann-Whitney U Statistic (averaged across dimensions) = {mw_stat}, p-value = {mw_p_value}")
    print(f"High-Dimensional Pearson Correlation (averaged across dimensions) = {corr}")
    print(f"High-Dimensional Wasserstein Distance = {wasserstein_dist}")