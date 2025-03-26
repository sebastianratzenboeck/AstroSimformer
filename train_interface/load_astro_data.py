import os
import pandas as pd
import numpy as np
import jax.random as jrandom
import jax.numpy as jnp
from sklearn.impute import KNNImputer
from new.Data import normalize_data, normalize_with_params



def generate_data_combined_normal(key: jrandom.PRNGKey, n: int):
    """
    Generates synthetic data or returns real data with `logAge` included if provided.
    Includes NaN handling using KNN imputation.

    Args:
        key (jrandom.PRNGKey): Random key for reproducibility.
        n (int): Number of data points to generate.

    Returns:
        tuple: Normalized data, raw data, means, and standard deviations.
    """
    # Load the dataset
    current_directory = os.path.dirname(os.path.abspath("__file__"))
    file_name = "sebdata.csv"
    file_path = os.path.join(current_directory, file_name)
    df = pd.read_csv(file_path)

    # Sample `n` rows from the data
    df = df.sample(n=n)

    # Specify the features to include
    features_X_max = [
        'parallax_obs', 'A_V_obs', 'phot_g_mean_mag_obs', 'phot_bp_mean_mag_obs', 'phot_rp_mean_mag_obs',
        'j_obs', 'h_obs', 'k_obs', 'w1_obs', 'w2_obs', 'irac1_obs', 'irac2_obs', 'irac3_obs', 'irac4_obs',
        'parallax_error', 'phot_g_mean_mag_error', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error',
        'j_error', 'h_error', 'k_error', 'w1_error', 'w2_error', 'irac1_error', 'irac2_error', 'irac3_error',
        'irac4_error'
    ]

    # Extract features and target
    real_data = df[features_X_max].values
    log_age = df['logAge'].values

    # Impute NaN values using KNN
    knn_imputer = KNNImputer(n_neighbors=5)
    real_data_imputed = knn_imputer.fit_transform(real_data)

    # Convert data to JAX arrays
    log_age_array = jnp.array(log_age).reshape(-1, 1)
    real_data_array = jnp.array(real_data_imputed)

    # Introduce NaNs (if required)
    real_data_array_dirty = real_data  # Placeholder

    # Combine logAge with features
    combined_data = jnp.concatenate([log_age_array, real_data_array], axis=1).reshape(n, -1, 1)
    combined_data_dirty = jnp.concatenate([log_age_array, real_data_array_dirty], axis=1).reshape(n, -1, 1)

    # Normalize data
    normalized_data, means, stds = normalize_data(combined_data_dirty)
    data = normalize_with_params(combined_data, means, stds)

    return normalized_data, data, combined_data_dirty, combined_data, means, stds


def generate_data_combined_normal_par(key: jrandom.PRNGKey, n: int, means, stds):
    """
    Generates synthetic data or returns real data with `logAge` included if provided.
    Includes NaN handling using KNN imputation.

    Args:
        key (jrandom.PRNGKey): Random key for reproducibility.
        n (int): Number of data points to generate.
        means (array): Means for normalization.
        stds (array): Standard deviations for normalization.

    Returns:
        tuple: Normalized data and raw data.
    """
    # Load the dataset
    current_directory = os.path.dirname(os.path.abspath("__file__"))
    file_name = "sebdata.csv"
    file_path = os.path.join(current_directory, file_name)
    df = pd.read_csv(file_path)

    # Sample `n` rows from the data
    df = df.sample(n=n)

    # Specify the features to include
    features_X_max = [
        'parallax_obs', 'A_V_obs', 'phot_g_mean_mag_obs', 'phot_bp_mean_mag_obs', 'phot_rp_mean_mag_obs',
        'j_obs', 'h_obs', 'k_obs', 'w1_obs', 'w2_obs', 'irac1_obs', 'irac2_obs', 'irac3_obs', 'irac4_obs',
        'parallax_error', 'phot_g_mean_mag_error', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error',
        'j_error', 'h_error', 'k_error', 'w1_error', 'w2_error', 'irac1_error', 'irac2_error', 'irac3_error',
        'irac4_error'
    ]

    # Extract features and target
    real_data = df[features_X_max].values
    log_age = df['logAge'].values

    # Impute NaN values using KNN
    knn_imputer = KNNImputer(n_neighbors=5)
    real_data_imputed = knn_imputer.fit_transform(real_data)

    # Convert data to JAX arrays
    log_age_array = jnp.array(log_age).reshape(-1, 1)
    real_data_array = jnp.array(real_data_imputed)

    # Introduce NaNs (if required)
    real_data_array_dirty = real_data  # Placeholder

    # Combine logAge with features
    combined_data = jnp.concatenate([log_age_array, real_data_array], axis=1).reshape(n, -1, 1)
    combined_data_dirty = jnp.concatenate([log_age_array, real_data_array_dirty], axis=1).reshape(n, -1, 1)

    # Normalize data
    normalized_data = normalize_with_params(combined_data_dirty, means, stds)
    data = normalize_with_params(combined_data, means, stds)

    return normalized_data, data


def generate_data_combined_normal_par_no_norm(key: jrandom.PRNGKey, n: int, means, stds):
    """
    Generates synthetic data or returns real data with `logAge` included if provided.
    Includes NaN handling using KNN imputation.

    Args:
        key (jrandom.PRNGKey): Random key for reproducibility.
        n (int): Number of data points to generate.
        means (array): Means for normalization.
        stds (array): Standard deviations for normalization.

    Returns:
        tuple: Normalized data and raw data.
    """
    # Load the dataset
    current_directory = os.path.dirname(os.path.abspath("__file__"))
    file_name = "sebdata.csv"
    file_path = os.path.join(current_directory, file_name)
    df = pd.read_csv(file_path)
    # Specify the features to include
    features_X_max = [
        'parallax_obs', 'A_V_obs', 'phot_g_mean_mag_obs', 'phot_bp_mean_mag_obs', 'phot_rp_mean_mag_obs',
        'j_obs', 'h_obs', 'k_obs', 'w1_obs', 'w2_obs', 'irac1_obs', 'irac2_obs', 'irac3_obs', 'irac4_obs',
        'parallax_error', 'phot_g_mean_mag_error', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag_error',
        'j_error', 'h_error', 'k_error', 'w1_error', 'w2_error', 'irac1_error', 'irac2_error', 'irac3_error',
        'irac4_error'
    ]

    # Check and handle missing features in the dataframe
    for feature in features_X_max:
        if feature not in df.columns:
            df[feature] = 0  # Add missing feature as a column of zeros

    # Sample `n` rows from the data (with replacement if necessary)
    if len(df) < n:
        df = df.sample(n=n, replace=True, random_state=42)
    else:
        df = df.sample(n=n, random_state=42)

    # Extract features and target
    real_data = df[features_X_max].values
    log_age = df['logAge'].values if 'logAge' in df.columns else np.zeros(len(df))

    # Impute NaN values using KNN
    knn_imputer = KNNImputer(n_neighbors=5)
    real_data_imputed = knn_imputer.fit_transform(real_data)

    # Convert data to JAX arrays
    log_age_array = jnp.array(log_age).reshape(-1, 1)
    real_data_array = jnp.array(real_data_imputed)

    # Introduce NaNs for dirty data (if required)
    real_data_array_dirty = jnp.array(real_data)  # Retain original array as-is for dirty data

    # Combine logAge with features
    combined_data = jnp.concatenate([log_age_array, real_data_array], axis=1).reshape(n, -1, 1)
    combined_data_dirty = jnp.concatenate([log_age_array, real_data_array_dirty], axis=1).reshape(n, -1, 1)

    # Ensure consistent shape (n, 28, 1)
    assert combined_data.shape == (n, 28, 1), f"Cleaned data shape mismatch: {combined_data.shape}"
    assert combined_data_dirty.shape == (n, 28, 1), f"Dirty data shape mismatch: {combined_data_dirty.shape}"

    return combined_data_dirty, combined_data

