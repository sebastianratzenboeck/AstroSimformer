from sbi.analysis import pairplot
from jax.random import PRNGKey

import jax.numpy as jnp
import jax.random as jrandom
from jax import random as jax_random


def random_nan(data, key, nan_fraction=0.1):
    """Randomly set `nan_fraction` of values in the data to NaN."""
    # Calculate number of elements to replace with NaN
    total_elements = data.size
    nan_elements = int(total_elements * nan_fraction)
    
    # Generate random indices to replace with NaN
    key, subkey = jrandom.split(key)
    nan_indices = jrandom.choice(subkey, total_elements, shape=(nan_elements,), replace=False)
    
    # Create a mask of the same shape as the data
    nan_mask = jnp.zeros(total_elements, dtype=bool)
    nan_mask = nan_mask.at[nan_indices].set(True)
    
    # Reshape the mask to match the data's shape and apply NaNs
    nan_mask = nan_mask.reshape(data.shape)
    data_with_nans = jnp.where(nan_mask, jnp.nan, data)
    
    return data_with_nans


def normalize_data(data, epsilon=1e-8):
    """
    Normalize the data using (x - mean) / std with handling for NaN values.
    
    Args:
        data: Array of shape (n, features, 1)
        epsilon: Small constant to avoid division by zero
        
    Returns:
        normalized_data: Array of same shape as input, normalized per feature
        means: Mean values per feature
        stds: Standard deviation values per feature
    """
    # Reshape to (n, features) for easier processing
    data_reshaped = data.reshape(data.shape[0], -1)
    
    # Calculate mean ignoring NaN values
    means = jnp.nanmean(data_reshaped, axis=0)
    
    # Calculate std ignoring NaN values
    stds = jnp.nanstd(data_reshaped, axis=0)
    
    # Replace zero standard deviations with 1 to avoid division by zero
    stds = jnp.where(stds < epsilon, 1.0, stds)
    
    # Normalize the data
    normalized = (data_reshaped - means) / stds
    
    # Reshape back to original shape
    normalized = normalized.reshape(data.shape)
    
    return normalized, means, stds


def normalize_with_params(data, means, stds, epsilon=1e-8):
    """
    Normalize data using pre-computed means and standard deviations.
    
    Args:
        data: Array to normalize
        means: Pre-computed mean values
        stds: Pre-computed standard deviation values
        epsilon: Small constant to avoid division by zero
        
    Returns:
        normalized_data: Normalized array
    """
    # Reshape to (n, features) for easier processing
    data_reshaped = data.reshape(data.shape[0], -1)
    # Ensure stds don't contain zeros
    stds = jnp.where(stds < epsilon, 1.0, stds)
    
    # Apply normalization
    normalized = (data_reshaped - means) / stds
    
    # Reshape back to original shape
    normalized = normalized.reshape(data.shape)
    return normalized


def denormalize_data(normalized_data, means, stds):
    """
    Reverse the normalization process to recover the original scale of the data.
    
    Args:
        normalized_data: Normalized array
        means: Mean values used in normalization
        stds: Standard deviation values used in normalization
        
    Returns:
        denormalized_data: Array in original scale
    """
    # Reshape to (n, features) for easier processing
    data_reshaped = normalized_data.reshape(normalized_data.shape[0], -1)
    
    # Reverse the normalization
    denormalized = (data_reshaped * stds) + means
    
    # Reshape back to original shape
    denormalized = denormalized.reshape(normalized_data.shape)
    
    return denormalized


def generate_data(key: PRNGKey, n: int, nan_fraction=0.1):
    key1, key2, key3, key4, key5 = jrandom.split(key, 5)
    theta1 = jrandom.normal(key1, (n, 1)) * 3
    theta2 = jrandom.normal(key4, (n, 1)) * 2
    
    x1 = 2 * jnp.sin(theta1) + jrandom.normal(key2, (n, 1)) * 0.5
    x2 = 0.1 * theta1**2 + 0.5 * jnp.abs(x1) * jrandom.normal(key3, (n, 1))
    x3 = 0.5 * theta2 + 0.2 * jnp.abs(x2) * jrandom.normal(key5, (n, 1))
    
    data = jnp.concatenate([theta1, theta2, x1, x2, x3], axis=1).reshape(n, -1, 1)
    data_with_nans = random_nan(data, key, nan_fraction)
    
    # Normalize data
    normalized_data, means, stds = normalize_data(data_with_nans)
    
    return normalized_data, means, stds


def denormalize_data(normalized_data: jnp.ndarray, means: jnp.ndarray, 
                     stds: jnp.ndarray) -> jnp.ndarray:
    """
    Reverse the normalization process to recover the original scale of the data.
    
    Args:
        normalized_data: Array of shape (n_samples, time_steps, n_features)
        means: Mean values used in normalization (shape: n_features)
        stds: Standard deviation values used in normalization (shape: n_features)
        
    Returns:
        denormalized_data: Array in original scale
    """
    # Reshape means and stds for broadcasting
    means_broadcast = means.reshape(1, 1, -1)  # Shape: (1, 1, n_features)
    stds_broadcast = stds.reshape(1, 1, -1)    # Shape: (1, 1, n_features)
    
    # Reverse the normalization
    denormalized = normalized_data * stds_broadcast + means_broadcast
    
    return denormalized


def denormalize_data_try(normalized_data, means, stds):
    """
    Reverse the normalization process to recover the original scale of the data.
    
    Args:
        normalized_data: Normalized array
        means: Mean values used in normalization
        stds: Standard deviation values used in normalization
        
    Returns:
        denormalized_data: Array in original scale
    """
    # Reshape to (n, features) for easier processing
    data_reshaped = normalized_data[:,-1,:]
    
    # Reverse the normalization
    denormalized = (data_reshaped * stds) + means
    
    # Reshape back to original shape
    denormalized = denormalized.reshape(normalized_data.shape)
    
    return denormalized