import jax.numpy as jnp
import jax.random as jrandom


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


