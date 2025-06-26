import re
import os
import glob
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array
import pickle


def marginalize(x: Array):
    # Ensure x is (T,)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    elif x.ndim != 1:
        raise ValueError(f"Expected x of shape (T,) or (T, 1), got {x.shape}")

    nan_indices = jnp.isnan(x)
    edge_mask = ~(nan_indices[:, None] | nan_indices[None, :])
    return edge_mask

def marginalize_node(rng: jrandom.PRNGKey, edge_mask: Array):
    # Remove a random node from the graph's adjacency matrix
    idx = jax.random.choice(rng, jnp.arange(edge_mask.shape[0]), shape=(1,), replace=False)
    edge_mask = edge_mask.at[idx, :].set(False)  # Remove outgoing edges
    edge_mask = edge_mask.at[:, idx].set(False)  # Remove incoming edges
    edge_mask = edge_mask.at[idx, idx].set(True) # Keep the self-loop
    return edge_mask


def safe_weight_fn(t):
    """
    Example of a numerically stable weight function.
    Replace this with your actual weight function.
    """
    # Ensure time values are finite
    t = jnp.nan_to_num(t, 0.0)
    # Compute weights with clipping to prevent extreme values
    weights = jnp.clip(1.0 / (1.0 + t), 1e-6, 1e6)
    return weights

def save_params(params, filename):
    # Convert device arrays to host (CPU) arrays.
    params_cpu = jax.device_get(params)
    with open(filename, 'wb') as f:
        pickle.dump(params_cpu, f)
    return

def load_params(filename):
    # Load the parameters (as CPU-backed NumPy arrays)
    with open(filename, 'rb') as f:
        params_cpu = pickle.load(f)

    # Optionally, convert NumPy arrays back to JAX arrays.
    # This works recursively over nested dictionaries.
    params = jax.tree_util.tree_map(jnp.array, params_cpu)
    return params

def file_with_largest_integer(folder_path, file_pattern='model_checkpoint_epoch_*.pkl'):
    """
    Scans all files in the given folder using glob and returns the file
    whose filename contains the largest integer.

    Parameters:
        folder_path (str): The directory to scan.
        file_pattern (str): The pattern to match files. Default is 'model_checkpoint_epoch_*.pkl'.

    Returns:
        str: The full path to the file with the largest integer in its name,
             or None if no such file is found.
    """
    # Create a pattern to match all entries in the folder
    pattern = os.path.join(folder_path, file_pattern)
    # Get a list of all files matching the pattern
    files = [f for f in glob.glob(pattern) if os.path.isfile(f)]

    largest_integer = None
    file_with_max_int = None

    for file in files:
        # Extract just the filename
        filename = os.path.basename(file)
        # Find all integers in the filename
        integers = re.findall(r'\d+', filename)
        if integers:
            # Convert found strings to integers and pick the largest one in the file
            max_in_file = max(map(int, integers))
            if (largest_integer is None) or (max_in_file > largest_integer):
                largest_integer = max_in_file
                file_with_max_int = file

    return file_with_max_int
