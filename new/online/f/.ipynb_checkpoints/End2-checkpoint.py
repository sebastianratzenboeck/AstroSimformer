# conditional_sampling.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import jax.random as jrandom
import jax.numpy as jnp
from typing import Tuple, Callable, List, Any
import seaborn as sns
from scipy import stats

def evaluate_conditional_sampling(
    generate_data_fn: Callable,
    sample_fn: Callable,
    denormalize_data: Callable,
    means: np.ndarray,
    stds: np.ndarray,
    node_ids: np.ndarray,
    iterations: int = 500,
    num_samples: int = 1000,
    plot_results: bool = True
) -> Tuple[float, List[float], dict]:
    """
    Evaluates conditional sampling by analyzing left percentages and distributions.
    
    Args:
        generate_data_fn: Function to generate data points
        sample_fn: Function to sample from conditional distribution
        denormalize_data: Callable,
        means: Normalization means
        stds: Normalization standard deviations
        node_ids: Node IDs for sampling
        iterations: Number of iterations to run
        num_samples: Number of samples per iteration
        plot_results: Whether to generate plots
        
    Returns:
        average_left_percentage: Average percentage of samples left of true value
        left_percentages: List of left percentages for each iteration
        metrics: Dictionary containing additional metrics
    """
    left_percentages = []
    
    def process_iteration(i: int) -> float:
        # Generate point and get condition
        _, point = generate_data_fn(jrandom.PRNGKey(i), 1, means, stds)
        point = point[0]
        
        num_features = len(point)
        theta1 = point[0].ravel()
        
        # Set up conditioning
        condition_mask = jnp.array([0] + [1] * (num_features - 1))
        condition_value = jnp.array(point).reshape(1, num_features)
        
        # Sample from conditional
        samples_conditional = sample_fn(
            jrandom.PRNGKey(i),
            (num_samples,),
            node_ids,
            condition_mask=condition_mask,
            condition_value=condition_value
        )
        
        # Process samples
        samples_conditional_denorm = denormalize_data(samples_conditional, means, stds)
        samples_conditional_last = samples_conditional_denorm[:, -1, :]
        theta1_samples = samples_conditional_last[:, 0]
        theta_float = float(theta1)
        
        # Calculate left percentage
        left_percentage = np.sum(theta1_samples < theta_float) / len(theta1_samples) * 100
        return left_percentage, theta1_samples, theta_float

    # Run iterations
    all_samples = []
    all_thetas = []
    for i in range(iterations):
        left_percentage, theta1_samples, theta_float = process_iteration(i)
        left_percentages.append(left_percentage)
        all_samples.append(theta1_samples)
        all_thetas.append(theta_float)
        
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{iterations} completed.")
    
    # Calculate metrics
    average_left_percentage = np.mean(left_percentages)
    metrics = calculate_metrics(left_percentages, all_samples, all_thetas)
    
    if plot_results:
        plot_evaluation_results(left_percentages, average_left_percentage, 
                              all_samples[-1], all_thetas[-1], iterations)
    
    return average_left_percentage, left_percentages, metrics

def calculate_metrics(left_percentages: List[float], 
                     all_samples: List[np.ndarray], 
                     all_thetas: List[float]) -> dict:
    """Calculate statistical metrics for evaluation."""
    metrics = {
        'mean_left_percentage': np.mean(left_percentages),
        'std_left_percentage': np.std(left_percentages),
        'median_left_percentage': np.median(left_percentages),
        'ks_statistic': stats.kstest(left_percentages, 'norm')[0],
        'ks_pvalue': stats.kstest(left_percentages, 'norm')[1]
    }
    return metrics

def plot_evaluation_results(left_percentages: List[float], 
                          average_left: float,
                          last_samples: np.ndarray,
                          last_theta: float,
                          iterations: int):
    """Generate evaluation plots."""
    # Distribution of left percentages
    plt.figure(figsize=(10, 6))
    plt.hist(left_percentages, bins=50, range=[0,100], alpha=0.7, 
             color='green', edgecolor='black')
    plt.axvline(x=average_left, color='red', linestyle='--', linewidth=2)
    plt.title(f"Distribution of Left Percentages ({iterations} Iterations)")
    plt.xlabel('Left Percentage Values')
    plt.ylabel('Frequency')
    plt.show()

    # QQ Plot
    plt.figure(figsize=(10, 6))
    stats.probplot(left_percentages, dist="norm", plot=plt)
    plt.title("QQ Plot vs Normal Distribution")
    plt.grid(True)
    plt.show()

    # CDF Plot
    plt.figure(figsize=(10, 6))
    sorted_vals = np.sort(left_percentages)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    plt.plot(sorted_vals, cdf*100, marker='.', linestyle='-', color='blue')
    plt.plot([0,100], [0,100])
    plt.title('CDF of Left Percentages')
    plt.xlabel('Left Percentage Values')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.show()

