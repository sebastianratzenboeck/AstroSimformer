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
    plot_results: bool = True,
    extended_plot: bool = False,
    samples_new = None,
    data_new = None,
) -> Tuple[float, List[float], dict]:
    """
    Evaluates conditional sampling by analyzing left percentages and distributions.
    
    Args:
        generate_data_fn: Function to generate data points
        sample_fn: Function to sample from conditional distribution
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
        point_original = point
        print(point_original.shape)
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
        return left_percentage, theta1_samples, theta_float,point_original

    # Run iterations
    all_samples = []
    all_thetas = []
    all_points_orginal = []
    for i in range(iterations):
        left_percentage, theta1_samples, theta_float,point_original = process_iteration(i)
        left_percentages.append(left_percentage)
        all_samples.append(theta1_samples)
        all_thetas.append(theta_float)
        all_points_orginal.append(point_original)
        print(added point)
        
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{iterations} completed.")
            if extended_plot:
                point_new = denormalize_data(all_points_orginal[i], means, stds)
                print(all_points_orginal[i].shape)
                point_new_plot = point_new[0]
                print(point_new.shape)
                print(point_new_plot.shape)
                plot_pairplot_comp_kde2_with_lines(
                    samples_new,  # Dataset 1
                    data_new,  # Dataset 2
                    feature_labels=[f"Feature {i+1}" for i in range(point_new_plot.shape[0])],
                    real_point=point_new_plot,  # The real point used in the iteration
                    title="Pair Plot with Highlighted Real Point"
                )
                
    
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_pairplot_comp_kde2_with_lines(samples1, samples2, feature_labels, real_point, title="Pair Plot of Features", bins=150):
    """
    Plots pairwise relationships with density and KDE for two datasets and highlights the real point using lines.
    
    Parameters:
    - samples1: Dataset 1 (array-like)
    - samples2: Dataset 2 (array-like)
    - feature_labels: List of feature names (list of str)
    - real_point: Real point to highlight (array-like, same dimension as feature_labels)
    - title: Plot title (str)
    - bins: Number of bins for histograms (int)
    """
    df1 = pd.DataFrame(samples1, columns=feature_labels)
    df2 = pd.DataFrame(samples2, columns=feature_labels)
    print(real_point.shape)
    real_point = np.array(real_point).ravel()  # Ensure the real_point is 1D
    print(real_point)
    n_features = len(feature_labels)
    
    fig, axes = plt.subplots(n_features, n_features, 
                              figsize=(4 * n_features, 4 * n_features), 
                              gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, feature1 in enumerate(feature_labels):
        for j, feature2 in enumerate(feature_labels):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: Histograms
                ax.hist(df1[feature1], bins=bins, density=True, 
                        alpha=0.5, edgecolor='white', label='Dataset 1', color='blue', linewidth=1.5)
                ax.hist(df2[feature1], bins=bins, density=True, 
                        alpha=0.5, edgecolor='white', label='Dataset 2', color='red', linewidth=1.5)
                
                # Highlight real point on the diagonal
                ax.axvline(real_point[i], color='red', linestyle='--', linewidth=1.5, label='Real Point')

                ax.set_ylabel("Density", fontsize=12)
                ax.set_xlabel(feature1, fontsize=12)
                ax.set_title(f"Density Comparison for {feature1}", fontsize=14)
            elif i < j:
                # Upper triangle: Dataset 1's 2D density plot
                x1 = df1[feature2]
                y1 = df1[feature1]
                
                h1, x_edges, y_edges = np.histogram2d(x1, y1, bins=bins)
                h1 = h1.T
                h1_log = np.log1p(h1)
                
                ax.imshow(h1_log, extent=[x_edges[0], x_edges[-1], 
                                          y_edges[0], y_edges[-1]], 
                          origin='lower', aspect='auto', alpha=0.8)
                
                # Highlight real point with lines
                ax.axhline(real_point[i], color='red', linestyle='--', linewidth=1.5)
                ax.axvline(real_point[j], color='red', linestyle='--', linewidth=1.5)
            elif i > j:
                # Lower triangle: Dataset 2's 2D density plot
                x2 = df2[feature2]
                y2 = df2[feature1]
                
                h2, x_edges, y_edges = np.histogram2d(x2, y2, bins=bins)
                h2 = h2.T
                h2_log = np.log1p(h2)
                
                ax.imshow(h2_log, extent=[y_edges[0], y_edges[-1], 
                                          x_edges[0], x_edges[-1]], 
                          origin='lower', aspect='auto', alpha=0.8)
                
                # Highlight real point with lines
                ax.axhline(real_point[i], color='red', linestyle='--', linewidth=1.5)
                ax.axvline(real_point[j], color='red', linestyle='--', linewidth=1.5)
                
            
            # Refine tick and label styling
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            if i == n_features - 1:
                ax.set_xlabel(feature2, fontsize=10, fontweight='bold')
            if j == 0:
                ax.set_ylabel(feature1, fontsize=10, fontweight='bold')
            
            if i < n_features - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])

    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.5)
    plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.1, 1.1))
    plt.show()
    plt.close()
