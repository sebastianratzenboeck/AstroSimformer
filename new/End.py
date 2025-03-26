import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import jax.random as jrandom
import jax.numpy as jnp
from typing import Tuple, Callable, List, Any
from scipy import stats


def bootstrap_sampling(data: np.ndarray, num_samples: int = 1000, confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Perform Bootstrap sampling to calculate the mean and confidence interval.
    
    Args:
        data: Array of data points
        num_samples: Number of bootstrap samples
        confidence_level: Confidence level for the interval
        
    Returns:
        mean: Mean of the data
        lower_bound: Lower bound of the confidence interval
        upper_bound: Upper bound of the confidence interval
    """
    means = []
    n = len(data)
    
    for _ in range(num_samples):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    
    mean = np.mean(means)
    lower_bound = np.percentile(means, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(means, (1 + confidence_level) / 2 * 100)
    
    return mean, lower_bound, upper_bound

def denormalize_data_point(normalized_data: jnp.ndarray, means: jnp.ndarray, 
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
    means_broadcast = means.reshape(1, -1, 1)  # Shape: (1, 5, 1)
    stds_broadcast = stds.reshape(1, -1, 1)    # Shape: (1, 5, 1)
    
    denormalized = normalized_data * stds_broadcast + means_broadcast
    
    return denormalized

def plot_kde_with_samples(theta_samples, real_theta, bandwidth='scott'):
    """
    Plots a histogram of sampled thetas and overlays the KDE plot.
    
    Args:
        theta_samples: Array of sampled theta values
        real_theta: The true value of theta to overlay
        bandwidth: Bandwidth parameter for KDE, default is 'scott'
    """
    # Fit KDE to the theta samples
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(theta_samples[:, np.newaxis])
    
    # Generate values for the x-axis (range around the real theta for better visualization)
    x_values = np.linspace(np.min(theta_samples), np.max(theta_samples), 1000)[:, np.newaxis]
    
    # Evaluate the KDE on the range of values
    kde_values = np.exp(kde.score_samples(x_values))
    
    # Plot the histogram of the theta samples
    plt.figure(figsize=(10, 6))
    plt.hist(theta_samples, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black', label='Sampled theta')
    
    # Overlay the KDE plot
    plt.plot(x_values, kde_values, color='red', linewidth=2, label='KDE (Density Estimation)')
    
    # Mark the true theta with a vertical line
    plt.axvline(real_theta, color='green', linestyle='--', linewidth=2, label=f'Real theta: {real_theta:.3f}')
    
    # Add labels and legend
    plt.title('KDE of Sampled Theta Values with Real Theta', fontsize=14)
    plt.xlabel('Theta Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.show()

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
    all_samples = []
    all_thetas = []
    all_points_orginal = []
    kde_bandwidths = []
    kde_scores = []

    def process_iteration(i: int) -> float:
        # Generate point and get condition
        _, point = generate_data_fn(jrandom.PRNGKey(i), 1, means, stds)
        point = point[0]
        num_features = len(point)
        point_original = point
        new_point=point
        new_point_theta = denormalize_data_point(new_point, means, stds)
        new_point_theta = new_point_theta[0].flatten()
        theta1 = float(new_point_theta[0])
        
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

        # KDE Metric Calculation
        # Fit KDE to the sampled thetas
        kde = KernelDensity(kernel='gaussian', bandwidth='silverman')
        kde.fit(theta1_samples.reshape(-1, 1))
        
        # Evaluate the KDE at the real theta value (theta_float)
        log_prob = kde.score_samples(np.array([[theta_float]]))
        log_prob = np.sum(log_prob)  # Sum of log probabilities

        # Store KDE bandwidth for later
        kde_bandwidths.append(kde.bandwidth)
        if (i + 1) % 100 == 0:
            if extended_plot:
                point_new = denormalize_data_point(point_original, means, stds)
                point_new_plot = point_new[0]
                plot_kde_with_samples(theta1_samples, theta_float)

        return left_percentage, theta1_samples, theta_float, point_original, log_prob
    
    # Run iterations
    for i in range(iterations):
        left_percentage, theta1_samples, theta_float, point_original, log_prob = process_iteration(i)
        left_percentages.append(left_percentage)
        all_samples.append(theta1_samples)
        all_thetas.append(theta_float)
        all_points_orginal.append(point_original)
        print(f"Iteration {i+1}/{iterations} - Metric (Log KDE): {log_prob}")
        kde_scores.append(log_prob)
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{iterations} completed.")
    
    # Calculate metrics
    average_left_percentage = np.mean(left_percentages)

    if plot_results:
        plot_evaluation_results(left_percentages, average_left_percentage, 
                              all_samples[-1], all_thetas[-1], iterations)
            # Calculate metrics
    average_left_percentage, lower_bound, upper_bound = bootstrap_sampling(np.array(left_percentages))
    
    print(f"Average Left Percentage: {average_left_percentage:.2f}%")
    print(f"95% Confidence Interval: [{lower_bound:.2f}%, {upper_bound:.2f}%]")
    return average_left_percentage, left_percentages



def plot_evaluation_results(left_percentages: List[float], 
                          average_left: float,
                          last_samples: np.ndarray,
                          last_theta: float,
                          iterations: int):
    """Generate evaluation plots."""
    plt.figure(figsize=(10, 6))
    plt.hist(left_percentages, bins=50, range=[0, 100], alpha=0.7, 
             color='green', edgecolor='black')
    plt.axvline(x=average_left, color='red', linestyle='--', linewidth=2)
    plt.title(f"Distribution of Left Percentages ({iterations} Iterations)")
    plt.xlabel('Left Percentage Values')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10, 6))
    stats.probplot(left_percentages, dist="norm", plot=plt)
    plt.title("QQ Plot vs Normal Distribution")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    sorted_vals = np.sort(left_percentages)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    plt.plot(sorted_vals, cdf*100, marker='.', linestyle='-', color='blue')
    plt.plot([0, 100], [0, 100])
    plt.title('CDF of Left Percentages')
    plt.xlabel('Left Percentage Values')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.show()

    # Calculate the absolute difference sum between the CDF and the plotted line
    abs_diff_sum = np.sum(np.abs(cdf*100 - sorted_vals))
    avg_abs_diff = abs_diff_sum / len(sorted_vals)
    print(f"Average Absolute Difference CDF: {avg_abs_diff:.2f}")


def plot_pairplot_comp_kde2_with_lines(samples1, samples2, feature_labels, real_point, title="Pair Plot of Features", bins=150):
    """
    Plots pairwise relationships with consistent scaling and styling.
    """
    df1 = pd.DataFrame(samples1, columns=feature_labels)
    df2 = pd.DataFrame(samples2, columns=feature_labels)
    real_point = np.array(real_point).ravel()
    n_features = len(feature_labels)
    
    # Calculate global min/max for each feature
    feature_ranges = {}
    for feature in feature_labels:
        min_val = min(df1[feature].min(), df2[feature].min())
        max_val = max(df1[feature].max(), df2[feature].max())
        feature_ranges[feature] = (min_val, max_val)
    
    # Create figure with consistent spacing
    fig, axes = plt.subplots(n_features, n_features, 
                            figsize=(4 * n_features, 4 * n_features),
                            gridspec_kw={'wspace': 0.2, 'hspace': 0.2})
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    
    # Use consistent colormap
    cmap = plt.cm.viridis
    
    for i, feature1 in enumerate(feature_labels):
        for j, feature2 in enumerate(feature_labels):
            ax = axes[i, j]
            
            # Set consistent axis limits
            ax.set_xlim(feature_ranges[feature2])
            ax.set_ylim(feature_ranges[feature1])
            
            if i == j:
                # Diagonal: Histograms
                bins_range = np.linspace(*feature_ranges[feature1], bins//2)
                ax.hist(df1[feature1], bins=bins_range, density=True, 
                       alpha=0.5, color='blue', label='Dataset 1', histtype='step', lw=2)
                ax.hist(df2[feature1], bins=bins_range, density=True, 
                       alpha=0.5, color='red', label='Dataset 2', histtype='step', lw=2)
                ax.axvline(real_point[i], color='black', linestyle='--', 
                          linewidth=1.5, label='Real Point')
                
                max_density = max(
                    np.histogram(df1[feature1], bins=bins_range, density=True)[0].max(),
                    np.histogram(df2[feature1], bins=bins_range, density=True)[0].max()
                )
                ax.set_ylim(0, max_density * 1.1)
                
            elif i < j:
                # Upper triangle: Dataset 1
                x1, y1 = df1[feature2], df1[feature1]
                h1 = np.histogram2d(x1, y1, bins=bins, 
                                  range=[feature_ranges[feature2], feature_ranges[feature1]])[0]
                ax.imshow(np.log1p(h1.T), extent=[*feature_ranges[feature2], *feature_ranges[feature1]], 
                         origin='lower', aspect='auto', cmap=cmap)
                ax.axhline(real_point[i], color='red', linestyle='--', linewidth=1.5)
                ax.axvline(real_point[j], color='red', linestyle='--', linewidth=1.5)
                
            elif i > j:
                # Lower triangle: Dataset 2
                x2, y2 = df2[feature2], df2[feature1]
                h2 = np.histogram2d(x2, y2, bins=bins,
                                  range=[feature_ranges[feature2], feature_ranges[feature1]])[0]
                ax.imshow(np.log1p(h2.T), extent=[*feature_ranges[feature2], *feature_ranges[feature1]], 
                         origin='lower', aspect='auto', cmap=cmap)
                ax.axhline(real_point[i], color='red', linestyle='--', linewidth=1.5)
                ax.axvline(real_point[j], color='red', linestyle='--', linewidth=1.5)
            
            # Consistent font sizes and tick parameters
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            if i == n_features - 1:
                ax.set_xlabel(feature2, fontsize=12)
            if j == 0:
                ax.set_ylabel(feature1, fontsize=12)
            
            if i < n_features - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])

    # Adjust layout and legend
    plt.tight_layout()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5), 
              fontsize=12, borderaxespad=0.)
    
    plt.show()
    plt.close()
