import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import jax.numpy as jnp
from sbi.analysis import pairplot


def plot_pairplot_kde(samples, feature_labels, title="Pair Plot of Features", bins=100):
    """
    Creates a detailed pairplot with density plots using Matplotlib.
    
    Parameters:
    samples (numpy.ndarray): Array of sample data
    feature_labels (list): Labels for features
    title (str): Plot title
    bins (int): Number of bins for density estimation
    """
    # Ensure input samples are a DataFrame
    df = pd.DataFrame(samples, columns=feature_labels)
    
    # Number of features
    n_features = len(feature_labels)

    
    # Initialize the Matplotlib figure with improved spacing
    fig, axes = plt.subplots(n_features, n_features, 
                              figsize=(4 * n_features, 4 * n_features), 
                              gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    fig.suptitle(title, fontsize=16, fontweight='bold')
    

    
    # Plot pairwise relationships
    for i, feature1 in enumerate(feature_labels):
        for j, feature2 in enumerate(feature_labels):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: Kernel Density Estimate
                ax.hist(df[feature1], bins=bins, density=True, 
                        alpha=0.7, edgecolor='white')
                ax.set_ylabel("Density", fontsize=10)
            else:
                # Off-diagonal: 2D Density Estimation
                x = df[feature2]
                y = df[feature1]
                
                # Enhanced 2D histogram for density
                h, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
                h = h.T  # Transpose to match imshow convention
                
                # Log transformation for better visualization of density variations
                h_log = np.log1p(h)
                
                # Plot heatmap with improved color mapping
                im = ax.imshow(h_log, extent=[x_edges[0], x_edges[-1], 
                                              y_edges[0], y_edges[-1]], 
                               origin='lower', aspect='auto', alpha=0.8)
            
            # Refine tick and label styling
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Set labels for outer plots only
            if i == n_features - 1:
                ax.set_xlabel(feature2, fontsize=10, fontweight='bold')
            if j == 0:
                ax.set_ylabel(feature1, fontsize=10, fontweight='bold')
            
            # Hide inner tick labels
            if i < n_features - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
    
    # Adjust layout with more precise spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.5)
    plt.show()
    plt.close()  # Close the plot to free memory



def plot_pairplot_comp_kde(samples1, samples2, feature_labels, title="Pair Plot of Features", bins=150):
    # Ensure input samples are a DataFrame
    df1 = pd.DataFrame(samples1, columns=feature_labels)
    df2 = pd.DataFrame(samples2, columns=feature_labels)
    
    # Number of features
    n_features = len(feature_labels)
    
    # Initialize the Matplotlib figure with improved spacing
    fig, axes = plt.subplots(n_features, n_features, 
                              figsize=(4 * n_features, 4 * n_features), 
                              gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot pairwise relationships
    for i, feature1 in enumerate(feature_labels):
        for j, feature2 in enumerate(feature_labels):
            ax = axes[i, j]
            
            if i == j:
                # Dataset 1
                ax.hist(df1[feature1], bins=bins, density=True, 
                        alpha=0.5, edgecolor='white', label='Dataset 1', color='blue', linewidth=1.5)

                # Dataset 2
                ax.hist(df2[feature1], bins=bins, density=True, 
                        alpha=0.5, edgecolor='white', label='Dataset 2', color='red', linewidth=1.5)

                # Adding labels and title
                ax.set_ylabel("Density", fontsize=12)
                ax.set_xlabel(feature1, fontsize=12)
                ax.set_title(f"Density Comparison for {feature1}", fontsize=14)
            elif i < j:
                # Upper triangle: Plot Dataset 1's 2D density plot
                x1 = df1[feature2]
                y1 = df1[feature1]
                
                # Enhanced 2D histogram for density for Dataset 1
                h1, x_edges, y_edges = np.histogram2d(x1, y1, bins=bins)
                h1 = h1.T  # Transpose to match imshow convention
                
                # Log transformation for better visualization of density variations
                h1_log = np.log1p(h1)
                
                # Plot heatmap for Dataset 1 (upper triangle)
                ax.imshow(h1_log, extent=[x_edges[0], x_edges[-1], 
                                          y_edges[0], y_edges[-1]], 
                          origin='lower', aspect='auto', alpha=0.8)
                
            elif i > j:
                # Lower triangle: Plot Dataset 2's 2D density plot
                x2 = df2[feature2]
                y2 = df2[feature1]
                
                # Enhanced 2D histogram for density for Dataset 2
                h2, x_edges, y_edges = np.histogram2d(x2, y2, bins=bins)
                h2 = h2.T  # Transpose to match imshow convention
                
                # Log transformation for better visualization of density variations
                h2_log = np.log1p(h2)
                
                # Plot heatmap for Dataset 2 (lower triangle)
                ax.imshow(h2_log, extent=[x_edges[0], x_edges[-1], 
                                          y_edges[0], y_edges[-1]], 
                          origin='lower', aspect='auto', alpha=0.8)
            
            # Refine tick and label styling
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Set labels for outer plots only
            if i == n_features - 1:
                ax.set_xlabel(feature2, fontsize=10, fontweight='bold')
            if j == 0:
                ax.set_ylabel(feature1, fontsize=10, fontweight='bold')
            
            # Hide inner tick labels
            if i < n_features - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])

    # Adjust layout with more precise spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.5)

    plt.show()
    plt.close()  # Close the plot to free memory


def plot_pairplot_comp_kde2(samples1, samples2, feature_labels, title="Pair Plot of Features", bins=150):
    """
    Function to generate a pair plot comparing two datasets (samples1 and samples2) with consistent axis scaling.
    
    Args:
        samples1 (array-like): First dataset to plot.
        samples2 (array-like): Second dataset to plot.
        feature_labels (list of str): Feature names to label axes.
        title (str): Title for the plot.
        bins (int): Number of bins for density and histogram plots.
    """
    # Ensure input samples are a DataFrame
    df1 = pd.DataFrame(samples1, columns=feature_labels)
    df2 = pd.DataFrame(samples2, columns=feature_labels)
    
    # Number of features
    n_features = len(feature_labels)
    
    # Determine global min and max for consistent axis scaling
    global_min = df1.min().min(), df2.min().min()
    global_max = df1.max().max(), df2.max().max()
    axis_limits = (min(global_min), max(global_max))
    
    # Initialize the Matplotlib figure with improved spacing
    fig, axes = plt.subplots(n_features, n_features, 
                              figsize=(4 * n_features, 4 * n_features), 
                              gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot pairwise relationships
    for i, feature1 in enumerate(feature_labels):
        for j, feature2 in enumerate(feature_labels):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: Plot density comparison (histograms)
                ax.hist(df1[feature1], bins=bins, density=True, 
                        alpha=0.5, edgecolor='white', label='Dataset 1', color='blue', linewidth=1.5)
                ax.hist(df2[feature1], bins=bins, density=True, 
                        alpha=0.5, edgecolor='white', label='Dataset 2', color='red', linewidth=1.5)

                ax.set_xlim(axis_limits)
                ax.set_ylabel("Density", fontsize=12)
                ax.set_xlabel(feature1, fontsize=12)
                ax.set_title(f"Density Comparison for {feature1}", fontsize=14)
            elif i < j:
                # Upper triangle: Dataset 1's 2D density plot
                x1 = df1[feature2]
                y1 = df1[feature1]
                
                h1, x_edges, y_edges = np.histogram2d(x1, y1, bins=bins, range=[axis_limits, axis_limits])
                h1 = h1.T
                
                h1_log = np.log1p(h1)
                
                ax.imshow(h1_log, extent=[axis_limits[0], axis_limits[1], 
                                          axis_limits[0], axis_limits[1]], 
                          origin='lower', aspect='auto', alpha=0.8)
            elif i > j:
                # Lower triangle: Dataset 2's 2D density plot, rotated 90 degrees left
                x2 = df2[feature1]
                y2 = df2[feature2]
                
                h2, x_edges, y_edges = np.histogram2d(x2, y2, bins=bins, range=[axis_limits, axis_limits])
                h2 = h2.T
                
                h2_log = np.log1p(h2)
                
                # Rotate plot by switching x and y axes
                ax.imshow(h2_log, extent=[axis_limits[0], axis_limits[1], 
                                          axis_limits[0], axis_limits[1]], 
                          origin='lower', aspect='auto', alpha=0.8)
            
            # Refine tick and label styling
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Set labels for outer plots only
            if i == n_features - 1:
                ax.set_xlabel(feature2, fontsize=10, fontweight='bold')
            if j == 0:
                ax.set_ylabel(feature1, fontsize=10, fontweight='bold')
            
            # Hide inner tick labels
            if i < n_features - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])

    # Adjust layout with more precise spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.5)

    plt.show()
    plt.close()  # Close the plot to free memory


def plot_pairplot_comp_kde3(samples1, samples2, feature_labels, title="Pair Plot of Features", bins=150):
    """
    This function creates a pair plot comparing two datasets using KDE and density heatmaps.
    
    Parameters:
    - samples1: Dataset 1 (2D array-like or DataFrame)
    - samples2: Dataset 2 (2D array-like or DataFrame)
    - feature_labels: List of feature names (strings)
    - title: Title of the plot
    - bins: Number of bins for histograms and 2D histograms
    """
    # Convert inputs to DataFrame
    df1 = pd.DataFrame(samples1, columns=feature_labels)
    df2 = pd.DataFrame(samples2, columns=feature_labels)
    
    # Number of features
    n_features = len(feature_labels)
    
    # Initialize the plot
    fig, axes = plt.subplots(n_features, n_features, 
                              figsize=(4 * n_features, 4 * n_features), 
                              gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Pairwise plots
    for i, feature1 in enumerate(feature_labels):
        for j, feature2 in enumerate(feature_labels):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: 1D KDE for each dataset
                ax.hist(df1[feature1], bins=bins, density=True, 
                        alpha=0.5, edgecolor='white', label='Dataset 1', color='blue')
                ax.hist(df2[feature1], bins=bins, density=True, 
                        alpha=0.5, edgecolor='white', label='Dataset 2', color='red')
                ax.set_title(f"{feature1}", fontsize=12)
                if i == 0:
                    ax.legend(fontsize=8)
            
            elif i < j:
                # Upper triangle: Dataset 1's 2D density
                x1, y1 = df1[feature2], df1[feature1]
                h1, x_edges, y_edges = np.histogram2d(x1, y1, bins=bins)
                ax.imshow(np.log1p(h1.T), extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], 
                          origin='lower', aspect='auto', cmap='Blues', alpha=0.8)
            
            elif i > j:
                # Lower triangle: Dataset 2's 2D density
                x2, y2 = df2[feature2], df2[feature1]
                h2, x_edges, y_edges = np.histogram2d(x2, y2, bins=bins)
                ax.imshow(np.log1p(h2.T), extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], 
                          origin='lower', aspect='auto', cmap='Reds', alpha=0.8)
            
            # Format the plot
            ax.tick_params(axis='both', which='major', labelsize=8)
            if j > 0: ax.set_yticks([])
            if i < n_features - 1: ax.set_xticks([])

    # Adjust spacing and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def replace_nan_with_min(data):
    """
    Replaces NaN values in a JAX array with the minimum non-NaN value from the entire dataset.

    Args:
        data (jnp.ndarray): Input data array with shape (samples, features).

    Returns:
        jnp.ndarray: Data array with NaN values replaced by the minimum value.
    """
    # Compute the minimum value while ignoring NaNs
    min_value = jnp.nanmin(data)
    
    # Create a mask where True represents NaN values
    nan_mask = jnp.isnan(data)
    
    # Replace NaN values with the minimum value
    data_filled = jnp.where(nan_mask, min_value, data)
    
    return data_filled


# Updated pairplot function using Matplotlib
def plot_pairplot_math(samples, feature_labels, title="Pair Plot of Features"):
    """
    Creates and saves a pairplot for a single dataset using Matplotlib.

    Parameters:
        samples (numpy.ndarray): Array of sample data with shape (n_samples, n_features).
        feature_labels (list): List of labels for the features.
        title (str): Title for the plot.
    """
    # Ensure input samples are a DataFrame
    df = pd.DataFrame(samples, columns=feature_labels)
    
    # Number of features
    n_features = len(feature_labels)
    
    
    # Initialize the Matplotlib figure
    fig, axes = plt.subplots(n_features, n_features, figsize=(3 * n_features, 3 * n_features))
    fig.suptitle(title, fontsize=16)
    
    # Plot pairwise relationships
    for i, feature1 in enumerate(feature_labels):
        for j, feature2 in enumerate(feature_labels):
            ax = axes[i, j]
            if i == j:
                # Diagonal: Kernel Density Estimate (KDE)
                ax.hist(df[feature1], bins=30, alpha=0.7, color='skyblue', density=True)
                ax.set_ylabel("Density", fontsize=10)
            else:
                # Off-diagonal: Scatter plots
                ax.scatter(df[feature2], df[feature1], alpha=0.6, s=10, color='steelblue')
            
            # Set labels for outer plots only
            if i == n_features - 1:
                ax.set_xlabel(feature2, fontsize=10)
            if j == 0:
                ax.set_ylabel(feature1, fontsize=10)
            
            # Hide inner tick labels
            if i < n_features - 1:
                ax.xaxis.set_visible(False)
            if j > 0:
                ax.yaxis.set_visible(False)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.close()  # Close the plot to free memory


# Pairplot comparison function
def plot_pairplot_comparison_seb_math(samples1, samples2, feature_labels, dataset_labels=("Dataset 1", "Dataset 2"), title="Pair Plot Comparison of Features"):
    """
    Creates and displays a pairplot comparing two datasets using Matplotlib, plotting NaN values on the axis.

    Parameters:
        samples1 (numpy.ndarray): Array of sample data for the first dataset with shape (n_samples, n_features).
        samples2 (numpy.ndarray): Array of sample data for the second dataset with shape (n_samples, n_features).
        feature_labels (list): List of labels for the features.
        dataset_labels (tuple): Tuple of two strings representing the dataset labels.
        title (str): Title for the plot.
    """
    # Convert both datasets to DataFrames
    df1 = pd.DataFrame(samples1, columns=feature_labels)
    df2 = pd.DataFrame(samples2, columns=feature_labels)
    
    # Number of features
    n_features = len(feature_labels)

    # Initialize the Matplotlib figure
    fig, axes = plt.subplots(n_features, n_features, figsize=(3 * n_features, 3 * n_features))
    fig.suptitle(title, fontsize=16)
    
    # Define colors for datasets
    colors = ['steelblue', 'darkorange']
    datasets = [df1, df2]
    labels = dataset_labels
    
    # Plot pairwise relationships
    for i, feature1 in enumerate(feature_labels):
        for j, feature2 in enumerate(feature_labels):
            ax = axes[i, j]
            if i == j:
                # Diagonal: Histogram for each dataset
                for dataset, color, label in zip(datasets, colors, labels):
                    feature_data = dataset[feature1]
                    ax.hist(feature_data.dropna(), bins=30, alpha=0.5, label=label, color=color, density=True)
                ax.legend(fontsize=8, loc='upper right')
            else:
                # Off-diagonal: Scatter plots for each dataset
                for dataset, color, label in zip(datasets, colors, labels):
                    # Extract data
                    x_data = dataset[feature2]
                    y_data = dataset[feature1]
                    
                    # Determine axis positions for NaN values
                    x_axis_pos = ax.get_xlim()[0]  # Minimum x-axis value
                    y_axis_pos = ax.get_ylim()[0]  # Minimum y-axis value
                    
                    # Replace NaN values with axis positions
                    x_data = x_data.fillna(x_axis_pos)
                    y_data = y_data.fillna(y_axis_pos)
                    
                    # Plot scatter points
                    ax.scatter(x_data, y_data, alpha=0.6, s=10, color=color, label=label if i == 0 and j == 1 else None)
            
            # Set labels for outer plots only
            if i == n_features - 1:
                ax.set_xlabel(feature2, fontsize=10)
            if j == 0:
                ax.set_ylabel(feature1, fontsize=10)
            
            # Hide inner tick labels
            if i < n_features - 1:
                ax.xaxis.set_visible(False)
            if j > 0:
                ax.yaxis.set_visible(False)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.close()  # Close the plot to free memory


# Pairplot comparison function
def plot_pairplot_comparison_points(
        samples1, samples2, feature_labels, dataset_labels=("Dataset 1", "Dataset 2"), title="Pair Plot Comparison of Features"
):
    """
    Creates and saves a pairplot comparing two datasets using Matplotlib.

    Parameters:
        samples1 (numpy.ndarray): Array of sample data for the first dataset with shape (n_samples, n_features).
        samples2 (numpy.ndarray): Array of sample data for the second dataset with shape (n_samples, n_features).
        feature_labels (list): List of labels for the features.
        dataset_labels (tuple): Tuple of two strings representing the dataset labels.
        title (str): Title for the plot.
    """
    # Convert both datasets to DataFrames
    df1 = pd.DataFrame(samples1, columns=feature_labels)
    df2 = pd.DataFrame(samples2, columns=feature_labels)
    
    # Number of features
    n_features = len(feature_labels)
    
    # Initialize the Matplotlib figure
    fig, axes = plt.subplots(n_features, n_features, figsize=(3 * n_features, 3 * n_features))
    fig.suptitle(title, fontsize=16)
    
    # Define colors for datasets
    colors = ['steelblue', 'darkorange']
    datasets = [df1, df2]
    labels = dataset_labels
    
    # Plot pairwise relationships
    for i, feature1 in enumerate(feature_labels):
        for j, feature2 in enumerate(feature_labels):
            ax = axes[i, j]
            if i == j:
                # Diagonal: Histogram for each dataset
                for dataset, color, label in zip(datasets, colors, labels):
                    ax.hist(dataset[feature1], bins=30, alpha=0.5, label=label, color=color, density=True)
                ax.legend(fontsize=8, loc='upper right')
            else:
                # Off-diagonal: Scatter plots for each dataset
                for k, (dataset, color, label) in enumerate(zip(datasets, colors, labels)):
                    zorder = k if i > j else 2-k  # Adjust zorder based on position
                    ax.scatter(
                        dataset[feature2], dataset[feature1], alpha=0.2, s=5,
                        color=color, label=label if i == 0 and j == 1 else None,
                        zorder=zorder
                    )
            
            # Set labels for outer plots only
            if i == n_features - 1:
                ax.set_xlabel(feature2, fontsize=10)
            if j == 0:
                ax.set_ylabel(feature1, fontsize=10)
            
            # Hide inner tick labels
            if i < n_features - 1:
                ax.xaxis.set_visible(False)
            if j > 0:
                ax.yaxis.set_visible(False)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.close()  # Close the plot to free memory


def plot_pairplot_comparison_seb_sns(samples1, samples2, feature_labels, dataset_labels=("Dataset 1", "Dataset 2"), title="Pair Plot Comparison of Features"):
    """
    Creates and saves a pairplot comparing two datasets.

    Parameters:
        samples1 (numpy.ndarray): Array of sample data for the first dataset with shape (n_samples, n_features). 
        samples2 (numpy.ndarray): Array of sample data for the second dataset with shape (n_samples, n_features).Replace nan with min
        feature_labels (list): List of labels for the features.
        dataset_labels (tuple): Tuple of two strings representing the dataset labels.
        title (str): Title for the plot.
    """

    samples2 = replace_nan_with_min(samples2)
    # Convert both samples to DataFrames and add dataset labels
    df1 = pd.DataFrame(samples1, columns=feature_labels)
    df1["Dataset"] = dataset_labels[0]

    df2 = pd.DataFrame(samples2, columns=feature_labels)
    df2["Dataset"] = dataset_labels[1]

    # Concatenate dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Create the pairplot
    sns.pairplot(combined_df, hue="Dataset", diag_kind='kde', height=2.5)
    plt.suptitle(title, size=16, y=1.02)

    # Save the plot
    plt.show()  # Display the plot


def plot_sbi_pairplot(samples, feature_labels, title="SBI Pair Plot of Features"):
    """
    Creates a pairplot for a single dataset using the sbi library.
    
    Parameters:
        samples (numpy.ndarray): Array of sample data with shape (n_samples, n_features).
        feature_labels (list): List of labels for the features.
        title (str): Title for the plot.
    """
    # Ensure the samples are a 2D array
    samples = np.array(samples)

    # Use sbi's pairplot to create the distribution plots
    pairplot(samples, labels=feature_labels, figsize=(12, 12))
    plt.suptitle(title, size=16, y=1.02)
    plt.show()  # Display the plot
    plt.close()  # Close plot to free memory

