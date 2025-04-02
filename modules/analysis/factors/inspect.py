"""
Influence Factors Inspection Module

This module provides functions to inspect and visualize the computed influence factors.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.types import ModuleName
from matplotlib import cm
from matplotlib.colors import Normalize
from datasets import load_dataset

# Import custom task for language modeling
from .task import LanguageModelingTask

logger = logging.getLogger(__name__)

def inspect_factors(config, layer_num=21):
    """
    Inspect the influence factors for a specific layer.
    
    Args:
        config: Configuration dictionary
        layer_num: Layer number to inspect (default: 21, last layer in TinyLlama)
    """
    factors_name = config['factors']['all_layers_name']
    output_dir = os.path.join(config['output']['influence_results'], f"layer_{layer_num}")
    clip_percentile = 99.5  # Clip extreme values for better visualization
    cmap = 'coolwarm'  # Color map for visualizations
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Inspecting factors for layer {layer_num}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create analyzer
    analyzer = Analyzer(
        analysis_name="influence_results",
        model=None,  # No need for model when just inspecting factors
        task=LanguageModelingTask(),
    )
    
    # Load factors
    try:
        factors = analyzer.load_factors(factors_name)
        logger.info(f"Factors loaded from {factors_name}")
    except Exception as e:
        logger.error(f"Error loading factors: {e}")
        raise
    
    # Construct module name for the specified layer
    module_name = ModuleName(f"model.layers.{layer_num}.mlp")
    logger.info(f"Inspecting module: {module_name}")
    
    # Check if the module exists in the factors
    if module_name not in factors:
        available_modules = list(factors.keys())
        logger.error(f"Module {module_name} not found in factors.")
        logger.error(f"Available modules: {available_modules}")
        raise ValueError(f"Module {module_name} not found in factors")
    
    # Get the factor data for the module
    module_factors = factors[module_name]
    
    # Get the lambda matrix and eigenvalues
    lambda_matrix = module_factors.lambdas
    eigenvalues = module_factors.fisher_eigenvalues
    
    # Visualize the lambda matrix
    visualize_lambda_matrix(lambda_matrix, output_dir, clip_percentile, cmap)
    
    # Visualize the eigenvalue distribution
    visualize_eigenvalues(eigenvalues, output_dir)
    
    logger.info(f"Factor inspection for layer {layer_num} completed.")
    logger.info(f"Results saved to {output_dir}")
    
    return output_dir

def visualize_lambda_matrix(lambda_matrix, output_dir, clip_percentile=99.5, cmap='coolwarm'):
    """
    Visualize the lambda matrix (covariance or precision matrix).
    
    Args:
        lambda_matrix: The lambda matrix to visualize
        output_dir: Directory to save the visualization
        clip_percentile: Percentile for clipping extreme values
        cmap: Color map for the visualization
    """
    # Convert to numpy for visualization
    if isinstance(lambda_matrix, torch.Tensor):
        lambda_np = lambda_matrix.cpu().numpy()
    else:
        lambda_np = np.array(lambda_matrix)
    
    # Clip extreme values for better visualization
    vmax = np.percentile(np.abs(lambda_np), clip_percentile)
    vmin = -vmax
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Normalize color map
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot heatmap
    plt.imshow(lambda_np, cmap=cmap, norm=norm)
    plt.colorbar(label='Value')
    plt.title('Lambda Matrix')
    plt.xlabel('Output dimension')
    plt.ylabel('Input dimension')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lambda_matrix.png'), dpi=300)
    plt.close()
    
    logger.info(f"Lambda matrix visualization saved to {output_dir}/lambda_matrix.png")

def visualize_eigenvalues(eigenvalues, output_dir, n_top=100):
    """
    Visualize the eigenvalue distribution.
    
    Args:
        eigenvalues: The eigenvalues to visualize
        output_dir: Directory to save the visualization
        n_top: Number of top eigenvalues to show in the detailed plot
    """
    # Convert to numpy for visualization
    if isinstance(eigenvalues, torch.Tensor):
        eig_np = eigenvalues.cpu().numpy()
    else:
        eig_np = np.array(eigenvalues)
    
    # Sort eigenvalues in descending order
    sorted_eig = np.sort(eig_np)[::-1]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot all eigenvalues
    ax1.semilogy(sorted_eig)
    ax1.set_title('All Eigenvalues (log scale)')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue (log scale)')
    ax1.grid(True, which="both", ls="--")
    
    # Plot top N eigenvalues
    top_n = min(n_top, len(sorted_eig))
    ax2.plot(sorted_eig[:top_n])
    ax2.set_title(f'Top {top_n} Eigenvalues (linear scale)')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Eigenvalue')
    ax2.grid(True, which="major", ls="--")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eigenvalue_distribution.png'), dpi=300)
    plt.close()
    
    logger.info(f"Eigenvalue distribution visualization saved to {output_dir}/eigenvalue_distribution.png")
    
    # Additional analysis - compute eigenvalue decay statistics
    total_eigval = np.sum(sorted_eig)
    cum_eigval = np.cumsum(sorted_eig) / total_eigval
    
    # Find how many eigenvalues capture 90%, 95%, 99% of the variance
    eigval_90 = np.argmax(cum_eigval >= 0.9) + 1
    eigval_95 = np.argmax(cum_eigval >= 0.95) + 1
    eigval_99 = np.argmax(cum_eigval >= 0.99) + 1
    
    # Plot cumulative explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(cum_eigval)
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of eigenvalues')
    plt.ylabel('Cumulative explained variance')
    plt.grid(True, which="major", ls="--")
    
    # Add lines for 90%, 95%, 99% thresholds
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.7)
    plt.axhline(y=0.95, color='g', linestyle='--', alpha=0.7)
    plt.axhline(y=0.99, color='b', linestyle='--', alpha=0.7)
    
    # Add text annotations
    plt.annotate(f'90%: {eigval_90} eigenvalues', 
                 xy=(eigval_90, 0.9), 
                 xytext=(eigval_90 + len(sorted_eig) // 20, 0.9),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10)
    
    plt.annotate(f'95%: {eigval_95} eigenvalues', 
                 xy=(eigval_95, 0.95), 
                 xytext=(eigval_95 + len(sorted_eig) // 20, 0.95),
                 arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10)
    
    plt.annotate(f'99%: {eigval_99} eigenvalues', 
                 xy=(eigval_99, 0.99), 
                 xytext=(eigval_99 + len(sorted_eig) // 20, 0.99),
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_variance.png'), dpi=300)
    plt.close()
    
    logger.info(f"Cumulative variance plot saved to {output_dir}/cumulative_variance.png")
    logger.info(f"Eigenvalue analysis: 90% variance: {eigval_90}, 95% variance: {eigval_95}, 99% variance: {eigval_99}")
    
    # Save a summary of the analysis
    with open(os.path.join(output_dir, 'eigenvalue_analysis.txt'), 'w') as f:
        f.write(f"Total number of eigenvalues: {len(sorted_eig)}\n")
        f.write(f"Largest eigenvalue: {sorted_eig[0]:.6f}\n")
        f.write(f"Smallest eigenvalue: {sorted_eig[-1]:.6f}\n")
        f.write(f"Mean eigenvalue: {np.mean(sorted_eig):.6f}\n")
        f.write(f"Median eigenvalue: {np.median(sorted_eig):.6f}\n")
        f.write(f"90% variance captured by {eigval_90} eigenvalues ({eigval_90 / len(sorted_eig) * 100:.2f}%)\n")
        f.write(f"95% variance captured by {eigval_95} eigenvalues ({eigval_95 / len(sorted_eig) * 100:.2f}%)\n")
        f.write(f"99% variance captured by {eigval_99} eigenvalues ({eigval_99 / len(sorted_eig) * 100:.2f}%)\n")
    
    logger.info(f"Eigenvalue analysis summary written to {output_dir}/eigenvalue_analysis.txt") 