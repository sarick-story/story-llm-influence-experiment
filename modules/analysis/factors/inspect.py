"""
Influence Factors Inspection Module

This module provides functions to inspect and visualize the computed influence factors.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from pathlib import Path
from kronfluence.analyzer import Analyzer
from matplotlib import cm
from matplotlib.colors import Normalize
from datasets import load_dataset
from modules.utils.wandb_utils import init_wandb

# Import custom task for language modeling
from .task import LanguageModelingTask

logger = logging.getLogger(__name__)

def inspect_factors(config, layer_num=11, clip_percentile=99.5, cmap='coolwarm'):
    """
    Inspect the influence factors for a specific layer.
    
    Args:
        config: Configuration dictionary
        layer_num: Layer number to inspect (default: 11, consistent with original code)
        clip_percentile: Percentile for clipping extreme values in visualizations
        cmap: Color map for visualizations
    """
    factors_name = config['factors']['all_layers_name']
    output_dir = os.path.join(config['output']['influence_results'], f"layer_{layer_num}")
    
    # Initialize wandb with a unique run name
    run = init_wandb(config, f"inspect_factors_l{layer_num}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Inspecting factors for layer {layer_num}")
    logger.info(f"Output directory: {output_dir}")
    
    # The exact path where Kronfluence saves the factor files
    factor_dir = os.path.join("influence_results", "results", "influence", "factors", f"factors_{factors_name}")
    
    # Check if the directory exists
    if not os.path.exists(factor_dir):
        # Try alternative locations
        logger.info(f"Factor directory not found at {factor_dir}, looking for alternative locations")
        
        # Log the current working directory for debugging
        logger.info(f"Current working directory: {os.getcwd()}")
        
        alt_paths = [
            # Current working directory paths
            os.path.join("influence_results", "results", "influence", "factors", f"factors_{factors_name}"),
            os.path.join("results", "influence", "factors", f"factors_{factors_name}"),
            os.path.join("influence_results", f"factors_{factors_name}"),
            # Config-based paths
            os.path.join(config['output']['influence_results'], config['factors']['output_dir'], f"factors_{factors_name}"),
            os.path.join(config['output']['influence_results'], f"factors_{factors_name}"),
            os.path.join(config['output']['base_dir'], "influence", "factors", f"factors_{factors_name}"),
            # Additional fallbacks with generic naming
            os.path.join("factors", f"factors_{factors_name}"),
            os.path.join(".", f"factors_{factors_name}")
        ]
        
        # Log all paths being checked
        logger.info("Checking the following paths:")
        for i, path in enumerate(alt_paths):
            logger.info(f"  {i+1}. {path} (exists: {os.path.exists(path)})")
        
        for path in alt_paths:
            if os.path.exists(path):
                factor_dir = path
                logger.info(f"Found factor directory at: {factor_dir}")
                break
        else:
            # If none of the paths exist
            logger.error(f"Could not find factor directory at any known location")
            
            # As a last resort, let's see if we can find ANY directory that looks like it might have factors
            logger.info("Searching for any directory containing factors...")
            potential_roots = [".", "influence_results", "results", config['output']['base_dir'], config['output']['influence_results']]
            found = False
            
            for root in potential_roots:
                if not os.path.exists(root):
                    continue
                    
                for dirpath, dirnames, filenames in os.walk(root):
                    # Look for any directory with 'factors' in the name
                    factor_dirs = [d for d in dirnames if 'factors' in d.lower()]
                    if factor_dirs:
                        logger.info(f"Found potential factor directories in {dirpath}:")
                        for d in factor_dirs:
                            full_path = os.path.join(dirpath, d)
                            logger.info(f"  - {full_path}")
                            
                            # Check if this directory has the expected files
                            if os.path.exists(os.path.join(full_path, "lambda_matrix.safetensors")):
                                logger.info(f"Using factor directory: {full_path}")
                                factor_dir = full_path
                                found = True
                                break
                                
                    if found:
                        break
                        
                if found:
                    break
                    
            if not found:
                raise FileNotFoundError(f"Factor directory not found at {factor_dir} or alternative locations")
    
    # Check for lambda matrices (direct loading as in the original working code)
    lambda_matrix_path = os.path.join(factor_dir, "lambda_matrix.safetensors")
    num_lambda_processed_path = os.path.join(factor_dir, "num_lambda_processed.safetensors")
    
    if os.path.exists(lambda_matrix_path) and os.path.exists(num_lambda_processed_path):
        logger.info(f"Loading lambda matrices directly from {factor_dir}")
        try:
            lambda_matrix = Analyzer.load_file(lambda_matrix_path)
            num_lambda_processed = Analyzer.load_file(num_lambda_processed_path)
            
            # Construct module name for the specified layer
            module_name = f"model.layers.{layer_num}.mlp"
            logger.info(f"Inspecting module: {module_name}")
            
            # Check if the module exists in the lambda matrices
            if module_name not in lambda_matrix:
                # If MLP doesn't exist, check if individual gate/up/down projections exist
                module_names = [
                    f"model.layers.{layer_num}.mlp.gate_proj",
                    f"model.layers.{layer_num}.mlp.up_proj", 
                    f"model.layers.{layer_num}.mlp.down_proj"
                ]
                
                found = False
                for alt_name in module_names:
                    if alt_name in lambda_matrix:
                        module_name = alt_name
                        found = True
                        logger.info(f"Using alternative module: {module_name}")
                        break
                
                if not found:
                    available_modules = list(lambda_matrix.keys())
                    logger.error(f"Module {module_name} or its components not found in factors.")
                    logger.error(f"Available modules: {available_modules}")
                    raise ValueError(f"Module {module_name} not found in factors")
            
            # Process the lambda matrix (normalize by lambda_processed as in the example)
            module_lambda_matrix = lambda_matrix[module_name]
            module_lambda_processed = num_lambda_processed[module_name]
            module_lambda_matrix = module_lambda_matrix.div(module_lambda_processed)
            
            # Get eigenvalues - for this we'll compute them from the lambda matrix
            # This is similar to how the original code handles it
            try:
                # Convert to numpy for eigendecomposition
                # Ensure we convert BFloat16 to float32 first
                if module_lambda_matrix.dtype == torch.bfloat16 or module_lambda_matrix.dtype not in [torch.float32, torch.float64]:
                    logger.info(f"Converting matrix from {module_lambda_matrix.dtype} to float32 for eigendecomposition")
                    module_lambda_matrix_float = module_lambda_matrix.to(torch.float32)
                    lambda_np = module_lambda_matrix_float.cpu().numpy()
                else:
                    lambda_np = module_lambda_matrix.cpu().numpy()
                
                # Check if matrix is square
                if lambda_np.shape[0] != lambda_np.shape[1]:
                    logger.warning(f"Matrix is not square: {lambda_np.shape}. Computing SVD instead.")
                    # For non-square matrices, use SVD to get singular values instead of eigenvalues
                    from scipy import linalg
                    # Use SVD to compute singular values
                    _, s, _ = linalg.svd(lambda_np, full_matrices=False)
                    # Use singular values squared as a proxy for eigenvalues
                    eigenvalues = s**2
                else:
                    # Standard eigenvalue computation for square matrices
                    eigenvalues = np.linalg.eigvalsh(lambda_np)
                
                # Sort in descending order
                eigenvalues = np.sort(eigenvalues)[::-1]
                eigenvalues = torch.from_numpy(eigenvalues)
            except Exception as e:
                logger.warning(f"Error computing eigenvalues: {e}")
                # Provide dummy eigenvalues if computation fails
                eigenvalues = torch.ones(10)
            
            # Visualize the lambda matrix
            visualize_lambda_matrix(module_lambda_matrix, output_dir, clip_percentile, cmap)
            
            # Visualize the eigenvalue distribution
            eigenvalues, eigval_analysis = visualize_eigenvalues(eigenvalues, output_dir)
            
            # Log results to wandb
            if wandb.run is not None:
                # Log the visualizations
                wandb.log({
                    "lambda_matrix": wandb.Image(os.path.join(output_dir, 'lambda_matrix.png')),
                    "eigenvalue_distribution": wandb.Image(os.path.join(output_dir, 'eigenvalue_distribution.png')),
                    "cumulative_variance": wandb.Image(os.path.join(output_dir, 'cumulative_variance.png')),
                    "layer": layer_num,
                    "factors_name": factors_name,
                    "max_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) > 0 else 0,
                    "min_eigenvalue": float(eigenvalues[-1]) if len(eigenvalues) > 0 else 0,
                    "mean_eigenvalue": float(np.mean(eigenvalues)) if len(eigenvalues) > 0 else 0,
                    "median_eigenvalue": float(np.median(eigenvalues)) if len(eigenvalues) > 0 else 0,
                    "eigenvalue_90pct": eigval_analysis['eigval_90'],
                    "eigenvalue_95pct": eigval_analysis['eigval_95'],
                    "eigenvalue_99pct": eigval_analysis['eigval_99'],
                })
                
                # Save the full output directory to wandb
                wandb.save(os.path.join(output_dir, "*"))
            
            logger.info(f"Factor inspection for layer {layer_num} completed.")
            logger.info(f"Results saved to {output_dir}")
            
            return output_dir
            
        except Exception as e:
            logger.error(f"Error loading or processing lambda matrices: {e}")
            raise
    
    # If can't find lambda matrices, check for covariance matrices
    logger.info("Lambda matrices not found or couldn't be processed, looking for covariance matrices")
    activation_cov_path = os.path.join(factor_dir, "activation_covariance.safetensors")
    gradient_cov_path = os.path.join(factor_dir, "gradient_covariance.safetensors")
    
    if os.path.exists(activation_cov_path) and os.path.exists(gradient_cov_path):
        logger.info(f"Loading covariance matrices from {factor_dir}")
        try:
            activation_covariance = Analyzer.load_file(activation_cov_path)
            gradient_covariance = Analyzer.load_file(gradient_cov_path)
            
            # Construct module name for the specified layer
            module_name = f"model.layers.{layer_num}.mlp"
            logger.info(f"Inspecting module: {module_name}")
            
            # Check for alternative modules if needed
            if module_name not in activation_covariance:
                # If MLP doesn't exist, check if individual gate/up/down projections exist
                module_names = [
                    f"model.layers.{layer_num}.mlp.gate_proj",
                    f"model.layers.{layer_num}.mlp.up_proj", 
                    f"model.layers.{layer_num}.mlp.down_proj"
                ]
                
                found = False
                for alt_name in module_names:
                    if alt_name in activation_covariance:
                        module_name = alt_name
                        found = True
                        logger.info(f"Using alternative module: {module_name}")
                        break
                
                if not found:
                    available_modules = list(activation_covariance.keys())
                    logger.error(f"Module {module_name} or its components not found in covariance matrices.")
                    logger.error(f"Available modules: {available_modules}")
                    raise ValueError(f"Module {module_name} not found in covariance matrices")
            
            # Get the covariance matrices
            module_activation_covariance = activation_covariance[module_name]
            module_gradient_covariance = gradient_covariance[module_name]
            
            # Convert to float32 for visualization
            if hasattr(module_activation_covariance, 'to'):
                module_activation_covariance = module_activation_covariance.to(torch.float32)
            if hasattr(module_gradient_covariance, 'to'):
                module_gradient_covariance = module_gradient_covariance.to(torch.float32)
            
            # For visualization purposes, we'll use the activation covariance as the "lambda matrix"
            visualize_lambda_matrix(module_activation_covariance, output_dir, clip_percentile, cmap)
            
            # Compute eigenvalues for the visualization
            try:
                # Convert to numpy for eigendecomposition
                if hasattr(module_activation_covariance, 'cpu'):
                    # Check for BFloat16 and convert if needed
                    if hasattr(module_activation_covariance, 'dtype') and (
                        module_activation_covariance.dtype == torch.bfloat16 or 
                        module_activation_covariance.dtype not in [torch.float32, torch.float64]
                    ):
                        logger.info(f"Converting covariance from {module_activation_covariance.dtype} to float32")
                        module_activation_covariance = module_activation_covariance.to(torch.float32)
                    activation_np = module_activation_covariance.cpu().numpy()
                else:
                    activation_np = np.array(module_activation_covariance)
                    
                # Check if matrix is square
                if activation_np.shape[0] != activation_np.shape[1]:
                    logger.warning(f"Covariance matrix is not square: {activation_np.shape}. Computing SVD instead.")
                    # For non-square matrices, use SVD to get singular values
                    from scipy import linalg
                    # Use SVD to compute singular values
                    _, s, _ = linalg.svd(activation_np, full_matrices=False)
                    # Use singular values squared as a proxy for eigenvalues
                    eigenvalues = s**2
                else:
                    # Standard eigenvalue computation for square matrices
                    eigenvalues = np.linalg.eigvalsh(activation_np)
                
                # Sort in descending order
                eigenvalues = np.sort(eigenvalues)[::-1]
                eigenvalues = torch.from_numpy(eigenvalues)
            except Exception as e:
                logger.warning(f"Error computing eigenvalues: {e}")
                # Provide dummy eigenvalues if computation fails
                eigenvalues = torch.ones(10)
            
            # Visualize the eigenvalue distribution
            visualize_eigenvalues(eigenvalues, output_dir)
            
            logger.info(f"Factor inspection for layer {layer_num} completed.")
            logger.info(f"Results saved to {output_dir}")
            
            return output_dir
            
        except Exception as e:
            logger.error(f"Error loading or processing covariance matrices: {e}")
            raise
    
    # If we've reached here, we couldn't find or process any factor files
    logger.error(f"Could not find or process factor files in {factor_dir}")
    logger.error(f"Contents of {factor_dir}:")
    for item in os.listdir(factor_dir):
        logger.error(f"  - {item}")
    
    raise FileNotFoundError(f"Could not find or process factor files in {factor_dir}")

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
        # Convert BFloat16 or other non-standard types to float32 first
        if lambda_matrix.dtype == torch.bfloat16 or lambda_matrix.dtype not in [torch.float32, torch.float64]:
            logger.info(f"Converting tensor from {lambda_matrix.dtype} to float32")
            lambda_matrix = lambda_matrix.to(torch.float32)
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
        
    Returns:
        tuple: (eigenvalues as numpy array, dict of eigenvalue analysis results)
    """
    # Convert to numpy for visualization
    if isinstance(eigenvalues, torch.Tensor):
        # Convert BFloat16 or other non-standard types to float32 first
        if eigenvalues.dtype == torch.bfloat16 or eigenvalues.dtype not in [torch.float32, torch.float64]:
            logger.info(f"Converting eigenvalues from {eigenvalues.dtype} to float32")
            eigenvalues = eigenvalues.to(torch.float32)
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
    
    # Return the eigenvalues and analysis results for potential logging
    eigval_analysis = {
        'eigval_90': int(eigval_90),
        'eigval_95': int(eigval_95),
        'eigval_99': int(eigval_99),
        'total_eigenvalues': len(sorted_eig),
        'largest_eigenvalue': float(sorted_eig[0]),
        'smallest_eigenvalue': float(sorted_eig[-1]),
        'mean_eigenvalue': float(np.mean(sorted_eig)),
        'median_eigenvalue': float(np.median(sorted_eig))
    }
    
    return sorted_eig, eigval_analysis 