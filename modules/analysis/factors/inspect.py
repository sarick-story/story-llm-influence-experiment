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
    
    # Try to determine the correct factor directory
    # Base path construction from config
    base_factor_dir = os.path.join(config['output']['influence_results'], config['factors']['output_dir'])
    factor_dir_name = f"factors_{factors_name}"
    factor_dir = os.path.join(base_factor_dir, factor_dir_name)
    
    logger.info(f"Attempting to find factor directory, starting with: {factor_dir}")

    # Check if the directory exists, use alternative paths if needed
    if not os.path.exists(factor_dir):
        logger.warning(f"Factor directory not found at configured path: {factor_dir}. Trying alternatives...")
        alt_paths = [
            os.path.join("results", "influence", "factors", factor_dir_name), # Common alternative
            os.path.join(config['output']['influence_results'], factor_dir_name), # Alt config structure
            os.path.join(config['output']['base_dir'], "influence", "factors", factor_dir_name), # Another alt config
            os.path.join("influence_results", "results", "influence", "factors", factor_dir_name), # Old potential path
            factor_dir_name # Relative path from pwd
        ]
        
        found = False
        for path in alt_paths:
            logger.info(f"Checking alternative path: {path}")
            if os.path.exists(path):
                factor_dir = path
                logger.info(f"Found factor directory at: {factor_dir}")
                found = True
                break
        
        if not found:
             logger.error(f"Could not find factor directory at any known location.")
             # Attempt a broader search as a last resort
             potential_roots = [".", "influence_results", "results", config['output']['base_dir'], config['output']['influence_results']]
             search_found = False
             for root in potential_roots:
                 if not os.path.exists(root): continue
                 for dirpath, dirnames, _ in os.walk(root):
                     if factor_dir_name in dirnames:
                         potential_path = os.path.join(dirpath, factor_dir_name)
                         if os.path.exists(os.path.join(potential_path, "lambda_matrix.safetensors")):
                             factor_dir = potential_path
                             logger.info(f"Found factor directory via search: {factor_dir}")
                             search_found = True
                             break
                 if search_found: break
             if not search_found:
                raise FileNotFoundError(f"Factor directory '{factor_dir_name}' not found.")

    # Check for lambda matrices
    lambda_matrix_path = os.path.join(factor_dir, "lambda_matrix.safetensors")
    num_lambda_processed_path = os.path.join(factor_dir, "num_lambda_processed.safetensors")
    
    processed_lambda_matrix = None
    actual_module_name = None # Keep track of the module name actually used

    if os.path.exists(lambda_matrix_path) and os.path.exists(num_lambda_processed_path):
        logger.info(f"Loading lambda matrices directly from {factor_dir}")
        try:
            lambda_matrix = Analyzer.load_file(lambda_matrix_path)
            num_lambda_processed = Analyzer.load_file(num_lambda_processed_path)
            
            # Construct potential module names based on layer_num
            # Prioritize 'down_proj' as in the example, then 'up_proj', 'gate_proj', then the base 'mlp'
            potential_module_names = [
                f"model.layers.{layer_num}.mlp.down_proj",
                f"model.layers.{layer_num}.mlp.up_proj",
                f"model.layers.{layer_num}.mlp.gate_proj",
                f"model.layers.{layer_num}.mlp"
            ]
            
            found_module = False
            for name in potential_module_names:
                if name in lambda_matrix:
                    actual_module_name = name
                    logger.info(f"Found and using module: {actual_module_name}")
                    
                    # Process the lambda matrix (normalize)
                    module_lambda_matrix = lambda_matrix[actual_module_name]
                    module_lambda_processed = num_lambda_processed[actual_module_name]
                    # Ensure division happens correctly, handling potential division by zero
                    processed_lambda_matrix = module_lambda_matrix.float().div(module_lambda_processed.float().clamp(min=1e-6))
                    found_module = True
                    break
            
            if not found_module:
                available_modules = list(lambda_matrix.keys())
                logger.error(f"None of the expected modules for layer {layer_num} found in factors.")
                logger.error(f"Available modules: {available_modules}")
                raise ValueError(f"No suitable module found for layer {layer_num} in factors")

        except Exception as e:
            logger.error(f"Error loading or processing lambda matrices: {e}")
            raise # Reraise after logging
            
    # Fallback to covariance matrices if lambda matrix processing failed or files don't exist
    if processed_lambda_matrix is None:
        logger.info("Lambda matrices not found or couldn't be processed. Looking for covariance matrices.")
        activation_cov_path = os.path.join(factor_dir, "activation_covariance.safetensors")
        
        if os.path.exists(activation_cov_path):
            logger.info(f"Loading activation covariance matrix from {factor_dir}")
            try:
                activation_covariance = Analyzer.load_file(activation_cov_path)
                
                # Construct potential module names
                potential_module_names = [
                    f"model.layers.{layer_num}.mlp.down_proj",
                    f"model.layers.{layer_num}.mlp.up_proj",
                    f"model.layers.{layer_num}.mlp.gate_proj",
                    f"model.layers.{layer_num}.mlp"
                ]

                found_module = False
                for name in potential_module_names:
                     if name in activation_covariance:
                        actual_module_name = name
                        logger.info(f"Found and using module (covariance): {actual_module_name}")
                        # Use covariance matrix as a proxy for lambda matrix visualization
                        processed_lambda_matrix = activation_covariance[actual_module_name].float()
                        found_module = True
                        break
                
                if not found_module:
                    available_modules = list(activation_covariance.keys())
                    logger.error(f"None of the expected modules for layer {layer_num} found in covariance matrices.")
                    logger.error(f"Available modules: {available_modules}")
                    raise ValueError(f"No suitable module found for layer {layer_num} in covariance matrices")

            except Exception as e:
                logger.error(f"Error loading or processing covariance matrices: {e}")
                raise
        else:
            # If neither lambda nor covariance found
            logger.error(f"Could not find lambda matrices or covariance matrices in {factor_dir}")
            logger.error(f"Contents of {factor_dir}: {os.listdir(factor_dir)}")
            raise FileNotFoundError(f"Necessary factor files (lambda or covariance) not found in {factor_dir}")

    # Ensure we have a matrix to visualize
    if processed_lambda_matrix is None or actual_module_name is None:
         raise RuntimeError("Failed to load or process any factor matrix for visualization.")

    # -- Visualizations --
    
    # 1. Visualize the lambda matrix heatmap
    visualize_lambda_matrix(processed_lambda_matrix, output_dir, actual_module_name, clip_percentile, cmap)
    
    # 2. Visualize the sorted lambda matrix values
    visualize_sorted_lambda_values(processed_lambda_matrix, output_dir, actual_module_name)
    
    # -- WandB Logging --
    if wandb.run is not None:
        try:
             # Prepare log data
            log_data = {
                "lambda_matrix": wandb.Image(os.path.join(output_dir, f'lambda_matrix_{actual_module_name}.png')),
                "sorted_lambda_values": wandb.Image(os.path.join(output_dir, f'sorted_lambda_values_{actual_module_name}.png')),
                "layer": layer_num,
                "module_name": actual_module_name,
                "factors_name": factors_name,
            }
             
            # Log metrics about the matrix values
            flat_values = processed_lambda_matrix.view(-1).cpu().numpy()
            log_data.update({
                "lambda_max_value": float(np.max(flat_values)),
                "lambda_min_value": float(np.min(flat_values)),
                "lambda_mean_value": float(np.mean(flat_values)),
                "lambda_median_value": float(np.median(flat_values)),
                "lambda_std_dev": float(np.std(flat_values)),
            })
            
            wandb.log(log_data)
            
            # Save the full output directory to wandb artifacts
            logger.info(f"Saving output directory '{output_dir}' to WandB artifacts.")
            artifact = wandb.Artifact(f'factor_inspection_layer_{layer_num}', type='factor_analysis')
            artifact.add_dir(output_dir)
            wandb.log_artifact(artifact)
            
        except Exception as e:
            logger.error(f"Failed to log results to WandB: {e}")

    logger.info(f"Factor inspection for layer {layer_num} (module: {actual_module_name}) completed.")
    logger.info(f"Results saved to {output_dir}")
    
    return output_dir


def visualize_lambda_matrix(lambda_matrix, output_dir, module_name, clip_percentile=99.5, cmap='coolwarm'):
    """
    Visualize the lambda matrix (covariance or precision matrix).
    """
    # Convert to numpy for visualization
    if isinstance(lambda_matrix, torch.Tensor):
        # Convert BFloat16 or other non-standard types to float32 first
        if lambda_matrix.dtype == torch.bfloat16 or lambda_matrix.dtype not in [torch.float32, torch.float64]:
            logger.info(f"Converting tensor from {lambda_matrix.dtype} to float32 for heatmap")
            lambda_matrix = lambda_matrix.to(torch.float32)
        lambda_np = lambda_matrix.cpu().numpy()
    else:
        lambda_np = np.array(lambda_matrix)
        
    # Handle potential NaN/Inf values
    lambda_np = np.nan_to_num(lambda_np)
    
    # Clip extreme values for better visualization
    # Use absolute value for percentile calculation
    valid_values = lambda_np[np.isfinite(lambda_np)]
    if valid_values.size > 0:
         vmax = np.percentile(np.abs(valid_values), clip_percentile)
         vmin = -vmax
    else:
         vmin, vmax = -1.0, 1.0 # Default if no valid values
         logger.warning("Lambda matrix contains no finite values for clipping.")

    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Normalize color map
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot heatmap
    im = plt.imshow(lambda_np, cmap=cmap, norm=norm, aspect='auto') # Use aspect='auto' for non-square
    plt.colorbar(im, label='Value')
    plt.title(f'Lambda Matrix: {module_name}')
    plt.xlabel('Output dimension')
    plt.ylabel('Input dimension')
    
    # Save figure
    filename = f'lambda_matrix_{module_name}.png'
    save_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"Lambda matrix visualization saved to {save_path}")


def visualize_sorted_lambda_values(lambda_matrix, output_dir, module_name):
    """
    Visualize the sorted values of the lambda matrix on a log scale.
    """
    # Convert to numpy for visualization
    if isinstance(lambda_matrix, torch.Tensor):
         # Ensure float32 for processing
        if lambda_matrix.dtype != torch.float32:
             lambda_matrix = lambda_matrix.to(torch.float32)
        lambda_np = lambda_matrix.view(-1).cpu().numpy()
    else:
        lambda_np = np.array(lambda_matrix).flatten()

    # Remove NaN/Inf values before sorting
    lambda_np = lambda_np[np.isfinite(lambda_np)]

    if lambda_np.size == 0:
        logger.warning(f"No finite values found in lambda matrix for module {module_name}. Skipping sorted values plot.")
        return

    # Sort the flattened array
    sorted_lambda_values = np.sort(lambda_np)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_lambda_values)
    plt.yscale("symlog", linthresh=1e-5) # Use symlog to handle potential negative values and zero
    plt.title(f'Sorted Lambda Values: {module_name}')
    plt.xlabel('Sorted Index')
    plt.ylabel('Value (symlog scale)')
    plt.grid(True, which="both", ls="--")
    
    # Save figure
    filename = f'sorted_lambda_values_{module_name}.png'
    save_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logger.info(f"Sorted lambda values visualization saved to {save_path}") 