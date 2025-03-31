import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import torch
try:
    from tueplots import markers  # Try to import tueplots if available
except ImportError:
    print("tueplots not found, continuing without it.")
    markers = None

from kronfluence.analyzer import Analyzer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors_name", type=str, default="olmoe_1b_factors")
    parser.add_argument("--layer_num", type=int, default=15)  # Updated to 15 for OLMoE-1B
    parser.add_argument("--output_dir", type=str, default="influence_results/factor_analysis")
    parser.add_argument("--clip_percentile", type=float, default=99.5, 
                       help="Percentile for clipping extreme values in visualization")
    parser.add_argument("--cmap", type=str, default="coolwarm", 
                       help="Colormap to use (try: coolwarm, RdBu_r, viridis, plasma)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up matplotlib parameters
    plt.rcParams.update({"figure.dpi": 150})
    # Use tueplots markers if available
    if markers:
        plt.rcParams.update(markers.with_edge())
    plt.rcParams["axes.axisbelow"] = True

    layer_num = args.layer_num
    # For OLMoE, try both the feed-forward modules
    # The exact module names may vary, will adapt based on what we find
    module_names = [
        f"model.layers.{layer_num}.feed_forward.w1",
        f"model.layers.{layer_num}.feed_forward.w2",
        f"model.layers.{layer_num}.feed_forward.w3"
    ]

    # Fix path to handle the nested directory structure from fit_factors.py
    factor_path = f"influence_results/influence_results/factors_{args.factors_name}"
    
    print(f"Looking for factors in: {factor_path}")
    
    # Verify that the directory exists
    if not os.path.exists(factor_path):
        print(f"ERROR: Factor directory not found at {factor_path}")
        # Check both possible directory structures
        if os.path.exists(f"influence_results/factors_{args.factors_name}"):
            factor_path = f"influence_results/factors_{args.factors_name}"
            print(f"Found alternative location at: {factor_path}")
        else:
            print("Available directories in influence_results:")
            for item in os.listdir("influence_results"):
                print(f"  - {item}")
                if os.path.isdir(os.path.join("influence_results", item)):
                    try:
                        subitems = os.listdir(os.path.join("influence_results", item))
                        for subitem in subitems:
                            print(f"    - {subitem}")
                    except:
                        pass
            return

    # First check if lambda matrices exist (similar to the example script)
    try:
        # Try to load lambda matrices first (like in kronfluence/examples/openwebtext/inspect_factors.py)
        print("Checking for lambda matrices (num_lambda_processed.safetensors and lambda_matrix.safetensors)")
        
        lambda_processed_path = f"{factor_path}/num_lambda_processed.safetensors"
        lambda_matrix_path = f"{factor_path}/lambda_matrix.safetensors"
        
        if os.path.exists(lambda_processed_path) and os.path.exists(lambda_matrix_path):
            print("Found lambda matrices, analyzing them first")
            lambda_processed = Analyzer.load_file(lambda_processed_path)
            lambda_matrix = Analyzer.load_file(lambda_matrix_path)
            
            # Get available modules in lambda matrices
            available_modules = list(lambda_matrix.keys())
            print(f"Available modules in lambda matrix: {available_modules}")
            
            # Check for available modules
            if not any(module in available_modules for module in module_names):
                print("None of the specified modules found in lambda matrices. Trying alternative modules.")
                module_names = available_modules[:3] if len(available_modules) >= 3 else available_modules
            
            # Analyze each module
            for module_name in module_names:
                if module_name not in lambda_matrix:
                    print(f"Module {module_name} not found in lambda matrices. Skipping.")
                    continue
                    
                print(f"Analyzing lambda matrices for module: {module_name}")
                
                # Process lambda matrices like in the example script
                module_lambda_processed = lambda_processed[module_name]
                module_lambda_matrix = lambda_matrix[module_name]
                
                # Normalize by dividing by lambda_processed (as in the example script)
                module_lambda_matrix = module_lambda_matrix.div(module_lambda_processed)
                
                # Convert to float for visualization
                module_lambda_matrix = module_lambda_matrix.float()
                
                # 1. Visualize full lambda matrix
                plt.figure(figsize=(12, 10))
                plt.matshow(module_lambda_matrix, cmap="PuBu", norm=LogNorm())
                plt.title(f"Lambda Matrix: {module_name}")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"{module_name.replace('.', '_')}_lambda_matrix.png"))
                plt.close()
                
                # 2. Plot sorted eigenvalues
                plt.figure(figsize=(12, 6))
                flattened_lambda = module_lambda_matrix.view(-1).cpu().numpy()
                sorted_lambda = np.sort(flattened_lambda)
                plt.plot(sorted_lambda)
                plt.title(f"Sorted Lambda Values: {module_name}")
                plt.grid(True)
                plt.yscale("log")
                plt.ylabel("Lambda Values (log scale)")
                plt.xlabel("Index")
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"{module_name.replace('.', '_')}_sorted_lambda.png"))
                plt.close()
            
            print("Lambda matrix analysis complete.")
            
    except FileNotFoundError as e:
        print(f"Lambda matrices not found: {e}")
        print("Continuing with covariance matrices instead...")
    
    # Look for raw covariance files instead of the combined lambda files
    try:
        # Try to load covariance files directly - the example from kronfluence/examples/cifar/inspect_factors.py
        print("Attempting to load activation_covariance.safetensors and gradient_covariance.safetensors")
        
        # Check if consolidated files exist
        if os.path.exists(f"{factor_path}/activation_covariance.safetensors"):
            activation_covariance = Analyzer.load_file(f"{factor_path}/activation_covariance.safetensors")
            gradient_covariance = Analyzer.load_file(f"{factor_path}/gradient_covariance.safetensors")
            print("Successfully loaded consolidated covariance files")
        else:
            # Look for partition files in the directory
            print("Consolidated files not found, looking for partition files")
            partitioned_files = [f for f in os.listdir(factor_path) if f.startswith("activation_covariance_data_partition")]
            if partitioned_files:
                print(f"Found {len(partitioned_files)} partitioned files. Analysis will use the first partition.")
                # Use the first partition for visualization
                partition_number = 0
                module_partition = 0
                activation_path = f"{factor_path}/activation_covariance_data_partition{partition_number}_module_partition{module_partition}.safetensors"
                gradient_path = f"{factor_path}/gradient_covariance_data_partition{partition_number}_module_partition{module_partition}.safetensors"
                
                activation_covariance = Analyzer.load_file(activation_path)
                gradient_covariance = Analyzer.load_file(gradient_path)
                print(f"Successfully loaded partition files from partition {partition_number}, module partition {module_partition}")
            else:
                print("No covariance files found. Please check that factors were computed correctly.")
                return
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        try:
            print(f"Files in {factor_path}:")
            for item in os.listdir(factor_path):
                print(f"  - {item}")
        except:
            print(f"Could not list files in {factor_path}")
        return
    
    # Print available modules
    print(f"Available modules in activation covariance: {list(activation_covariance.keys())}")
    
    # Check for available modules if our guesses don't match
    available_modules = list(activation_covariance.keys())
    if not any(module in available_modules for module in module_names):
        print("None of the specified modules found. Using first few available modules instead.")
        module_names = available_modules[:3] if len(available_modules) >= 3 else available_modules
    
    for module_name in module_names:
        if module_name not in activation_covariance:
            print(f"Module {module_name} not found in factors. Skipping.")
            continue
            
        print(f"Analyzing module: {module_name}")
        
        # Get the covariance matrices for this module
        module_activation_covariance = activation_covariance[module_name]
        module_gradient_covariance = gradient_covariance[module_name]
        
        # Convert tensors to float32 to avoid BFloat16 errors with matplotlib
        if hasattr(module_activation_covariance, 'to'):
            module_activation_covariance = module_activation_covariance.to(torch.float32)
        if hasattr(module_gradient_covariance, 'to'):
            module_gradient_covariance = module_gradient_covariance.to(torch.float32)
        
        # Convert to numpy if needed
        if hasattr(module_activation_covariance, 'cpu'):
            module_activation_covariance = module_activation_covariance.cpu().numpy()
        if hasattr(module_gradient_covariance, 'cpu'):
            module_gradient_covariance = module_gradient_covariance.cpu().numpy()
            
        # Print statistics about the matrices
        print(f"\nActivation covariance matrix statistics for {module_name}:")
        print(f"Shape: {module_activation_covariance.shape}")
        print(f"Min: {np.min(module_activation_covariance)}")
        print(f"Max: {np.max(module_activation_covariance)}")
        print(f"Mean: {np.mean(module_activation_covariance)}")
        print(f"Std: {np.std(module_activation_covariance)}")
        print(f"Percentiles: 1%={np.percentile(module_activation_covariance, 1)}, " +
              f"99%={np.percentile(module_activation_covariance, 99)}, " +
              f"99.9%={np.percentile(module_activation_covariance, 99.9)}")
        
        print(f"\nGradient covariance matrix statistics for {module_name}:")
        print(f"Shape: {module_gradient_covariance.shape}")
        print(f"Min: {np.min(module_gradient_covariance)}")
        print(f"Max: {np.max(module_gradient_covariance)}")
        print(f"Mean: {np.mean(module_gradient_covariance)}")
        print(f"Std: {np.std(module_gradient_covariance)}")
        print(f"Percentiles: 1%={np.percentile(module_gradient_covariance, 1)}, " +
              f"99%={np.percentile(module_gradient_covariance, 99)}, " +
              f"99.9%={np.percentile(module_gradient_covariance, 99.9)}")
        
        # --------- IMPROVED VISUALIZATION APPROACH ---------
        
        # 1. Visualize activation covariance
        plt.figure(figsize=(12, 10))
        
        # Find outliers and clip them for better visualization
        vmin_act = np.percentile(module_activation_covariance, 100 - args.clip_percentile)
        vmax_act = np.percentile(module_activation_covariance, args.clip_percentile)
        
        # For covariance matrices, use symmetric color scaling around zero if the data spans negative values
        has_negative = np.min(module_activation_covariance) < 0
        has_positive = np.max(module_activation_covariance) > 0
        
        if has_negative and has_positive:
            # Symmetric data - use a diverging colormap centered at zero
            abs_max = max(abs(vmin_act), abs(vmax_act))
            vmin_act, vmax_act = -abs_max, abs_max
            norm = Normalize(vmin=vmin_act, vmax=vmax_act)
            
            # Use a diagonal mask to highlight the diagonal elements
            diagonal = np.diag(np.ones(module_activation_covariance.shape[0])).astype(bool)
            
            # Plot diagonal elements separately with higher visibility
            diag_values = np.copy(module_activation_covariance)
            diag_values[~diagonal] = 0  # Zero out non-diagonal
            plt.matshow(diag_values, cmap="hot", norm=norm, alpha=0.5)
            
            # Plot full matrix
            plt.matshow(module_activation_covariance, cmap=args.cmap, norm=norm, alpha=0.8)
            plt.colorbar(label="Covariance Value")
        elif has_negative or has_positive:
            # If data is mostly positive or mostly negative, use SymLogNorm for better contrast
            linthresh = max(1e-8, min(abs(vmin_act), abs(vmax_act)) / 10)
            norm = SymLogNorm(linthresh=linthresh, vmin=vmin_act, vmax=vmax_act)
            plt.matshow(module_activation_covariance, cmap=args.cmap, norm=norm)
            plt.colorbar(label="Covariance Value (symlog scale)")
        
        plt.title(f"Activation Covariance Matrix: {module_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{module_name.replace('.', '_')}_activation_heatmap.png"))
        plt.close()
        
        # Also save a version focusing on the diagonal to see variance structure
        plt.figure(figsize=(12, 10))
        diagonal_values = np.diag(module_activation_covariance)
        plt.plot(diagonal_values)
        plt.title(f"Diagonal Values (Variance) for {module_name}")
        plt.ylabel("Variance")
        plt.xlabel("Feature Index")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{module_name.replace('.', '_')}_diagonal_variance.png"))
        plt.close()

        # 2. Visualize gradient covariance
        plt.figure(figsize=(12, 10))
        
        # Find outliers and clip them for better visualization
        vmin_grad = np.percentile(module_gradient_covariance, 100 - args.clip_percentile)
        vmax_grad = np.percentile(module_gradient_covariance, args.clip_percentile)
        
        # For covariance matrices, use symmetric color scaling around zero if the data spans negative values
        has_negative = np.min(module_gradient_covariance) < 0
        has_positive = np.max(module_gradient_covariance) > 0
        
        if has_negative and has_positive:
            # Symmetric data - use a diverging colormap centered at zero
            abs_max = max(abs(vmin_grad), abs(vmax_grad))
            vmin_grad, vmax_grad = -abs_max, abs_max
            norm = Normalize(vmin=vmin_grad, vmax=vmax_grad)
            
            # Use a diagonal mask to highlight the diagonal elements
            diagonal = np.diag(np.ones(module_gradient_covariance.shape[0])).astype(bool)
            
            # Plot diagonal elements separately with higher visibility
            diag_values = np.copy(module_gradient_covariance)
            diag_values[~diagonal] = 0  # Zero out non-diagonal
            plt.matshow(diag_values, cmap="hot", norm=norm, alpha=0.5)
            
            # Plot full matrix
            plt.matshow(module_gradient_covariance, cmap=args.cmap, norm=norm, alpha=0.8)
            plt.colorbar(label="Covariance Value")
        elif has_negative or has_positive:
            # If data is mostly positive or mostly negative, use SymLogNorm for better contrast
            linthresh = max(1e-8, min(abs(vmin_grad), abs(vmax_grad)) / 10)
            norm = SymLogNorm(linthresh=linthresh, vmin=vmin_grad, vmax=vmax_grad)
            plt.matshow(module_gradient_covariance, cmap=args.cmap, norm=norm)
            plt.colorbar(label="Covariance Value (symlog scale)")
        
        plt.title(f"Gradient Covariance Matrix: {module_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{module_name.replace('.', '_')}_gradient_heatmap.png"))
        plt.close()
        
        # 3. Generate eigenvalue plots to visualize the spectrum
        # This can reveal rank and important directions in the data
        plt.figure(figsize=(12, 6))
        
        # Compute eigenvalues for both matrices
        try:
            # Use np.linalg.eigh for symmetric matrices (more stable)
            act_eigenvalues = np.linalg.eigvalsh(module_activation_covariance)
            grad_eigenvalues = np.linalg.eigvalsh(module_gradient_covariance)
            
            # Sort in descending order
            act_eigenvalues = np.sort(act_eigenvalues)[::-1]
            grad_eigenvalues = np.sort(grad_eigenvalues)[::-1]
            
            # Plot on same figure with log scale
            plt.subplot(1, 2, 1)
            plt.plot(act_eigenvalues, label="Activation")
            plt.xlabel("Eigenvalue Index")
            plt.ylabel("Eigenvalue Magnitude")
            plt.title("Linear Scale")
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.semilogy(act_eigenvalues, label="Activation")
            plt.xlabel("Eigenvalue Index")
            plt.ylabel("Eigenvalue Magnitude (log)")
            plt.title("Log Scale")
            plt.grid(True)
            
            plt.suptitle(f"Eigenvalue Spectrum for {module_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"{module_name.replace('.', '_')}_eigenvalue_spectrum.png"))
            plt.close()
            
            # Plot the top eigenvalues and their percentage of total variance
            top_k = min(20, len(act_eigenvalues))
            plt.figure(figsize=(12, 6))
            
            # Calculate cumulative explained variance
            total_var = np.sum(act_eigenvalues)
            cum_var_ratio = np.cumsum(act_eigenvalues) / total_var
            
            # Plot top eigenvalues
            plt.subplot(1, 2, 1)
            plt.bar(range(top_k), act_eigenvalues[:top_k])
            plt.xlabel("Eigenvalue Index")
            plt.ylabel("Eigenvalue Magnitude")
            plt.title(f"Top {top_k} Eigenvalues")
            plt.grid(True)
            
            # Plot cumulative explained variance
            plt.subplot(1, 2, 2)
            plt.plot(cum_var_ratio, 'r-')
            plt.xlabel("Number of Components")
            plt.ylabel("Cumulative Explained Variance")
            plt.title("Explained Variance")
            plt.grid(True)
            
            plt.suptitle(f"Principal Components Analysis for {module_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"{module_name.replace('.', '_')}_pca_analysis.png"))
            plt.close()
            
        except np.linalg.LinAlgError:
            print("Warning: Unable to compute eigenvalues - matrix may not be positive definite")
            
        print(f"Analysis complete for {module_name}. Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main() 