import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from kronfluence.analyzer import Analyzer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors_name", type=str, default="tiny_lm_factors")
    parser.add_argument("--layer_num", type=int, default=11)  # Last layer by default
    parser.add_argument("--output_dir", type=str, default="influence_results/factor_analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up matplotlib parameters
    plt.rcParams.update({"figure.dpi": 150})
    plt.rcParams["axes.axisbelow"] = True

    layer_num = args.layer_num
    # Try both MLP modules
    module_names = [
        f"transformer.h.{layer_num}.mlp.c_fc",
        f"transformer.h.{layer_num}.mlp.c_proj"
    ]

    # Update the file paths to the correct location
    # The files are in influence_results/{factors_name}/factors_{factors_name}/
    factor_path = f"influence_results/{args.factors_name}/factors_{args.factors_name}"
    
    print(f"Looking for factors in: {factor_path}")
    
    # Verify that the directory exists
    if not os.path.exists(factor_path):
        print(f"ERROR: Factor directory not found at {factor_path}")
        print("Available directories in influence_results:")
        for item in os.listdir("influence_results"):
            print(f"  - {item}")
        return

    # Load factor files
    try:
        lambda_processed = Analyzer.load_file(f"{factor_path}/num_lambda_processed.safetensors")
        lambda_matrix = Analyzer.load_file(f"{factor_path}/lambda_matrix.safetensors")
        print("Successfully loaded factor files")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(f"Files in {factor_path}:")
        for item in os.listdir(factor_path):
            print(f"  - {item}")
        return
    
    print(f"Available modules: {list(lambda_processed.keys())}")
    
    for module_name in module_names:
        if module_name not in lambda_processed:
            print(f"Module {module_name} not found in factors. Skipping.")
            continue
            
        print(f"Analyzing module: {module_name}")
        
        # Get the lambda matrix for this module
        module_lambda_processed = lambda_processed[module_name]
        module_lambda_matrix = lambda_matrix[module_name]
        
        # Normalize by number of processed examples
        module_lambda_matrix = module_lambda_matrix.div(module_lambda_processed).float()
        
        # Create heatmap visualization
        plt.figure(figsize=(10, 8))
        plt.matshow(module_lambda_matrix, cmap="PuBu", norm=LogNorm())
        plt.title(f"Lambda Matrix: {module_name}")
        plt.colorbar(label="Eigenvalues")
        plt.savefig(os.path.join(args.output_dir, f"{module_name.replace('.', '_')}_heatmap.png"))
        plt.close()

        # Create sorted eigenvalues plot
        plt.figure(figsize=(10, 6))
        eigenvalues = module_lambda_matrix.view(-1).numpy()
        sorted_eigenvalues = np.sort(eigenvalues)
        plt.plot(sorted_eigenvalues)
        plt.title(f"Sorted Eigenvalues: {module_name}")
        plt.grid()
        plt.yscale("log")
        plt.ylabel("Eigenvalues (log scale)")
        plt.xlabel("Index")
        plt.savefig(os.path.join(args.output_dir, f"{module_name.replace('.', '_')}_eigenvalues.png"))
        plt.close()
        
        print(f"Analysis complete for {module_name}. Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main() 