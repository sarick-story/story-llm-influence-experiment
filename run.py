#!/usr/bin/env python3
"""
Simple wrapper script for the LLM Influence Analysis Framework.

This script is a convenience wrapper around main.py that provides a simplified
command-line interface for common operations.
"""

import os
import sys
import subprocess
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM Influence Analysis Framework - Simple Wrapper"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "command",
        choices=[
            "train",            # Train the model
            "analysis",         # Run the full analysis pipeline
            "factors",          # Compute influence factors
            "scores",           # Compute influence scores
            "evaluate",         # Run the evaluation
            "custom-eval",      # Run just the custom evaluation
            "deepeval-eval",    # Run just the DeepEval evaluation
            "help"              # Show help
        ],
        help="Command to execute"
    )
    
    return parser.parse_args()

def run_main_command(config_path, command, *args):
    """Run a command using main.py."""
    main_args = ["python", "main.py", "--config", config_path]
    
    if command == "train":
        main_args.append("train")
    elif command == "analysis":
        main_args.append("run_full_analysis")
    elif command == "factors":
        main_args.append("compute_factors")
    elif command == "scores":
        main_args.append("compute_scores")
    elif command == "evaluate":
        main_args.append("evaluate")
        main_args.append("--type")
        main_args.append("all")
    elif command == "custom-eval":
        main_args.append("evaluate")
        main_args.append("--type")
        main_args.append("custom")
    elif command == "deepeval-eval":
        main_args.append("evaluate")
        main_args.append("--type")
        main_args.append("deepeval")
    
    # Add any additional arguments
    main_args.extend(args)
    
    # Run the command
    subprocess.run(main_args, check=True)

def print_help():
    """Print help information."""
    help_text = """
    LLM Influence Analysis Framework - Simple Wrapper
    
    Commands:
        train           Train the model from scratch
        analysis        Run the full analysis pipeline (train, factors, scores, inspection)
        factors         Compute influence factors for the trained model
        scores          Compute influence scores for the prompts
        evaluate        Run both custom and DeepEval evaluations
        custom-eval     Run just the custom evaluation
        deepeval-eval   Run just the DeepEval evaluation
        help            Show this help message
    
    Examples:
        python run.py train              # Train the model
        python run.py analysis           # Run the full analysis pipeline
        python run.py evaluate           # Run both evaluations
        python run.py --config my.yaml train  # Use a custom config file
    """
    print(help_text)

def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == "help":
        print_help()
        return
    
    try:
        run_main_command(args.config, args.command)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 