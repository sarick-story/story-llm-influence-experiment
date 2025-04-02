#!/usr/bin/env python3
"""
LLM Influence Analysis & Evaluation Framework - Main Orchestrator

This script serves as the central entry point for all operations in the LLM influence analysis
and evaluation framework. It reads configuration from a YAML file and coordinates the execution
of various tasks: training, factor computation, score analysis, and evaluation.
"""

import argparse
import os
import sys
import yaml
import logging
from datetime import datetime
from pathlib import Path

# Add custom packages to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import modules - these will be created in subsequent steps
from modules.training import train_model
from modules.analysis.factors import compute_factors, inspect_factors
from modules.analysis.scores import compute_scores, inspect_scores
from modules.evaluation.custom import generate_model_answers, compare_models
from modules.evaluation.olmes import run_olmes_evaluation
from modules.evaluation.reporting import combine_evaluation_results

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging():
    """Configure logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"llm_influence_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_output_directories(config):
    """Create output directories defined in the configuration."""
    for directory in [
        config['output']['influence_results'],
        config['output']['comparison_results'],
        config['output']['olmes_results'],
        config['output']['combined_results']
    ]:
        Path(directory).mkdir(parents=True, exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM Influence Analysis & Evaluation Framework"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train model command
    train_parser = subparsers.add_parser("train", help="Train the model")
    
    # Compute factors command
    factors_parser = subparsers.add_parser("compute_factors", help="Compute influence factors")
    
    # Inspect factors command
    inspect_factors_parser = subparsers.add_parser("inspect_factors", help="Inspect influence factors")
    inspect_factors_parser.add_argument("--layer", type=int, default=21, help="Layer to inspect")
    
    # Compute scores command
    scores_parser = subparsers.add_parser("compute_scores", help="Compute influence scores")
    scores_parser.add_argument("--use_generated", action="store_true", help="Use generated answers")
    
    # Inspect scores command
    inspect_scores_parser = subparsers.add_parser("inspect_scores", help="Inspect influence scores")
    
    # Evaluation commands
    eval_parser = subparsers.add_parser("evaluate", help="Run model evaluation")
    eval_parser.add_argument(
        "--type", 
        choices=["custom", "olmes", "all"], 
        default="all",
        help="Type of evaluation to run (default: all)"
    )
    
    # Full analysis command
    full_parser = subparsers.add_parser("run_full_analysis", help="Run full analysis pipeline")
    
    return parser.parse_args()

def run_train(config, logger):
    """Run model training."""
    logger.info("Starting model training...")
    train_model(config)
    logger.info("Model training completed")

def run_compute_factors(config, logger):
    """Compute influence factors."""
    logger.info("Starting influence factor computation...")
    compute_factors(config)
    logger.info("Factor computation completed")

def run_inspect_factors(config, layer, logger):
    """Inspect influence factors for a specific layer."""
    logger.info(f"Inspecting influence factors for layer {layer}...")
    inspect_factors(config, layer)
    logger.info("Factor inspection completed")

def run_compute_scores(config, use_generated, logger):
    """Compute influence scores."""
    logger.info("Computing influence scores...")
    compute_scores(config, use_generated)
    logger.info("Score computation completed")

def run_inspect_scores(config, logger):
    """Inspect influence scores."""
    logger.info("Inspecting influence scores...")
    inspect_scores(config)
    logger.info("Score inspection completed")

def run_custom_evaluation(config, logger):
    """Run custom model evaluation."""
    logger.info("Starting custom model evaluation...")
    
    # Generate answers from both models
    logger.info("Generating model answers...")
    generate_model_answers(config)
    
    # Compute scores using generated answers
    logger.info("Computing influence scores for generated answers...")
    compute_scores(config, use_generated=True)
    
    # Compare models and analyze influences
    logger.info("Comparing models and analyzing influences...")
    compare_models(config)
    
    logger.info("Custom evaluation completed")

def run_olmes_evaluation(config, logger):
    """Run OLMES evaluation."""
    logger.info("Starting OLMES evaluation...")
    run_olmes_evaluation(config)
    logger.info("OLMES evaluation completed")

def run_combined_evaluation(config, logger):
    """Run both custom and OLMES evaluations and combine results."""
    logger.info("Starting comprehensive evaluation...")
    
    # Run custom evaluation
    run_custom_evaluation(config, logger)
    
    # Run OLMES evaluation
    run_olmes_evaluation(config, logger)
    
    # Combine results
    logger.info("Combining evaluation results...")
    combine_evaluation_results(config)
    
    logger.info("Comprehensive evaluation completed")

def run_full_analysis(config, logger):
    """Run the full analysis pipeline."""
    logger.info("Starting full analysis pipeline...")
    
    # Step 1: Train the model
    run_train(config, logger)
    
    # Step 2: Compute influence factors
    run_compute_factors(config, logger)
    
    # Step 3: Inspect factors for the last layer
    run_inspect_factors(config, 21, logger)
    
    # Step 4: Compute influence scores
    run_compute_scores(config, False, logger)
    
    # Step 5: Inspect scores
    run_inspect_scores(config, logger)
    
    # Step 6: Run comprehensive evaluation
    run_combined_evaluation(config, logger)
    
    logger.info("Full analysis pipeline completed")

def main():
    """Main entry point."""
    args = parse_args()
    logger = setup_logging()
    
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    create_output_directories(config)
    
    # Execute the requested command
    if args.command == "train":
        run_train(config, logger)
    elif args.command == "compute_factors":
        run_compute_factors(config, logger)
    elif args.command == "inspect_factors":
        run_inspect_factors(config, args.layer, logger)
    elif args.command == "compute_scores":
        run_compute_scores(config, args.use_generated, logger)
    elif args.command == "inspect_scores":
        run_inspect_scores(config, logger)
    elif args.command == "evaluate":
        if args.type == "custom":
            run_custom_evaluation(config, logger)
        elif args.type == "olmes":
            run_olmes_evaluation(config, logger)
        elif args.type == "all":
            run_combined_evaluation(config, logger)
    elif args.command == "run_full_analysis":
        run_full_analysis(config, logger)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main() 