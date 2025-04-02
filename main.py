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

# Import modules conditionally in each function instead of all at once

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
    # Create base results directory
    base_dir = Path(config['output']['base_dir'])
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create main output directories
    for directory in [
        config['output']['influence_results'],
        config['output']['comparison_results'],
        config['output']['olmes_results'],
        config['output']['combined_results']
    ]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create additional subdirectories
    # Factors and scores directories
    Path(os.path.join(config['output']['influence_results'], config['factors'].get('output_dir', 'factors'))).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(config['output']['influence_results'], config['scores'].get('output_dir', 'scores'))).mkdir(parents=True, exist_ok=True)
    
    # Generated answers directory
    generated_dir = config['evaluation'].get('generated_dir', 'results/generated')
    Path(generated_dir).mkdir(parents=True, exist_ok=True)
    
    # Log directory if not created yet
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Created output directories: {base_dir}")

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
    inspect_factors_parser.add_argument("--layer", type=int, default=None, help="Layer to inspect (if not specified, will use from config)")
    inspect_factors_parser.add_argument("--clip_percentile", type=float, default=None, help="Percentile for clipping extreme values (default: from config or 99.5)")
    inspect_factors_parser.add_argument("--cmap", type=str, default=None, help="Colormap for visualizations (default: from config or 'coolwarm')")
    
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
    # Only import the training module when needed
    from modules.training import train_model
    
    logger.info("Starting model training...")
    train_model(config)
    logger.info("Model training completed")

def run_compute_factors(config, logger):
    """Compute influence factors."""
    # Only import the factors module when needed
    from modules.analysis.factors import compute_factors
    
    logger.info("Starting influence factor computation...")
    compute_factors(config)
    logger.info("Factor computation completed")

def run_inspect_factors(config, layer, clip_percentile=None, cmap=None, logger=None):
    """Inspect influence factors for a specific layer."""
    # Only import the factors module when needed
    from modules.analysis.factors import inspect_factors
    
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # If layer is not specified, use the layer from config
    if layer is None:
        # Get the inspection layer from config
        layer = config['factors'].get('inspection_layer', 11)  # Fallback to 11 if not in config
        logger.info(f"No layer specified, using layer {layer} from config")
    
    logger.info(f"Inspecting influence factors for layer {layer}...")
    
    # Get visualization parameters from config or use defaults
    vis_config = config['factors'].get('visualization', {})
    
    # Command line args take precedence over config
    if clip_percentile is None:
        clip_percentile = vis_config.get('clip_percentile', 99.5)
    
    if cmap is None:
        cmap = vis_config.get('cmap', 'coolwarm')
    
    logger.info(f"Using visualization parameters: clip_percentile={clip_percentile}, cmap={cmap}")
    inspect_factors(config, layer, clip_percentile, cmap)
    logger.info("Factor inspection completed")

def run_compute_scores(config, use_generated, logger):
    """Compute influence scores."""
    # Only import the scores module when needed
    from modules.analysis.scores import compute_scores
    
    logger.info("Computing influence scores...")
    compute_scores(config, use_generated)
    logger.info("Score computation completed")

def run_inspect_scores(config, logger):
    """Inspect influence scores."""
    # Only import the scores module when needed
    from modules.analysis.scores import inspect_scores
    
    logger.info("Inspecting influence scores...")
    inspect_scores(config)
    logger.info("Score inspection completed")

def run_custom_evaluation(config, logger):
    """Run custom model evaluation."""
    # Only import the evaluation modules when needed
    from modules.evaluation.custom import generate_model_answers, compare_models
    from modules.analysis.scores import compute_scores
    
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
    # Only import the OLMES module when needed
    from modules.evaluation.olmes import run_olmes_evaluation as run_olmes
    
    logger.info("Starting OLMES evaluation...")
    run_olmes(config)
    logger.info("OLMES evaluation completed")

def run_combined_evaluation(config, logger):
    """Run both custom and OLMES evaluations and combine results."""
    # Only import the reporting module when needed
    from modules.evaluation.reporting import combine_evaluation_results
    
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
    # Import each module only when it's about to be used
    logger.info("Starting full analysis pipeline...")
    
    # Step 1: Train the model
    run_train(config, logger)
    
    # Step 2: Compute influence factors
    run_compute_factors(config, logger)
    
    # Step 3: Inspect factors 
    # Use the inspection layer from config
    inspection_layer = config['factors'].get('inspection_layer', 11)  # Fallback to 11 if not in config
    # Get visualization parameters from config
    vis_config = config['factors'].get('visualization', {})
    clip_percentile = vis_config.get('clip_percentile', 99.5)
    cmap = vis_config.get('cmap', 'coolwarm')
    run_inspect_factors(config, inspection_layer, clip_percentile, cmap, logger)
    
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
        run_inspect_factors(config, args.layer, args.clip_percentile, args.cmap, logger)
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