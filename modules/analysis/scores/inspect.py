"""
Influence Scores Inspection Module

This module analyzes the computed influence scores and generates a report highlighting
the most influential training examples for each prompt.
"""

import json
import os
import torch
import wandb
import numpy as np
import pandas as pd
import glob
from pathlib import Path
from kronfluence.analyzer import Analyzer
from datasets import load_dataset
import logging
from modules.utils.wandb_utils import init_wandb

logger = logging.getLogger(__name__)

def format_example(text, max_length=100):
    """Format a training example for display in the report."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def get_dataset_examples(config, dataset_type='main'):
    """Load examples from the dataset."""
    # We now use the same dataset for all operations
    dataset_name = config['dataset']['name']
    # Use analysis_samples instead of num_samples for influence analysis
    num_samples = config['dataset'].get('analysis_samples', config['dataset']['num_samples'])
    
    # Get the text column name from config or default to common options
    text_column = config['dataset'].get('text_column')
    
    logger.info(f"Loading dataset examples from: {dataset_name} (samples: {num_samples})")
    
    if num_samples > 0:
        dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    return dataset

def get_prompts(config):
    """Load prompts from the configuration file."""
    prompts_file = config['general']['prompts_file']
    
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file {prompts_file} not found")
    
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    return prompts

def find_scores_file(scores_name):
    """Find the scores file in various possible locations."""
    possible_paths = [
        f"./results/influence/scores/scores_{scores_name}/pairwise_scores.safetensors",
        f"./influence_results/scores/scores_{scores_name}/pairwise_scores.safetensors",
        f"./influence_results/influence_results/scores_{scores_name}/pairwise_scores.safetensors",
        f"./results/influence/scores/scores_{scores_name}",
        f"./influence_results/results/influence/scores/scores_{scores_name}/pairwise_scores.safetensors",
    ]
    
    # Also try to find any scores_ directory
    score_dirs = glob.glob("./influence_results/**/scores_*/pairwise_scores.safetensors", recursive=True)
    possible_paths.extend(score_dirs)
    
    for path in possible_paths:
        logger.info(f"Checking for scores file at: {path}")
        if os.path.exists(path):
            if os.path.isdir(path):
                # If it's a directory, look for the pairwise_scores.safetensors file inside
                file_path = os.path.join(path, "pairwise_scores.safetensors")
                if os.path.exists(file_path):
                    logger.info(f"Found scores file at: {file_path}")
                    return file_path
            else:
                logger.info(f"Found scores file at: {path}")
                return path
    
    # Last resort - try to find any pairwise_scores.safetensors file
    logger.info("Searching for any pairwise_scores.safetensors file...")
    all_score_files = glob.glob("**/pairwise_scores.safetensors", recursive=True)
    if all_score_files:
        logger.info(f"Found potential score files: {all_score_files}")
        return all_score_files[0]
    
    return None

def inspect_scores(config):
    """
    Inspect influence scores and generate a report.
    
    Args:
        config: Configuration dictionary
    """
    scores_name = config['scores']['all_layers_name']
    num_influential = config['scores'].get('num_influential', 10)
    
    # Initialize wandb with a unique run name
    run = init_wandb(config, "inspect_scores")
    
    logger.info(f"Inspecting influence scores: {scores_name}")
    logger.info(f"Number of influential examples to show: {num_influential}")
    
    # Try to find and load the scores file directly
    scores_file = find_scores_file(scores_name)
    
    if not scores_file:
        logger.error(f"Could not find scores file for {scores_name}")
        raise FileNotFoundError(f"Scores file for {scores_name} not found")
    
    logger.info(f"Loading scores from: {scores_file}")
    
    try:
        # Load the scores using Analyzer.load_file
        scores_dict = Analyzer.load_file(scores_file)
        all_scores = scores_dict["all_modules"]
        # Convert to float32 for consistency
        if hasattr(all_scores, 'to'):
            all_scores = all_scores.to(torch.float32)
        logger.info(f"Scores loaded successfully. Shape: {all_scores.shape}")
    except Exception as e:
        logger.error(f"Error loading scores: {e}")
        raise
    
    # Load dataset examples
    dataset = get_dataset_examples(config)
    
    # Load prompts
    prompts = get_prompts(config)
    
    # Convert scores to numpy if they're torch tensors
    if isinstance(all_scores, torch.Tensor):
        all_scores = all_scores.cpu().numpy()
    
    # Generate report in the proper directory
    scores_dir = os.path.join(config['output']['influence_results'], config['scores'].get('output_dir', 'scores'))
    os.makedirs(scores_dir, exist_ok=True)
    report_filename = f"{scores_name}_report.md"
    report_file = os.path.join(scores_dir, report_filename)
    
    logger.info(f"Generating influence scores report: {report_file}")
    
    with open(report_file, "w") as f:
        f.write(f"# Influence Scores Analysis: {scores_name}\n\n")
        
        # For each prompt
        for prompt_idx, prompt in enumerate(prompts):
            prompt_text = prompt["prompt"]
            completion = prompt["completion"]
            
            f.write(f"## Prompt {prompt_idx + 1}: \"{prompt_text}\"\n\n")
            f.write(f"**Completion:** {completion}\n\n")
            
            # Get the scores for this prompt
            prompt_scores = all_scores[prompt_idx]
            
            # Sort indices by score (descending for positive influence)
            most_influential_idx = np.argsort(-prompt_scores)[:num_influential]
            least_influential_idx = np.argsort(prompt_scores)[:num_influential]
            
            # Most influential examples (positive influence)
            f.write("### Most Influential Examples (Positive Influence)\n\n")
            f.write("| Rank | Score | Example |\n")
            f.write("|------|-------|--------|\n")
            
            for rank, idx in enumerate(most_influential_idx):
                score = prompt_scores[idx]
                
                # Get the example text
                text_field = None
                column_names = dataset.column_names
                
                text_column = config['dataset'].get('text_column')
                if text_column and text_column in column_names:
                    # Use the configured text column if it exists
                    text_field = text_column
                elif 'text' in column_names:
                    text_field = 'text'
                elif 'description' in column_names:
                    text_field = 'description'
                else:
                    text_field = column_names[0]  # Fall back to the first column
                
                example_text = dataset[int(idx)][text_field]
                
                example_formatted = format_example(example_text)
                f.write(f"| {rank+1} | {score:.6f} | {example_formatted} |\n")
            
            f.write("\n")
            
            # Least influential examples (negative influence)
            f.write("### Least Influential Examples (Negative Influence)\n\n")
            f.write("| Rank | Score | Example |\n")
            f.write("|------|-------|--------|\n")
            
            for rank, idx in enumerate(least_influential_idx):
                score = prompt_scores[idx]
                
                # Get the example text
                text_field = None
                column_names = dataset.column_names
                
                text_column = config['dataset'].get('text_column')
                if text_column and text_column in column_names:
                    # Use the configured text column if it exists
                    text_field = text_column
                elif 'text' in column_names:
                    text_field = 'text'
                elif 'description' in column_names:
                    text_field = 'description'
                else:
                    text_field = column_names[0]  # Fall back to the first column
                
                example_text = dataset[int(idx)][text_field]
                
                example_formatted = format_example(example_text)
                f.write(f"| {rank+1} | {score:.6f} | {example_formatted} |\n")
            
            f.write("\n")
            
            # Add a statistical summary
            f.write("### Statistical Summary\n\n")
            f.write(f"- **Maximum influence score:** {np.max(prompt_scores):.6f}\n")
            f.write(f"- **Minimum influence score:** {np.min(prompt_scores):.6f}\n")
            f.write(f"- **Mean influence score:** {np.mean(prompt_scores):.6f}\n")
            f.write(f"- **Median influence score:** {np.median(prompt_scores):.6f}\n")
            f.write(f"- **Standard deviation:** {np.std(prompt_scores):.6f}\n")
            
            # Add percentile information
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            f.write("\n**Percentiles:**\n\n")
            for p in percentiles:
                f.write(f"- **{p}th percentile:** {np.percentile(prompt_scores, p):.6f}\n")
            
            f.write("\n---\n\n")
    
    logger.info(f"Report generated: {report_file}")
    
    # Log results to wandb
    if wandb.run is not None:
        # Log summary statistics
        stats_by_prompt = []
        for prompt_idx, prompt in enumerate(prompts):
            prompt_scores = all_scores[prompt_idx]
            stats = {
                "prompt_idx": prompt_idx,
                "prompt_text": prompt["prompt"][:100] + "..." if len(prompt["prompt"]) > 100 else prompt["prompt"],
                "max_score": float(np.max(prompt_scores)),
                "min_score": float(np.min(prompt_scores)),
                "mean_score": float(np.mean(prompt_scores)),
                "median_score": float(np.median(prompt_scores)),
                "std_score": float(np.std(prompt_scores))
            }
            stats_by_prompt.append(stats)
        
        # Create a wandb Table
        columns = list(stats_by_prompt[0].keys())
        data = [[row[col] for col in columns] for row in stats_by_prompt]
        table = wandb.Table(columns=columns, data=data)
        
        # Log metrics
        wandb.log({
            "scores_report": table,
            "num_prompts": len(prompts),
            "num_influential_examples": num_influential,
            "report_file": report_file,
            "scores_name": scores_name
        })
        
        # Upload the report file from the correct location
        wandb.save(report_file)
    
    return report_file 