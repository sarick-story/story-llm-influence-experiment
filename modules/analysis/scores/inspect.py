"""
Influence Scores Inspection Module

This module analyzes the computed influence scores and generates a report highlighting
the most influential training examples for each prompt.
"""

import json
import os
import torch
import numpy as np
import pandas as pd
from kronfluence.analyzer import Analyzer
from datasets import load_dataset
import logging
from pathlib import Path

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
    num_samples = config['dataset']['num_samples']
    
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

def inspect_scores(config):
    """
    Inspect influence scores and generate a report.
    
    Args:
        config: Configuration dictionary
    """
    scores_name = config['scores']['all_layers_name']
    num_influential = config['scores'].get('num_influential', 10)
    
    logger.info(f"Inspecting influence scores: {scores_name}")
    logger.info(f"Number of influential examples to show: {num_influential}")
    
    # Create analyzer
    analyzer = Analyzer(
        analysis_name="influence_results",
        model=None,  # No need for model when just inspecting scores
        task=None,   # No need for task when just inspecting scores
    )
    
    # Load scores
    try:
        scores_dict = analyzer.load_pairwise_scores(scores_name)
        all_scores = scores_dict["all_modules"]
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
    
    # Generate report
    report_file = f"{scores_name}_report.md"
    
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
                if "text" in dataset.column_names:
                    example_text = dataset[idx]["text"]
                elif "description" in dataset.column_names:
                    example_text = dataset[idx]["description"]
                else:
                    # Default to the first column
                    example_text = dataset[idx][dataset.column_names[0]]
                
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
                if "text" in dataset.column_names:
                    example_text = dataset[idx]["text"]
                elif "description" in dataset.column_names:
                    example_text = dataset[idx]["description"]
                else:
                    # Default to the first column
                    example_text = dataset[idx][dataset.column_names[0]]
                
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
    
    return report_file 