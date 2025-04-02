import argparse
import json
import logging
import os
import numpy as np
from datasets import load_dataset
import torch
import glob
from pathlib import Path

from kronfluence.analyzer import Analyzer

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect influence scores for small language model.")

    parser.add_argument(
        "--scores_name",
        type=str,
        default="tinyllama_prompt_scores",  # Updated default to match your computed scores
        help="Name of the scores to inspect.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="prompts.json",
        help="JSON file with prompts that were used for computing influence.",
    )
    parser.add_argument(
        "--num_influential",
        type=int,
        default=5,
        help="Number of most influential training examples to show.",
    )
    args = parser.parse_args()

    return args

def find_available_scores():
    """Find all available score files in the influence_results directory."""
    base_path = "./influence_results/influence_results/scores_*"
    
    logger = logging.getLogger(__name__)
    available_scores = []
    for path in glob.glob(base_path):
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "pairwise_scores.safetensors")):
            # Get the name without removing "scores_" prefix
            score_name = os.path.basename(path)
            logger.debug(f"Found score directory: {score_name}")
            # Now remove the "scores_" prefix for the actual score name
            score_name = score_name.replace("scores_", "")
            available_scores.append(score_name)
    
    return available_scores

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Hardcoded approach - try the path directly first
    scores_path = f"./influence_results/influence_results/scores_{args.scores_name}/pairwise_scores.safetensors"
    logger.info(f"Trying to load scores directly from: {scores_path}")
    
    scores = None
    if os.path.exists(scores_path):
        try:
            scores = Analyzer.load_file(scores_path)["all_modules"]
            # Convert to float32 to ensure compatibility with numpy
            scores = scores.to(torch.float32)
            logger.info(f"Successfully loaded scores from {scores_path}")
            logger.info(f"Scores shape: {scores.shape}")
        except Exception as e:
            logger.error(f"Error loading scores from {scores_path}: {e}")
    
    # If direct path didn't work, fall back to detecting available scores
    if scores is None:
        # Find all available scores
        available_scores = find_available_scores()
        if available_scores:
            logger.info(f"Found available scores: {', '.join(available_scores)}")
        else:
            logger.warning("No score files found. Make sure you've computed scores first.")
        
        # Check if requested scores exist
        if args.scores_name not in available_scores:
            logger.error(f"Requested scores '{args.scores_name}' not found in available scores.")
            if available_scores:
                logger.info(f"Available scores: {', '.join(available_scores)}")
                # Try to use the first available score instead
                logger.info(f"Falling back to '{available_scores[0]}'")
                args.scores_name = available_scores[0]
            else:
                logger.error("No scores available. Please compute scores first.")
                return
        
        # Try to load scores from the fallback path
        scores_path = f"./influence_results/influence_results/scores_{args.scores_name}/pairwise_scores.safetensors"
        logger.info(f"Trying to load scores from fallback path: {scores_path}")
        if os.path.exists(scores_path):
            try:
                scores = Analyzer.load_file(scores_path)["all_modules"]
                # Convert to float32 to ensure compatibility with numpy
                scores = scores.to(torch.float32)
                logger.info(f"Successfully loaded scores from {scores_path}")
                logger.info(f"Scores shape: {scores.shape}")
            except Exception as e:
                logger.error(f"Error loading scores from {scores_path}: {e}")
    
    if scores is None:
        # Last desperate attempt - try to load the file we know exists from the terminal output
        absolute_path = "/root/training_run/story-llm-influence-experiment/influence_results/influence_results/scores_tinyllama_prompt_scores_all_layers/pairwise_scores.safetensors"
        logger.info(f"Last attempt - trying absolute path: {absolute_path}")
        if os.path.exists(absolute_path):
            try:
                scores = Analyzer.load_file(absolute_path)["all_modules"]
                # Convert to float32 to ensure compatibility with numpy
                scores = scores.to(torch.float32)
                logger.info(f"Successfully loaded scores from {absolute_path}")
                logger.info(f"Scores shape: {scores.shape}")
            except Exception as e:
                logger.error(f"Error loading scores from {absolute_path}: {e}")
    
    if scores is None:
        logger.error("Failed to load scores from any location.")
        return
    
    # Load prompts
    try:
        with open(args.prompts_file, 'r') as f:
            prompts = json.load(f)
    except Exception as e:
        logger.error(f"Error loading prompts file: {e}")
        return
    
    # Load dataset to get texts of influential examples
    try:
        dataset = load_dataset("Trelis/big_patent_sample", split="train[:10000]")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # For each prompt, find the most influential training examples
    for prompt_idx, prompt_data in enumerate(prompts):
        prompt = prompt_data["prompt"]
        completion = prompt_data["completion"]
        
        print(f"\n{'='*80}")
        print(f"Query {prompt_idx + 1}: {prompt}{completion}")
        print(f"{'='*80}")
        
        # Get scores for this prompt
        prompt_scores = scores[prompt_idx]
        
        # Find the most influential examples (both positive and negative)
        most_positive_indices = np.argsort(-prompt_scores)[:args.num_influential]
        most_negative_indices = np.argsort(prompt_scores)[:args.num_influential]
        
        # Print most positive influences
        print("\nTop Positive Influences:")
        print(f"{'='*80}")
        for i, idx in enumerate(most_positive_indices):
            score = prompt_scores[idx]
            description = dataset[int(idx)]["description"]
            
            # Truncate text if too long
            if len(description) > 500:
                description = description[:500] + "..."
                
            print(f"Rank = {i}; Score = {score:.2f}")
            print(description)
            print(f"{'='*80}")
            
        # Print most negative influences
        print("\nTop Negative Influences:")
        print(f"{'='*80}")
        for i, idx in enumerate(most_negative_indices):
            score = prompt_scores[idx]
            description = dataset[int(idx)]["description"]
            
            # Truncate text if too long
            if len(description) > 500:
                description = description[:500] + "..."
                
            print(f"Rank = {i}; Score = {score:.2f}")
            print(description)
            print(f"{'='*80}")
    
    # Save a report
    with open(f"{args.scores_name}_report.md", 'w') as f:
        for prompt_idx, prompt_data in enumerate(prompts):
            prompt = prompt_data["prompt"]
            completion = prompt_data["completion"]
            
            f.write(f"## Query {prompt_idx + 1}: {prompt}{completion}\n\n")
            
            # Get scores for this prompt
            prompt_scores = scores[prompt_idx]
            
            # Find the most influential examples
            most_positive_indices = np.argsort(-prompt_scores)[:args.num_influential]
            
            # Write most positive influences
            f.write("### Top Influential Training Examples:\n\n")
            for i, idx in enumerate(most_positive_indices):
                score = prompt_scores[idx]
                description = dataset[int(idx)]["description"]
                
                # Truncate text if too long
                if len(description) > 500:
                    description = description[:500] + "..."
                    
                f.write(f"**Rank {i+1}** (Score: {score:.2f})\n\n")
                f.write(f"```\n{description}\n```\n\n")
    
    logger.info(f"Report saved to {args.scores_name}_report.md")

if __name__ == "__main__":
    main() 