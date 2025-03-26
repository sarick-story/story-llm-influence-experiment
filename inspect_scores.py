import argparse
import json
import logging
import os
import numpy as np
import torch
from datasets import load_dataset

from kronfluence.analyzer import Analyzer

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect influence scores for small language model.")

    parser.add_argument(
        "--scores_name",
        type=str,
        default="prompt_scores",
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
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of training examples that were used."
    )
    parser.add_argument(
        "--handle_inf",
        choices=["skip", "clip", "rank"],
        default="clip",
        help="How to handle inf values: skip (ignore), clip (replace with max finite), rank (keep but use rank not magnitude)"
    )
    parser.add_argument(
        "--clip_value",
        type=float,
        default=1e6,
        help="Value to clip scores to if using handle_inf=clip"
    )
    args = parser.parse_args()

    return args

def clean_scores(scores, method="clip", clip_value=1e6):
    """Clean scores by handling NaN and inf values."""
    # Convert to PyTorch tensor if it's a numpy array
    if isinstance(scores, np.ndarray):
        scores = torch.tensor(scores)
    
    # Detect problematic values
    inf_mask = torch.isinf(scores)
    nan_mask = torch.isnan(scores)
    num_inf = inf_mask.sum().item()
    num_nan = nan_mask.sum().item()
    
    if num_inf > 0 or num_nan > 0:
        logging.warning(f"Found {num_inf} inf values and {num_nan} NaN values in scores")
    
    # Handle based on method
    if method == "skip":
        # Replace inf and nan with 0
        scores = torch.where(inf_mask | nan_mask, torch.tensor(0.0), scores)
    elif method == "clip":
        # Replace inf with clip_value (preserving sign)
        pos_inf_mask = (scores == float('inf'))
        neg_inf_mask = (scores == float('-inf'))
        scores = torch.where(pos_inf_mask, torch.tensor(clip_value), scores)
        scores = torch.where(neg_inf_mask, torch.tensor(-clip_value), scores)
        # Replace nan with 0
        scores = torch.where(nan_mask, torch.tensor(0.0), scores)
    # For "rank" method, we don't modify the scores but rely on argsort, which handles inf correctly
    
    return scores

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load the scores
    logger.info(f"Loading scores: {args.scores_name}")
    try:
        # Try multiple possible paths for the scores file
        possible_paths = [
            f"./influence_results/tiny_lm_influence/scores_{args.scores_name}/pairwise_scores.safetensors",
            f"./influence_results/influence_results/tiny_lm_factors/scores_{args.scores_name}/pairwise_scores.safetensors",
            f"./influence_results/tiny_lm_factors/scores_{args.scores_name}/pairwise_scores.safetensors"
        ]
        
        scores = None
        used_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found scores at: {path}")
                scores = Analyzer.load_file(path)["all_modules"]
                used_path = path
                break
        
        if scores is None:
            raise FileNotFoundError(f"Could not find scores file in any of these locations: {possible_paths}")
        
        logger.info(f"Loaded scores from: {used_path}")
        logger.info(f"Scores shape: {scores.shape}")
    except Exception as e:
        logger.error(f"Error loading scores: {e}")
        return
    
    # Load prompts
    with open(args.prompts_file, 'r') as f:
        prompts = json.load(f)
    
    # Load dataset to get texts of influential examples
    logger.info(f"Loading dataset with {args.num_samples} examples")
    dataset = load_dataset("openwebtext", split=f"train[:{args.num_samples}]")
    
    # Create a report file
    report_file = f"{args.scores_name}_report.md"
    with open(report_file, 'w') as f:
        f.write(f"# Influence Analysis Report for {args.scores_name}\n\n")
        f.write(f"Analysis of {len(prompts)} prompts, showing the top {args.num_influential} influential training examples for each.\n\n")
        
        # For each prompt, find the most influential training examples
        for prompt_idx, prompt_data in enumerate(prompts):
            prompt = prompt_data["prompt"]
            completion = prompt_data["completion"]
            
            print(f"\n{'='*80}")
            print(f"Query {prompt_idx + 1}: {prompt}{completion}")
            print(f"{'='*80}")
            
            # Get scores for this prompt
            prompt_scores = scores[prompt_idx]
            
            # Clean the scores to handle inf/nan values
            cleaned_scores = clean_scores(prompt_scores, method=args.handle_inf, clip_value=args.clip_value)
            
            # Convert BFloat16 to float32 before numpy conversion
            cleaned_scores = cleaned_scores.float()
            
            # Find the most influential examples (both positive and negative)
            most_positive_indices = np.argsort(-cleaned_scores.numpy())[:args.num_influential]
            most_negative_indices = np.argsort(cleaned_scores.numpy())[:args.num_influential]
            
            # Print most positive influences
            print("\nTop Positive Influences:")
            print(f"{'='*80}")
            for i, idx in enumerate(most_positive_indices):
                score = prompt_scores[idx].item()  # Use original score for display
                text = dataset[int(idx)]["text"]
                
                # Truncate text if too long
                if len(text) > 500:
                    text = text[:500] + "..."
                    
                print(f"Rank = {i}; Score = {score:.2f}")
                print(text)
                print(f"{'='*80}")
                
            # Print most negative influences
            print("\nTop Negative Influences:")
            print(f"{'='*80}")
            for i, idx in enumerate(most_negative_indices):
                score = prompt_scores[idx].item()  # Use original score for display
                text = dataset[int(idx)]["text"]
                
                # Truncate text if too long
                if len(text) > 500:
                    text = text[:500] + "..."
                    
                print(f"Rank = {i}; Score = {score:.2f}")
                print(text)
                print(f"{'='*80}")
            
            # Write to report file
            f.write(f"## Query {prompt_idx + 1}: {prompt}{completion}\n\n")
            
            # Write most positive influences
            f.write("### Top Positive Influences:\n\n")
            for i, idx in enumerate(most_positive_indices):
                score = prompt_scores[idx].item()
                text = dataset[int(idx)]["text"]
                
                # Truncate text if too long
                if len(text) > 500:
                    text = text[:500] + "..."
                
                # Format the score display
                if np.isinf(score) and score > 0:
                    score_display = "inf"
                elif np.isinf(score) and score < 0:
                    score_display = "-inf"
                elif np.isnan(score):
                    score_display = "NaN"
                else:
                    score_display = f"{score:.2f}"
                
                f.write(f"#### Rank = {i}; Score = {score_display}\n")
                f.write(f"```\n{text}\n```\n\n")
            
            # Write most negative influences
            f.write("### Top Negative Influences:\n\n")
            for i, idx in enumerate(most_negative_indices):
                score = prompt_scores[idx].item()
                text = dataset[int(idx)]["text"]
                
                # Truncate text if too long
                if len(text) > 500:
                    text = text[:500] + "..."
                
                # Format the score display
                if np.isinf(score) and score > 0:
                    score_display = "inf"
                elif np.isinf(score) and score < 0:
                    score_display = "-inf"
                elif np.isnan(score):
                    score_display = "NaN"
                else:
                    score_display = f"{score:.2f}"
                
                f.write(f"#### Rank = {i}; Score = {score_display}\n")
                f.write(f"```\n{text}\n```\n\n")
            
            f.write("---\n\n")  # Add separator between prompts
    
    logger.info(f"Report saved to {report_file}")
    logger.info("Done!")

if __name__ == "__main__":
    main() 