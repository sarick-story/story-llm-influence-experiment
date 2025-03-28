import argparse
import json
import logging
import os
import numpy as np
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
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load the scores
    logger.info(f"Loading scores: {args.scores_name}")
    try:
        # Use static method instead of creating an Analyzer instance
        scores_path = f"./influence_results/influence_results/tiny_lm_factors/scores_{args.scores_name}/pairwise_scores.safetensors"
        scores = Analyzer.load_file(scores_path)["all_modules"]
        logger.info(f"Scores shape: {scores.shape}")
    except Exception as e:
        logger.error(f"Error loading scores: {e}")
        return
    
    # Load prompts
    with open(args.prompts_file, 'r') as f:
        prompts = json.load(f)
    
    # Load dataset to get texts of influential examples
    dataset = load_dataset("openwebtext", split="train[:1000]")
    
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
            score = prompt_scores[idx]
            text = dataset[int(idx)]["text"]
            
            # Truncate text if too long
            if len(text) > 500:
                text = text[:500] + "..."
                
            print(f"Rank = {i}; Score = {score:.2f}")
            print(text)
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
                text = dataset[int(idx)]["text"]
                
                # Truncate text if too long
                if len(text) > 500:
                    text = text[:500] + "..."
                    
                f.write(f"**Rank {i+1}** (Score: {score:.2f})\n\n")
                f.write(f"```\n{text}\n```\n\n")
    
    logger.info(f"Report saved to {args.scores_name}_report.md")

if __name__ == "__main__":
    main() 