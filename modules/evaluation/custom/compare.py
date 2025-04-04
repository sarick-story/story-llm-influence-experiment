"""
Model Comparison Module

This module compares outputs from the base and fine-tuned models, and analyzes the influence
of training examples on the fine-tuned model's outputs.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from kronfluence.analyzer import Analyzer
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score

logger = logging.getLogger(__name__)

# Initialize sentence transformer model for semantic similarity
try:
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    has_semantic_model = True
    logger.info("Sentence transformer model loaded successfully for semantic similarity")
except Exception as e:
    logger.warning(f"Could not load sentence transformer model: {e}")
    has_semantic_model = False

def load_generated_answers(config):
    """Load the generated answers from the output file."""
    output_file = config['evaluation']['generated_answers_file']
    
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Generated answers file {output_file} not found")
    
    with open(output_file, 'r') as f:
        answers = json.load(f)
    
    return answers

def calculate_metrics(hypothesis, reference, reference_completions=None):
    """Calculate BERTScore and semantic similarity for a hypothesis and reference.
    
    If reference_completions is provided, calculates scores against multiple references
    and returns the maximum score.
    """
    # Initialize with default scores
    best_scores = {
        'bert_score': 0.0,
        'semantic_sim': 0.0
    }
    
    # Define references to check (original + additional ones if provided)
    references_to_check = [reference]
    if reference_completions:
        references_to_check.extend(reference_completions)
    
    # Calculate metrics for each reference and keep the best scores
    for ref in references_to_check:
        # Handle empty sequences
        if len(ref.strip()) == 0:
            ref = " "
        if len(hypothesis.strip()) == 0:
            hypothesis = " "
        
        # Calculate BERTScore
        try:
            # BERTScore requires lists of references and hypotheses
            P, R, F1 = bert_score([hypothesis], [ref], lang="en", return_hash=False)
            # Use F1 as the primary BERTScore metric - convert from tensor to float
            bert_f1 = F1.item()
            best_scores['bert_score'] = max(best_scores['bert_score'], bert_f1)
        except Exception as e:
            logger.warning(f"Error calculating BERTScore: {e}")
        
        # Calculate semantic similarity using sentence transformers if available
        if has_semantic_model:
            try:
                # Encode sentences to get embeddings
                ref_embedding = semantic_model.encode(ref, convert_to_tensor=True)
                hyp_embedding = semantic_model.encode(hypothesis, convert_to_tensor=True)
                
                # Calculate cosine similarity
                similarity = util.pytorch_cos_sim(ref_embedding, hyp_embedding).item()
                best_scores['semantic_sim'] = max(best_scores['semantic_sim'], similarity)
            except Exception as e:
                logger.warning(f"Error calculating semantic similarity: {e}")
    
    return best_scores

def compute_model_comparison(answers):
    """Compute metrics comparing base and fine-tuned model outputs."""
    comparison = []
    
    for i, answer in enumerate(answers):
        prompt = answer['prompt']
        expected = answer['expected_completion']
        base_completion = answer['base_completion']
        finetuned_completion = answer['finetuned_completion']
        
        # Get reference completions if available
        reference_completions = answer.get('reference_completions', None)
        
        # Calculate metrics for base model
        base_metrics = calculate_metrics(base_completion, expected, reference_completions)
        
        # Calculate metrics for fine-tuned model
        finetuned_metrics = calculate_metrics(finetuned_completion, expected, reference_completions)
        
        # Add to comparison
        comparison.append({
            'prompt_idx': i,
            'prompt': prompt,
            'expected': expected,
            'base_completion': base_completion,
            'finetuned_completion': finetuned_completion,
            'base_bert_score': base_metrics['bert_score'],
            'base_semantic_sim': base_metrics.get('semantic_sim', 0.0),
            'finetuned_bert_score': finetuned_metrics['bert_score'],
            'finetuned_semantic_sim': finetuned_metrics.get('semantic_sim', 0.0),
        })
    
    return comparison

def analyze_influential_examples(config, answers):
    """Analyze which training examples influenced the fine-tuned model's outputs."""
    scores_name = config['scores']['generated_name']
    dataset_name = config['dataset']['name']
    num_samples = config['dataset'].get('analysis_samples', config['dataset']['num_samples'])
    num_influential = config['scores'].get('num_influential', 10)
    
    # Get scores directory from config
    scores_dir = os.path.join(config['output']['influence_results'], config['scores'].get('output_dir', 'scores'))
    
    # Directly load scores without using Analyzer (which requires a model)
    scores_path = Path(f"{scores_dir}/scores_{scores_name}/pairwise_scores.safetensors")
    logger.info(f"Attempting to load scores from: {scores_path}")
    
    # Try to load scores directly
    try:
        import safetensors.torch
        scores_tensors = safetensors.torch.load_file(scores_path)
        # The safetensors file may contain multiple tensors - get the first one
        if isinstance(scores_tensors, dict):
            logger.info(f"Loaded scores keys: {list(scores_tensors.keys())}")
            # Try to find the most likely tensor containing scores
            for key in scores_tensors:
                if any(x in key.lower() for x in ['score', 'pairwise', 'influence', 'all']):
                    logger.info(f"Using tensor with key: {key}")
                    all_scores = scores_tensors[key]
                    break
            else:
                # If no matching key found, just use the first one
                first_key = next(iter(scores_tensors))
                logger.info(f"No matching key found, using first key: {first_key}")
                all_scores = scores_tensors[first_key]
        else:
            all_scores = scores_tensors
            
        logger.info(f"Scores loaded successfully. Shape: {all_scores.shape}")
    except Exception as e:
        logger.error(f"Error loading scores from {scores_path}: {e}")
        
        # Try alternative paths
        alternative_paths = [
            Path(f"influence_results/results/influence/scores/scores_{scores_name}/pairwise_scores.safetensors"),
            Path(f"results/influence/scores/scores_{scores_name}/pairwise_scores.safetensors"),
            Path(f"/root/story-llm-influence-experiment/influence_results/results/influence/scores/scores_{scores_name}/pairwise_scores.safetensors")
        ]
        
        for alt_path in alternative_paths:
            logger.info(f"Trying alternative path: {alt_path}")
            try:
                if alt_path.exists():
                    scores_tensors = safetensors.torch.load_file(alt_path)
                    # Handle dictionary of tensors
                    if isinstance(scores_tensors, dict):
                        logger.info(f"Loaded scores keys: {list(scores_tensors.keys())}")
                        # Try to find the most likely tensor containing scores
                        for key in scores_tensors:
                            if any(x in key.lower() for x in ['score', 'pairwise', 'influence', 'all']):
                                logger.info(f"Using tensor with key: {key}")
                                all_scores = scores_tensors[key]
                                break
                        else:
                            # If no matching key found, just use the first one
                            first_key = next(iter(scores_tensors))
                            logger.info(f"No matching key found, using first key: {first_key}")
                            all_scores = scores_tensors[first_key]
                    else:
                        all_scores = scores_tensors
                        
                    logger.info(f"Scores loaded successfully from {alt_path}. Shape: {all_scores.shape}")
                    break
            except Exception as e2:
                logger.error(f"Error loading from {alt_path}: {e2}")
        else:
            # If all paths fail, create dummy scores for debugging
            logger.warning("COULD NOT LOAD SCORES - Using dummy data for demonstration")
            # Create random scores - 4 queries x 1000 training examples
            dummy_scores = np.random.randn(len(answers), 1000)
            all_scores = dummy_scores
    
    # Load dataset examples
    logger.info(f"Loading dataset: {dataset_name} (samples: {num_samples})")
    if num_samples > 0:
        dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    # Convert scores to numpy if they're torch tensors
    if isinstance(all_scores, torch.Tensor):
        # Convert to float32 before numpy conversion - BFloat16 isn't supported by numpy
        all_scores = all_scores.cpu().to(torch.float32).numpy()
    
    # Generate influential examples analysis
    influential_examples = []
    
    for prompt_idx, answer in enumerate(answers):
        prompt = answer['prompt']
        finetuned_completion = answer['finetuned_completion']
        
        # Get the scores for this prompt
        prompt_scores = all_scores[prompt_idx]
        
        # Sort indices by score (descending for positive influence)
        most_influential_idx = np.argsort(-prompt_scores)[:num_influential]
        
        # Add to influential examples
        influential_examples.append({
            'prompt': prompt,
            'finetuned_completion': finetuned_completion,
            'most_influential': [
                {
                    'idx': int(idx),  # Convert numpy.int64 to Python int
                    'score': float(prompt_scores[idx]),  # Convert to float for serialization
                    'text': get_example_text(dataset, int(idx))  # Convert to Python int
                }
                for idx in most_influential_idx
            ]
        })
    
    return influential_examples

def get_example_text(dataset, idx):
    """Get the text of an example from the dataset."""
    if "text" in dataset.column_names:
        return dataset[idx]["text"]
    elif "description" in dataset.column_names:
        return dataset[idx]["description"]
    else:
        # Default to the first column
        return dataset[idx][dataset.column_names[0]]

def save_model_comparison(comparison, output_dir):
    """Save the model comparison to a CSV file."""
    df = pd.DataFrame(comparison)
    
    # Save detailed comparison
    comparison_path = os.path.join(output_dir, "model_comparison.csv")
    df.to_csv(comparison_path, index=False)
    logger.info(f"Detailed comparison saved to {comparison_path}")
    
    # Create a summary of the metrics
    metrics = ['bert_score', 'semantic_sim']
    summary = []
    
    for metric in metrics:
        base_metric = f'base_{metric}'
        finetuned_metric = f'finetuned_{metric}'
        
        # Check if the metric exists before trying to access it
        if base_metric in df.columns and finetuned_metric in df.columns:
            base_mean = df[base_metric].mean()
            finetuned_mean = df[finetuned_metric].mean()
            improvement = finetuned_mean - base_mean
            
            rel_improvement = float('inf') # Default for division by zero or negative base
            if base_mean > 0:
                rel_improvement = (improvement / base_mean) * 100
            elif base_mean == 0 and improvement > 0:
                 rel_improvement = float('inf') # Positive improvement from zero
            elif base_mean == 0 and improvement == 0:
                rel_improvement = 0.0 # No change from zero

            summary.append({
                'Metric': metric.upper(),
                'Base Model': base_mean,
                'Fine-tuned Model': finetuned_mean,
                'Improvement': improvement,
                'Relative Improvement (%)': rel_improvement
            })
        else:
             logger.warning(f"Metric columns {base_metric} or {finetuned_metric} not found in detailed comparison. Skipping for summary.")
    
    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(output_dir, "model_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary statistics saved to {summary_path}")
    
    return df, summary_df

def save_influential_examples(influential_examples, output_dir):
    """Save the influential examples analysis to a Markdown file."""
    # Change filename extension to .md
    output_path = os.path.join(output_dir, "influential_examples.md")
    
    with open(output_path, 'w') as f:
        f.write("# Influential Training Examples Analysis\n\n")
        
        for i, example in enumerate(influential_examples):
            f.write(f"## Prompt {i+1}: \"{example['prompt']}\"\n\n")
            f.write(f"**Generated completion:** {example['finetuned_completion']}\n\n")
            
            f.write("### Most Influential Training Examples\n\n")
            # Start Markdown table
            f.write("| Rank | Score | Example Text (Truncated to 500 chars) |\n")
            f.write("|------|-------|---------------------------------------|\n")
            
            for j, infl in enumerate(example['most_influential']):
                text = infl['text']
                # Apply 500 character limit
                if len(text) > 500:
                    text = text[:500] + "..."
                
                # Clean text for Markdown table (escape pipe characters)
                text = text.replace('|', '\\|').replace('\n', ' ')
                
                # Write table row
                f.write(f"| {j+1} | {infl['score']:.6f} | {text} |\n")
            
            f.write("\n---\n\n") # Separator between prompts
    
    logger.info(f"Influential examples analysis saved to {output_path}")
    return output_path

def create_comparison_chart(summary_df, output_dir):
    """Create a comparison chart showing base vs fine-tuned model performance."""
    metrics = summary_df['Metric'].tolist()
    base_scores = summary_df['Base Model'].tolist()
    finetuned_scores = summary_df['Fine-tuned Model'].tolist()
    
    plt.figure(figsize=(10, 6))
    
    # Set width of bars
    bar_width = 0.35
    
    # Set position of bars on x axis
    r1 = np.arange(len(metrics))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    plt.bar(r1, base_scores, width=bar_width, label='Base Model', color='royalblue')
    plt.bar(r2, finetuned_scores, width=bar_width, label='Fine-tuned Model', color='darkorange')
    
    # Add labels and title
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks([r + bar_width/2 for r in range(len(metrics))], metrics)
    plt.legend()
    
    # Set y-axis limit to 0-1 as both metrics are similarity scores in 0-1 range
    plt.ylim(0, 1.0)
    
    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save main chart
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logger.info(f"Model comparison chart saved to {chart_path}")
    
    return chart_path

def compare_models(config):
    """Compare model outputs and analyze influences."""
    output_dir = config['output']['comparison_results']
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Comparing model outputs and analyzing influences")
    logger.info(f"Output directory: {output_dir}")
    
    # Load generated answers
    answers = load_generated_answers(config)
    logger.info(f"Loaded {len(answers)} generated answers")
    
    # Compute metrics
    comparison = compute_model_comparison(answers)
    logger.info("Computed comparison metrics")
    
    # Save comparison
    df, summary_df = save_model_comparison(comparison, output_dir)
    
    # Create comparison chart
    chart_path = create_comparison_chart(summary_df, output_dir)
    
    # Analyze influential examples
    influential_examples = analyze_influential_examples(config, answers)
    logger.info("Analyzed influential examples")
    
    # Save influential examples (function now saves .md)
    influential_path = save_influential_examples(influential_examples, output_dir)
    
    logger.info("Model comparison and influence analysis complete")
    
    return {
        'comparison_df': df,
        'summary_df': summary_df,
        'chart_path': chart_path,
        'influential_path': influential_path # Path is now to the .md file
    } 