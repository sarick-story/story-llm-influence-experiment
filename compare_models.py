import argparse
import json
import logging
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tabulate import tabulate
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import numpy as np
from inspect_scores import load_scores, print_score_summary, print_extreme_examples

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to download nltk data (if not already present)
try:
    nltk.download('punkt', quiet=True)
except:
    logger.warning("Could not download NLTK data. BLEU scores may not work properly.")

def parse_args():
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned model outputs and analyze influences.")
    
    parser.add_argument(
        "--generated_answers_file",
        type=str,
        default="generated_answers.json",
        help="File with generated answers from both models."
    )
    parser.add_argument(
        "--scores_name",
        type=str,
        default="olmoe_prompt_scores",
        help="Name of computed influence scores."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Elriggs/openwebtext-100k",
        help="Dataset name to use for training examples."
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of most influential examples to display."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_results",
        help="Directory to save comparison results."
    )
    
    return parser.parse_args()

def calculate_metrics(reference, hypothesis):
    """Calculate BLEU and ROUGE scores between reference and hypothesis."""
    # Tokenize for BLEU
    reference_tokens = word_tokenize(reference.lower())
    hypothesis_tokens = word_tokenize(hypothesis.lower())
    
    # Calculate BLEU score
    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens)
    
    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, hypothesis)
    
    return {
        'bleu': bleu_score,
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure
    }

def compare_model_outputs(generated_answers_file, output_dir):
    """Compare outputs from base and fine-tuned models."""
    # Load generated answers
    with open(generated_answers_file, 'r') as f:
        answers_data = json.load(f)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics for each prompt
    results = []
    for i, item in enumerate(answers_data):
        prompt = item["prompt"]
        original = item["original_completion"]
        base = item["base_model_completion"]
        finetuned = item["finetuned_model_completion"]
        
        # Calculate metrics for base model vs original
        base_metrics = calculate_metrics(original, base)
        
        # Calculate metrics for fine-tuned model vs original
        finetuned_metrics = calculate_metrics(original, finetuned)
        
        # Calculate metrics between base and fine-tuned
        base_vs_finetuned = calculate_metrics(base, finetuned)
        
        result = {
            "prompt_id": i,
            "prompt": prompt,
            "original_completion": original,
            "base_completion": base,
            "finetuned_completion": finetuned,
            "base_bleu": base_metrics["bleu"],
            "base_rouge1": base_metrics["rouge1"],
            "base_rouge2": base_metrics["rouge2"],
            "base_rougeL": base_metrics["rougeL"],
            "finetuned_bleu": finetuned_metrics["bleu"],
            "finetuned_rouge1": finetuned_metrics["rouge1"],
            "finetuned_rouge2": finetuned_metrics["rouge2"],
            "finetuned_rougeL": finetuned_metrics["rougeL"],
            "improvement_bleu": finetuned_metrics["bleu"] - base_metrics["bleu"],
            "improvement_rouge1": finetuned_metrics["rouge1"] - base_metrics["rouge1"],
            "improvement_rouge2": finetuned_metrics["rouge2"] - base_metrics["rouge2"],
            "improvement_rougeL": finetuned_metrics["rougeL"] - base_metrics["rougeL"],
        }
        results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save full results to CSV
    csv_file = os.path.join(output_dir, "model_comparison.csv")
    df.to_csv(csv_file, index=False)
    logger.info(f"Full comparison results saved to {csv_file}")
    
    # Create summary statistics
    summary = {
        "metric": ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"],
        "base_mean": [
            df["base_bleu"].mean(),
            df["base_rouge1"].mean(),
            df["base_rouge2"].mean(),
            df["base_rougeL"].mean()
        ],
        "finetuned_mean": [
            df["finetuned_bleu"].mean(),
            df["finetuned_rouge1"].mean(),
            df["finetuned_rouge2"].mean(),
            df["finetuned_rougeL"].mean()
        ],
        "improvement": [
            df["improvement_bleu"].mean(),
            df["improvement_rouge1"].mean(),
            df["improvement_rouge2"].mean(),
            df["improvement_rougeL"].mean()
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    
    # Save summary to CSV
    summary_csv = os.path.join(output_dir, "model_comparison_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Summary comparison results saved to {summary_csv}")
    
    # Print summary table
    print("\nModel Comparison Summary:")
    print(tabulate(summary_df, headers="keys", tablefmt="grid", floatfmt=".4f"))
    
    # Create visualization of improvements
    plt.figure(figsize=(10, 6))
    
    # Bar chart of average scores
    x = np.arange(4)
    width = 0.35
    
    plt.bar(x - width/2, summary_df["base_mean"], width, label="Base Model")
    plt.bar(x + width/2, summary_df["finetuned_mean"], width, label="Fine-tuned Model")
    
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Comparison of Base vs. Fine-tuned Model Performance")
    plt.xticks(x, summary_df["metric"])
    plt.legend()
    
    # Save chart
    chart_file = os.path.join(output_dir, "model_comparison_chart.png")
    plt.savefig(chart_file)
    logger.info(f"Comparison chart saved to {chart_file}")
    
    return df, summary_df

def analyze_model_influences(scores_name, dataset_name, df, num_examples, output_dir):
    """Analyze influences for the fine-tuned model's answers."""
    # Load scores
    scores = load_scores(scores_name)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get prompt IDs with the biggest improvements
    top_improved = df.sort_values("improvement_rougeL", ascending=False).head(10)
    
    # Print and save influence results for top improved answers
    influences_file = os.path.join(output_dir, "influential_examples.txt")
    with open(influences_file, 'w') as f:
        for _, row in top_improved.iterrows():
            prompt_id = row["prompt_id"]
            prompt = row["prompt"]
            base_completion = row["base_completion"]
            finetuned_completion = row["finetuned_completion"]
            improvement = row["improvement_rougeL"]
            
            header = f"\n\nPrompt {prompt_id} (Rouge-L improvement: {improvement:.4f})"
            f.write(header + "\n")
            f.write("=" * len(header) + "\n")
            
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Base model output: {base_completion}\n")
            f.write(f"Fine-tuned model output: {finetuned_completion}\n\n")
            
            f.write("Most influential positive examples:\n")
            f.write("---------------------------------\n")
            pos_examples = print_extreme_examples(
                scores, dataset_name, query_idx=prompt_id, 
                k=num_examples, mode="pos", output_file=f
            )
            
            f.write("\nMost influential negative examples:\n")
            f.write("---------------------------------\n")
            neg_examples = print_extreme_examples(
                scores, dataset_name, query_idx=prompt_id, 
                k=num_examples, mode="neg", output_file=f
            )
            
    logger.info(f"Influence analysis saved to {influences_file}")

def main():
    args = parse_args()
    
    # Compare model outputs
    df, summary_df = compare_model_outputs(args.generated_answers_file, args.output_dir)
    
    # Analyze influences for the fine-tuned model
    analyze_model_influences(
        args.scores_name, args.dataset_name, df, args.num_examples, args.output_dir
    )
    
    logger.info("Comparison and influence analysis complete.")
    logger.info(f"All results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 