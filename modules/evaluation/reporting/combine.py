"""
Combined Evaluation Results Module

This module combines results from custom and standardized benchmark evaluations into a single comprehensive report.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_custom_results(config):
    """Load results from the custom evaluation."""
    custom_dir = config['output']['comparison_results']
    results = {}
    
    # Load model comparison summary
    summary_path = os.path.join(custom_dir, "model_comparison_summary.csv")
    if os.path.exists(summary_path):
        logger.info(f"Loading custom summary from {summary_path}")
        results["summary"] = pd.read_csv(summary_path)
    else:
        logger.warning(f"Custom summary file not found: {summary_path}")
    
    # Load influential examples (now from .md file)
    influential_path = os.path.join(custom_dir, "influential_examples.md")
    if os.path.exists(influential_path):
        logger.info(f"Loading influential examples from {influential_path}")
        with open(influential_path, "r") as f:
            results["influential_examples"] = f.read()
    else:
        logger.warning(f"Influential examples file not found: {influential_path}")
    
    return results

def load_benchmark_results(config):
    """Load results from the configured benchmark evaluation (e.g., DeepEval)."""
    # Use the generic benchmark results path from config
    benchmark_dir_path = config['output'].get('deepeval_results') # Use get for backward compatibility
    if not benchmark_dir_path:
        logger.error("'deepeval_results' path not found in config output section.")
        return {}
        
    benchmark_dir = Path(benchmark_dir_path)
    results = {}
    
    # Load the benchmark summary file
    summary_file = benchmark_dir / "benchmark_summary.json"
    if summary_file.exists():
        logger.info(f"Loading benchmark summary from {summary_file}")
        try:
            with open(summary_file, "r") as f:
                results["summary"] = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {summary_file}: {e}")
            # Try to provide more context about the file content
            try:
                with open(summary_file, "r") as f_err:
                    content_preview = f_err.read(500) # Read first 500 chars
                    logger.error(f"File content preview:\n{content_preview}")
            except Exception as read_err:
                logger.error(f"Could not read file content for preview: {read_err}")
            results["summary"] = {} # Set to empty dict on error
            
        # Optionally load individual benchmark results if needed for more detail
        # For MMLU specifically:
        mmlu_dir = benchmark_dir / "mmlu"
        if mmlu_dir.exists():
            task_scores_file = mmlu_dir / "finetuned_model_task_scores.csv"
            if task_scores_file.exists():
                try:
                    results["mmlu_task_scores"] = pd.read_csv(task_scores_file)
                except pd.errors.EmptyDataError:
                    logger.warning(f"DeepEval MMLU task scores file is empty: {task_scores_file}")
                    results["mmlu_task_scores"] = pd.DataFrame() # Use empty DataFrame
                except Exception as e:
                     logger.error(f"Error reading MMLU task scores CSV: {e}")
            else:
                 logger.warning(f"DeepEval MMLU task scores file not found: {task_scores_file}")
                 # Look for JSON as fallback
                 task_scores_json_file = mmlu_dir / "finetuned_model_task_scores.json"
                 if task_scores_json_file.exists():
                     try:
                         with open(task_scores_json_file, 'r') as f_json:
                            results["mmlu_task_scores"] = pd.DataFrame(json.load(f_json))
                     except Exception as e:
                         logger.error(f"Error loading MMLU task scores from JSON: {e}")
    else:
        logger.error(f"Benchmark summary file not found: {summary_file}")
        
    return results

def create_combined_report(config, custom_results, benchmark_results, output_dir):
    """Create a combined evaluation report."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "combined_evaluation_report.md")
    
    logger.info(f"Creating combined report at {report_path}")
    
    with open(report_path, "w") as f:
        f.write("# Combined Model Evaluation Report\n\n")
        
        # Part 1: Custom Evaluation Results
        f.write("## Part 1: Custom Influence-Based Evaluation\n\n")
        if "summary" in custom_results and not custom_results["summary"].empty:
            f.write("### Model Comparison Summary\n\n")
            f.write(custom_results["summary"].to_markdown(index=False))
            f.write("\n\n")

            # Add explanations for BERTScore and semantic similarity
            f.write("\n> **Metric Explanations:**\n")
            f.write("> - **BERT_SCORE**: Measures semantic similarity using contextual embeddings. Higher values indicate better alignment of meaning between model outputs and reference texts.\n")
            f.write("> - **SEMANTIC_SIM**: Measures cosine similarity between sentence embeddings of model outputs and reference texts, capturing overall semantic similarity.\n")
            f.write("> Both metrics range from 0 to 1, with higher values indicating better performance.\n\n")
        else:
            f.write("*No custom evaluation comparison summary found.*\n\n")
        
        # Part 2: Standardized Benchmark Results (e.g., DeepEval)
        f.write("## Part 2: Standardized Benchmarks\n\n")
        if "summary" in benchmark_results and benchmark_results["summary"]:
            # Determine which models were evaluated based on summary keys
            evaluated_models = [k for k in benchmark_results["summary"] if k.endswith('_model')]
            
            if not evaluated_models:
                 f.write("*Benchmark summary found, but no model results detected.*\n")
            
            for model_key in evaluated_models:
                model_results = benchmark_results["summary"][model_key]
                model_name = model_results.get('name', model_key) # Use model ID if available
                f.write(f"### Benchmark Results for: {model_name}\n\n")
                
                # Display overall scores from summary
                if "benchmarks" in model_results and model_results["benchmarks"]:
                    f.write("| Benchmark | Overall Score |\n")
                    f.write("|-----------|---------------|\n")
                    for bench_name, score in model_results["benchmarks"].items():
                        f.write(f"| {bench_name.upper()} | {score:.4f} |\n")
                    f.write("\n")
                else:
                     f.write("*No overall benchmark scores found in summary for this model.*\n")

                # Display MMLU task scores if available (assuming MMLU run for this model)
                # Note: This assumes file names like 'finetuned_model_task_scores.csv'
                # If base model was also evaluated, this logic might need adjustment
                if model_key == "finetuned_model" and "mmlu_task_scores" in benchmark_results and not benchmark_results["mmlu_task_scores"].empty:
                    f.write("#### MMLU Task-Specific Scores\n\n")
                    mmlu_scores_df = benchmark_results["mmlu_task_scores"].copy()
                    if "task" in mmlu_scores_df.columns and "accuracy" in mmlu_scores_df.columns:
                         mmlu_scores_df = mmlu_scores_df[["task", "accuracy"]]
                         mmlu_scores_df.columns = ["MMLU Task", "Accuracy"]
                         f.write(mmlu_scores_df.to_markdown(index=False))
                         f.write("\n\n")
                    else:
                         f.write("*MMLU task scores table found but columns mismatch. Displaying raw table:*\n\n")
                         f.write(mmlu_scores_df.to_markdown(index=False))
                         f.write("\n\n")
                elif model_key == "finetuned_model": # Only mention if expected for fine-tuned
                    f.write("*No detailed MMLU task scores found.*\n\n")
        else:
            f.write("*No standardized benchmark results found.*\n\n")
        
        # Part 3: Influential Examples Analysis
        if "influential_examples" in custom_results:
            f.write("\n## Part 3: Training Data Influence Analysis\n\n")
            # Construct relative path if possible
            try:
                relative_influential_path = os.path.relpath(
                    os.path.join(config['output']['comparison_results'], "influential_examples.md"), 
                    output_dir
                )
                f.write(f"For detailed analysis, see: [{os.path.basename(relative_influential_path)}]({relative_influential_path})\n\n")
            except ValueError:
                 # Fallback if paths are on different drives
                 full_influential_path = os.path.abspath(os.path.join(config['output']['comparison_results'], "influential_examples.md"))
                 f.write(f"For detailed analysis, see the influential examples report at: `{full_influential_path}`\n\n")

            # Include a sample of the analysis
            sample_lines = custom_results["influential_examples"].split('\n')[:30]
            f.write("### Sample of Influence Analysis\n\n")
            f.write("```markdown\n")
            f.write('\n'.join(sample_lines))
            f.write("\n...\n```\n\n")
        
        # Conclusion
        f.write("\n## Conclusion\n\n")
        f.write("This report combines custom influence-based analysis with standardized benchmarks (e.g., DeepEval). ")
        f.write("The combined evaluation provides a view of the model's performance and the impact of fine-tuning.\n\n")
        
        f.write("Key findings:\n\n")
        
        # Add some key findings based on the results
        best_custom_metric_found = False
        if "summary" in custom_results and not custom_results["summary"].empty:
            try:
                 best_metric_row = custom_results["summary"].loc[custom_results["summary"]["Improvement"].astype(float).idxmax()]
                 f.write(f"- Custom Comparison: The fine-tuned model showed the greatest improvement in **{best_metric_row['Metric']}** ")
                 f.write(f"with an absolute improvement of {best_metric_row['Improvement']:.4f} ")
                 f.write(f"({best_metric_row['Relative Improvement (%)']:.2f}%).\n")
                 best_custom_metric_found = True
            except (KeyError, ValueError, TypeError) as e:
                 logger.warning(f"Could not determine best custom metric due to error: {e}")

        if not best_custom_metric_found:
             f.write("- No summary data available to determine best custom metric.\n")

        best_benchmark_score_found = False
        if "summary" in benchmark_results and benchmark_results["summary"]:
             # Report score for the fine-tuned model if available
             if "finetuned_model" in benchmark_results["summary"]:
                 ft_benchmarks = benchmark_results["summary"]["finetuned_model"].get("benchmarks", {})
                 if ft_benchmarks:
                     # Simple reporting of the first benchmark score
                     first_bench = list(ft_benchmarks.keys())[0]
                     first_score = ft_benchmarks[first_bench]
                     f.write(f"- Benchmarks: The fine-tuned model achieved an overall score of **{first_score:.4f}** on the {first_bench.upper()} benchmark.\n")
                     best_benchmark_score_found = True
        
        if not best_benchmark_score_found:
            f.write("- No benchmark scores available for the fine-tuned model.\n")
        
        if "influential_examples" in custom_results:
            f.write("- The influence analysis identifies training examples with significant impact on specific prompt outputs.\n")
        else:
             f.write("- Influence analysis results were not found.\n")
    
    logger.info(f"Combined report created: {report_path}")
    return report_path

def create_combined_visualization(custom_results, benchmark_results, output_dir):
    """Create visualizations comparing both evaluation approaches."""
    # Skip if no data
    if "summary" not in custom_results or custom_results["summary"].empty:
        logger.warning("Missing custom summary data for combined visualization")
        return None
    
    # Create raw scores comparison (base vs fine-tuned)
    custom_summary = custom_results["summary"]
    metrics = custom_summary["Metric"].tolist()
    base_scores = custom_summary["Base Model"].tolist()
    finetuned_scores = custom_summary["Fine-tuned Model"].tolist()
    
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
    plt.ylabel('Score (0-1 range)')
    plt.title('Model Performance Comparison')
    plt.xticks([r + bar_width/2 for r in range(len(metrics))], metrics)
    plt.legend()
    
    # Set y-axis limit 
    plt.ylim(0, 1.0)
    
    # Add grid for readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    scores_viz_path = os.path.join(output_dir, "model_scores_comparison.png")
    plt.savefig(scores_viz_path, dpi=300)
    plt.close()
    logger.info(f"Created model scores comparison visualization: {scores_viz_path}")
    
    # Create improvement visualization
    try:
        custom_improvements = custom_summary["Improvement"].astype(float).tolist()
        
        plt.figure(figsize=(8, 6))
        colors = ['green' if x > 0 else 'red' for x in custom_improvements]
        plt.bar(metrics, custom_improvements, color=colors)
        plt.title("Fine-tuned vs. Base Model Improvements")
        plt.ylabel("Absolute Score Improvement")
        plt.xticks(rotation=45, ha="right")
        plt.axhline(y=0, color='grey', linestyle='--', linewidth=0.7)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        viz_path = os.path.join(output_dir, "custom_improvements.png")
        plt.savefig(viz_path, dpi=300)
        plt.close()
        
        logger.info(f"Created custom improvements visualization: {viz_path}")
        return {"scores": scores_viz_path, "improvements": viz_path}
    except (KeyError, ValueError) as e:
        logger.warning(f"Could not parse custom improvements for visualization: {e}")
        return {"scores": scores_viz_path}

def combine_evaluation_results(config):
    """Combine results from both custom and benchmark evaluations."""
    custom_dir = config['output']['comparison_results']
    # Use the correct config key for benchmark results directory
    benchmark_dir = config['output'].get('deepeval_results', config['output'].get('olmes_results')) # Fallback just in case
    output_dir = config['output']['combined_results']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Combining evaluation results")
    logger.info(f"Custom results source: {custom_dir}")
    logger.info(f"Benchmark results source: {benchmark_dir}")
    logger.info(f"Combined output directory: {output_dir}")
    
    # Load custom results
    custom_results = load_custom_results(config)
    
    # Load benchmark results
    benchmark_results = load_benchmark_results(config)
    
    # Create combined report
    report_path = create_combined_report(config, custom_results, benchmark_results, output_dir)
    
    # Create combined visualizations
    viz_paths = create_combined_visualization(custom_results, benchmark_results, output_dir)
    
    logger.info("Combined evaluation reporting complete")
    
    return {
        "report_path": report_path,
        "viz_paths": viz_paths
    } 