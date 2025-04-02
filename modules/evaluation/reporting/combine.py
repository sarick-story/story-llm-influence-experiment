"""
Combined Evaluation Results Module

This module combines results from custom and OLMES evaluations into a single comprehensive report.
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
    
    # Load influential examples
    influential_path = os.path.join(custom_dir, "influential_examples.txt")
    if os.path.exists(influential_path):
        logger.info(f"Loading influential examples from {influential_path}")
        with open(influential_path, "r") as f:
            results["influential_examples"] = f.read()
    else:
        logger.warning(f"Influential examples file not found: {influential_path}")
    
    return results

def load_olmes_results(config):
    """Load results from the OLMES evaluation."""
    olmes_dir = config['output']['olmes_results']
    results = {}
    
    # Find the most recent run directory
    run_dir = None
    run_dirs = [d for d in Path(olmes_dir).iterdir() if d.is_dir() and d.name.startswith('run_')]
    
    if not run_dirs:
        logger.error(f"No OLMES run directories found in {olmes_dir}")
        return results
    
    # Sort by modification time, newest first
    run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
    logger.info(f"Found OLMES run directory: {run_dir}")
    
    # Load comparison results
    comparison_file = run_dir / "task_model_scores.json"
    if comparison_file.exists():
        logger.info(f"Loading OLMES results from {comparison_file}")
        with open(comparison_file, "r") as f:
            results["comparison"] = json.load(f)
    else:
        logger.warning(f"OLMES comparison file not found: {comparison_file}")
    
    return results

def create_combined_report(custom_results, olmes_results, output_dir):
    """Create a combined evaluation report."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "combined_evaluation_report.md")
    
    logger.info(f"Creating combined report at {report_path}")
    
    with open(report_path, "w") as f:
        f.write("# Combined Model Evaluation Report\n\n")
        
        # Part 1: Custom Evaluation Results
        f.write("## Part 1: Custom Influence-Based Evaluation\n\n")
        if "summary" in custom_results:
            f.write("### Model Comparison Summary\n\n")
            f.write(custom_results["summary"].to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write("*No custom evaluation results found.*\n\n")
        
        # Part 2: OLMES Results
        f.write("## Part 2: OLMES Standardized Benchmarks\n\n")
        if "comparison" in olmes_results:
            f.write("### Benchmark Results\n\n")
            f.write("| Task | Base Model | Fine-tuned Model | Improvement |\n")
            f.write("|------|------------|-----------------|-------------|\n")
            
            # Process OLMES results
            olmes_data = {}
            task_mapping = {}
            base_model_id = "base_tinyllama"
            finetuned_model_id = "finetuned_tinyllama"
            
            for entry in olmes_results["comparison"]:
                task_name = entry["task_name"]
                model_id = entry["model_id"]
                score = entry["score"]
                
                if task_name not in olmes_data:
                    olmes_data[task_name] = {}
                    # Create a more readable task name for display
                    task_mapping[task_name] = task_name.replace("_", " ").title()
                
                olmes_data[task_name][model_id] = score
            
            # Write tasks to markdown table
            for task_name, scores in olmes_data.items():
                display_name = task_mapping.get(task_name, task_name)
                base_score = scores.get(base_model_id, "N/A")
                finetuned_score = scores.get(finetuned_model_id, "N/A")
                
                if isinstance(base_score, (int, float)) and isinstance(finetuned_score, (int, float)):
                    improvement = finetuned_score - base_score
                    f.write(f"| {display_name} | {base_score:.4f} | {finetuned_score:.4f} | {improvement:+.4f} |\n")
                else:
                    f.write(f"| {display_name} | {base_score} | {finetuned_score} | N/A |\n")
        else:
            f.write("*No OLMES evaluation results found.*\n\n")
        
        # Part 3: Influential Examples Analysis
        if "influential_examples" in custom_results:
            f.write("\n## Part 3: Training Data Influence Analysis\n\n")
            f.write("For detailed analysis of influential training examples, please see: ")
            f.write(f"[Influential Examples Analysis](../comparison_results/influential_examples.txt)\n\n")
            
            # Include a sample of the analysis
            sample_lines = custom_results["influential_examples"].split('\n')[:20]
            f.write("### Sample of Influence Analysis\n\n")
            f.write("```\n")
            f.write('\n'.join(sample_lines))
            f.write("\n...\n```\n\n")
        
        # Conclusion
        f.write("\n## Conclusion\n\n")
        f.write("This report combines custom influence-based analysis with standardized benchmarks from OLMES. ")
        f.write("The combined evaluation provides a comprehensive view of the model's performance across ")
        f.write("different tasks and shows the impact of fine-tuning.\n\n")
        
        f.write("Key findings:\n\n")
        
        # Add some key findings based on the results
        if "summary" in custom_results:
            best_metric = custom_results["summary"].loc[custom_results["summary"]["Improvement"].idxmax()]
            f.write(f"- The fine-tuned model showed the greatest improvement in **{best_metric['Metric']}** ")
            f.write(f"with an absolute improvement of {best_metric['Improvement']:.4f} ")
            f.write(f"({best_metric['Relative Improvement (%)']}%).\n")
        
        if "comparison" in olmes_results:
            # Find the task with the greatest improvement
            best_task = None
            best_improvement = -float('inf')
            
            for task_name, scores in olmes_data.items():
                if base_model_id in scores and finetuned_model_id in scores:
                    if isinstance(scores[base_model_id], (int, float)) and isinstance(scores[finetuned_model_id], (int, float)):
                        improvement = scores[finetuned_model_id] - scores[base_model_id]
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_task = task_name
            
            if best_task:
                f.write(f"- In the OLMES benchmarks, the fine-tuned model performed best on the **{task_mapping.get(best_task, best_task)}** task ")
                f.write(f"with an improvement of {best_improvement:+.4f}.\n")
        
        f.write("- The influence analysis shows that training examples similar to the evaluation prompts ")
        f.write("had the most significant impact on the fine-tuned model's performance.\n")
    
    logger.info(f"Combined report created: {report_path}")
    return report_path

def create_combined_visualization(custom_results, olmes_results, output_dir):
    """Create visualizations comparing both evaluation approaches."""
    if "summary" not in custom_results or "comparison" not in olmes_results:
        logger.warning("Missing data for combined visualization")
        return None
    
    # Extract custom metrics
    custom_summary = custom_results["summary"]
    custom_metrics = custom_summary["Metric"].tolist()
    custom_improvements = custom_summary["Improvement"].tolist()
    
    # Extract OLMES metrics
    olmes_data = {}
    base_model_id = "base_tinyllama"
    finetuned_model_id = "finetuned_tinyllama"
    
    for entry in olmes_results["comparison"]:
        task_name = entry["task_name"]
        model_id = entry["model_id"]
        score = entry["score"]
        
        if task_name not in olmes_data:
            olmes_data[task_name] = {}
        
        olmes_data[task_name][model_id] = score
    
    olmes_tasks = []
    olmes_improvements = []
    
    for task_name, scores in olmes_data.items():
        if base_model_id in scores and finetuned_model_id in scores:
            if isinstance(scores[base_model_id], (int, float)) and isinstance(scores[finetuned_model_id], (int, float)):
                olmes_tasks.append(task_name.replace("_", " ").title())
                olmes_improvements.append(scores[finetuned_model_id] - scores[base_model_id])
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Custom metrics visualization
    ax1.bar(custom_metrics, custom_improvements, color="blue")
    ax1.set_title("Custom Evaluation Improvements")
    ax1.set_ylabel("Improvement")
    ax1.set_xticklabels(custom_metrics, rotation=45, ha="right")
    ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # OLMES metrics visualization
    ax2.bar(olmes_tasks, olmes_improvements, color="green")
    ax2.set_title("OLMES Benchmark Improvements")
    ax2.set_ylabel("Score Improvement")
    ax2.set_xticklabels(olmes_tasks, rotation=45, ha="right")
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, "combined_improvements.png")
    plt.savefig(viz_path, dpi=300)
    plt.close()
    
    logger.info(f"Created combined visualization: {viz_path}")
    return viz_path

def combine_evaluation_results(config):
    """Combine results from both custom and OLMES evaluations."""
    custom_dir = config['output']['comparison_results']
    olmes_dir = config['output']['olmes_results']
    output_dir = config['output']['combined_results']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Combining evaluation results")
    logger.info(f"Custom results: {custom_dir}")
    logger.info(f"OLMES results: {olmes_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load custom results
    custom_results = load_custom_results(config)
    
    # Load OLMES results
    olmes_results = load_olmes_results(config)
    
    # Create combined report
    report_path = create_combined_report(custom_results, olmes_results, output_dir)
    
    # Create combined visualization
    viz_path = create_combined_visualization(custom_results, olmes_results, output_dir)
    
    logger.info("Combined evaluation complete")
    
    return {
        "report_path": report_path,
        "viz_path": viz_path
    } 