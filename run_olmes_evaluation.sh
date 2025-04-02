#!/bin/bash

# Script to run OLMES evaluation on base and fine-tuned models
# This will be run from the story-llm-influence-experiment directory

# Stop on any error
set -e

# Configure paths and parameters
BASE_MODEL_NAME="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
FINETUNED_MODEL_PATH="./tinyllama_1b_model"  # Path to your fine-tuned model
BASE_MODEL_ID="base_tinyllama"
FINETUNED_MODEL_ID="finetuned_tinyllama"
OLMES_DIR="../olmes"  # Path to the OLMES directory
OLMES_OUTPUT_DIR="olmes_results"
COMBINED_RESULTS_DIR="combined_evaluation_results"

# Create output directories
mkdir -p $OLMES_OUTPUT_DIR
mkdir -p $COMBINED_RESULTS_DIR

# Create a configuration file for OLMES
echo "Creating OLMES configuration file..."
cat > olmes_config.yaml << EOL
tasks:
  # ARC tasks
  - task_name: arc_challenge
    description: AI2 Reasoning Challenge (25-shot)
    type: multiple_choice
    shots: 25
    data_source: data:ai2_arc/test?subset=Challenge

  # MMLU tasks
  - task_name: mmlu_humanities
    description: MMLU Humanities (5-shot)
    type: multiple_choice
    shots: 5
    data_source: data:cais_mmlu/test?subset=humanities
    
  - task_name: mmlu_stem
    description: MMLU STEM (5-shot)
    type: multiple_choice
    shots: 5
    data_source: data:cais_mmlu/test?subset=stem

  # GSM8K for math reasoning
  - task_name: gsm8k
    description: Grade School Math 8K (5-shot)
    type: generation
    shots: 5
    data_source: data:gsm8k/test
    metrics:
      - name: accuracy

  # TruthfulQA
  - task_name: truthful_qa
    description: TruthfulQA Multiple Choice (0-shot)
    type: multiple_choice
    shots: 0
    data_source: data:truthful_qa/generation

models:
  - model_id: "$BASE_MODEL_ID"
    name: "Base TinyLlama"
    source: "$BASE_MODEL_NAME"
    module: transformers
    
  - model_id: "$FINETUNED_MODEL_ID"
    name: "Fine-tuned TinyLlama"
    source: "$FINETUNED_MODEL_PATH"
    module: transformers

# Limit the number of examples for faster evaluation
# Remove this for a complete evaluation
limit_examples: 50
EOL

# Run OLMES evaluations
echo "Running OLMES evaluations..."
oe_eval run olmes_config.yaml --output-dir $OLMES_OUTPUT_DIR

# Create a script to combine OLMES results with our custom evaluation results
echo "Creating script to combine results..."
cat > combine_results.py << EOL
#!/usr/bin/env python3

import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Combine OLMES and custom evaluation results")
    parser.add_argument("--custom_results_dir", type=str, default="comparison_results", 
                        help="Directory with custom evaluation results")
    parser.add_argument("--olmes_results_dir", type=str, default="olmes_results", 
                        help="Directory with OLMES evaluation results")
    parser.add_argument("--output_dir", type=str, default="combined_evaluation_results", 
                        help="Output directory for combined results")
    return parser.parse_args()

def load_custom_results(custom_dir):
    custom_results = {}
    
    # Load model comparison summary
    summary_path = os.path.join(custom_dir, "model_comparison_summary.csv")
    if os.path.exists(summary_path):
        custom_results["summary"] = pd.read_csv(summary_path)
    
    # Load influential examples
    influential_path = os.path.join(custom_dir, "influential_examples.txt")
    if os.path.exists(influential_path):
        with open(influential_path, "r") as f:
            custom_results["influential_examples"] = f.read()
    
    return custom_results

def load_olmes_results(olmes_dir):
    olmes_results = {}
    
    run_dir = None
    # Find the most recent run directory
    for d in Path(olmes_dir).iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            if run_dir is None or d.stat().st_mtime > run_dir.stat().st_mtime:
                run_dir = d
    
    if run_dir is None:
        print(f"Warning: Could not find OLMES run directory in {olmes_dir}")
        return olmes_results
    
    # Load comparison results
    comparison_file = run_dir / "task_model_scores.json"
    if comparison_file.exists():
        with open(comparison_file, "r") as f:
            olmes_results["comparison"] = json.load(f)
    
    return olmes_results

def create_combined_report(custom_results, olmes_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Create combined markdown report
    report_path = os.path.join(output_dir, "combined_evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("# Combined Model Evaluation Report\n\n")
        
        # Custom evaluation section
        f.write("## Custom Influence-Based Evaluation\n\n")
        if "summary" in custom_results:
            f.write("### Model Comparison Summary\n\n")
            f.write(custom_results["summary"].to_markdown(index=False))
            f.write("\n\n")
        
        # OLMES evaluation section
        f.write("## OLMES Standardized Benchmarks\n\n")
        if "comparison" in olmes_results:
            f.write("### Benchmark Results\n\n")
            f.write("| Task | Base Model | Fine-tuned Model | Improvement |\n")
            f.write("|------|------------|------------------|-------------|\n")
            
            # Process OLMES results
            olmes_data = {}
            task_mapping = {}
            
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
                base_score = scores.get("base_tinyllama", "N/A")
                finetuned_score = scores.get("finetuned_tinyllama", "N/A")
                
                if isinstance(base_score, (int, float)) and isinstance(finetuned_score, (int, float)):
                    improvement = finetuned_score - base_score
                    f.write(f"| {display_name} | {base_score:.4f} | {finetuned_score:.4f} | {improvement:+.4f} |\n")
                else:
                    f.write(f"| {display_name} | {base_score} | {finetuned_score} | N/A |\n")
        
        # Influential examples section
        if "influential_examples" in custom_results:
            f.write("\n## Training Data Influence Analysis\n\n")
            f.write("For detailed analysis of influential training examples, please see the file: ")
            f.write("[influential_examples.txt](../comparison_results/influential_examples.txt)\n\n")
        
        # Conclusion
        f.write("\n## Conclusion\n\n")
        f.write("This report combines custom influence-based analysis with standardized benchmarks from OLMES. ")
        f.write("The combined evaluation provides a comprehensive view of the model's performance across ")
        f.write("different tasks and shows the impact of fine-tuning.\n")
    
    print(f"Created combined report at {report_path}")
    
    # Create visualization comparing both evaluation approaches
    if "summary" in custom_results and "comparison" in olmes_results:
        create_combined_visualization(custom_results, olmes_results, output_dir)

def create_combined_visualization(custom_results, olmes_results, output_dir):
    # Create a combined visualization showing performance improvements
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Custom metrics visualization
    custom_summary = custom_results["summary"]
    metrics = custom_summary["Metric"].tolist()
    improvements = custom_summary["Improvement"].tolist()
    
    ax1.bar(metrics, improvements, color="blue")
    ax1.set_title("Custom Evaluation Improvements")
    ax1.set_ylabel("Improvement")
    ax1.set_xticklabels(metrics, rotation=45, ha="right")
    ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # OLMES metrics visualization
    olmes_data = {}
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
        if "base_tinyllama" in scores and "finetuned_tinyllama" in scores:
            if isinstance(scores["base_tinyllama"], (int, float)) and isinstance(scores["finetuned_tinyllama"], (int, float)):
                olmes_tasks.append(task_name.replace("_", " ").title())
                olmes_improvements.append(scores["finetuned_tinyllama"] - scores["base_tinyllama"])
    
    ax2.bar(olmes_tasks, olmes_improvements, color="green")
    ax2.set_title("OLMES Benchmark Improvements")
    ax2.set_ylabel("Score Improvement")
    ax2.set_xticklabels(olmes_tasks, rotation=45, ha="right")
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, "combined_improvements.png")
    plt.savefig(viz_path)
    print(f"Created visualization at {viz_path}")

def main():
    args = parse_args()
    
    print("Loading custom evaluation results...")
    custom_results = load_custom_results(args.custom_results_dir)
    
    print("Loading OLMES evaluation results...")
    olmes_results = load_olmes_results(args.olmes_results_dir)
    
    print("Creating combined report and visualizations...")
    create_combined_report(custom_results, olmes_results, args.output_dir)
    
    print("Completed combined analysis!")

if __name__ == "__main__":
    main()
EOL

# Run the script to combine results
echo "Combining evaluation results..."
python combine_results.py \
  --custom_results_dir "comparison_results" \
  --olmes_results_dir "$OLMES_OUTPUT_DIR" \
  --output_dir "$COMBINED_RESULTS_DIR"

echo "OLMES evaluation complete!"
echo "Results saved to:"
echo "  - Raw OLMES results: $OLMES_OUTPUT_DIR"
echo "  - Combined evaluation report: $COMBINED_RESULTS_DIR/combined_evaluation_report.md"
echo "  - Combined visualization: $COMBINED_RESULTS_DIR/combined_improvements.png" 