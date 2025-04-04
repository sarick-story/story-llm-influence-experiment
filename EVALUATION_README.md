# Model Evaluation Framework

This directory contains scripts for evaluating language models using both custom, influence-based metrics and standardized benchmarks from OLMES.

## Evaluation Scripts

There are three main evaluation scripts:

1. `run_comparison_analysis.sh` - Runs the custom influence-based evaluation
2. `run_olmes_evaluation.sh` - Runs the OLMES standardized benchmarks
3. `run_comprehensive_evaluation.sh` - Runs both evaluations and combines the results

## Prerequisites

Before running the evaluation scripts, ensure you have:

1. Trained your fine-tuned model (using `run_all_analysis.sh`)
2. Computed influence factors for the model
3. Installed OLMES (will be installed automatically if not present)

## How to Use

### Option 1: Run the Comprehensive Evaluation

For a complete evaluation using both custom metrics and OLMES benchmarks, run:

```bash
cd story-llm-influence-experiment
./run_comprehensive_evaluation.sh
```

This will:
1. Run the custom influence-based evaluation
2. Run the OLMES standardized benchmarks
3. Combine the results into a comprehensive report

### Option 2: Run Individual Evaluations

If you want to run just one evaluation method:

```bash
# For custom influence-based evaluation only
./run_comparison_analysis.sh

# For OLMES standardized benchmarks only
./run_olmes_evaluation.sh
```

## Outputs

The evaluation framework generates the following outputs:

### Custom Evaluation (in `comparison_results/`)
- `model_comparison.csv`: Detailed comparison of model performance
- `model_comparison_summary.csv`: Summary statistics
- `model_comparison_chart.png`: Visualization of model performance
- `influential_examples.txt`: Analysis of training examples that influenced the fine-tuned model

### OLMES Benchmarks (in `olmes_results/`)
- Standard benchmark results (ARC, MMLU, GSM8K, TruthfulQA)
- Raw evaluation data and scores

### Combined Analysis (in `combined_evaluation_results/`)
- `combined_evaluation_report.md`: Comprehensive report combining both evaluations
- `combined_improvements.png`: Visualization comparing improvements across both evaluation methods

## Customizing the Evaluation

### Custom Evaluation
To customize the custom evaluation, modify:
- `prompts.json`: Change the prompts used for evaluation

### OLMES Benchmarks
To customize the OLMES evaluation, modify:
- `olmes_config.yaml`: Add or remove benchmark tasks
- Adjust the `limit_examples` parameter for faster/more comprehensive evaluation

## Technical Notes

- The OLMES evaluation uses a limited number of examples (50 by default) for faster evaluation. Remove or increase the `limit_examples` parameter in `olmes_config.yaml` for a more thorough evaluation.
- Both evaluations use the same base and fine-tuned models for consistency. 