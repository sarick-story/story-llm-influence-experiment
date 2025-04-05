# Small Language Model Influence Analysis

This project demonstrates how to train a TinyLlama model from scratch and then use [Kronfluence](https://github.com/amorthryn/kronfluence) to analyze the influence of training data on model outputs. It also includes a comprehensive evaluation framework that combines custom influence-based metrics with standardized benchmarks from DeepEval.

## Overview

The project consists of several components:

1. **Training**: Train a TinyLlama model on a subset of OpenWebText or patent data
2. **Influence Analysis**:
   - Compute influence factors for the trained model
   - Calculate influence scores for specific prompts
   - Visualize and analyze the most influential training examples
3. **Model Evaluation**:
   - Custom evaluation with BLEU, ROUGE metrics and influence analysis
   - Standardized benchmarks using DeepEval
   - Combined evaluation reporting

## Hardware Requirements

This project is best run on a machine with at least one NVIDIA A100 GPU (or equivalent) due to the memory requirements of training and analyzing the TinyLlama model.

## Setup

### Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### DeepEval Setup

DeepEval is automatically installed by the requirements.txt file, but if you want to install it manually, you can do:

```bash
pip install deepeval
```

## Using tmux for Long-Running Tasks

Since many of the tasks in this project can take hours to complete, it's recommended to use tmux to manage your sessions:

### Installing tmux

On Ubuntu/Debian:
```bash
sudo apt-get update && sudo apt-get install -y tmux
```

On MacOS:
```bash
brew install tmux
```

### Using tmux

1. Start a new tmux session:
```bash
tmux new -s llm_influence
```

2. Run your commands within the tmux session.

3. To detach from the session without stopping it:
Press `Ctrl+b` followed by `d`

4. To reattach to an existing session:
```bash
tmux attach-session -t llm_influence
```

### Monitoring with nvitop

To monitor GPU usage during your tasks, you can use nvitop. Run the following command in a separate terminal:

```bash
nvitop --colorful
```

## Centralized Configuration

All configuration parameters are now centralized in the `config.yaml` file. This includes:

- Model paths and names
- Dataset configurations
- Output directories
- Factor computation settings
- Evaluation parameters

You can modify this file to customize the behavior of the scripts.

## Running the Framework

### Using the Simple Wrapper

A simple wrapper script `run.py` is provided for common operations:

```bash
# Train the model from scratch
python run.py train

# Run the full analysis pipeline (train, factors, scores, inspection)
# Note: This does NOT use generated answers by default
python run.py analysis

# Compute influence factors for the trained model
python run.py factors

# Compute influence scores for the prompts
python run.py scores

# Run both custom and DeepEval evaluations
python run.py evaluate

# Run just the custom evaluation
python run.py custom-eval

# Run just the DeepEval evaluation
python run.py deepeval-eval

# Use a custom config file
python run.py --config custom.yaml train
```

### Using the Main Script Directly

For more flexibility, you can use the main script directly:

```bash
# Train the model
python main.py --config config.yaml train

# Compute influence factors
python main.py --config config.yaml compute_factors

# Inspect influence factors for a specific layer
python main.py --config config.yaml inspect_factors --layer 21

# Compute influence scores
python main.py --config config.yaml compute_scores

# Compute influence scores using generated answers
python main.py --config config.yaml compute_scores --use_generated

# Run evaluation
python main.py --config config.yaml evaluate --type all/custom/deepeval

# Run the full analysis pipeline
python main.py --config config.yaml run_full_analysis
```

## Understanding the Results

### Custom Evaluation Results

The custom evaluation focuses on comparing the base and fine-tuned models using standard NLP metrics and influence analysis:

- `comparison_results/model_comparison.csv`: Detailed comparison of both models
- `comparison_results/model_comparison_summary.csv`: Summary statistics
- `comparison_results/model_comparison_chart.png`: Visualization of model performance
- `comparison_results/influential_examples.txt`: Analysis of training examples that influenced the fine-tuned model's answers

### DeepEval Evaluation Results

The DeepEval evaluation provides standardized benchmark results:

- `deepeval_results/run_*/task_model_scores.json`: Raw benchmark scores
- Various logs and detailed task results in the run directory

### Combined Results

The combined evaluation provides a comprehensive view:

- `combined_evaluation_results/combined_evaluation_report.md`: Comprehensive report combining both evaluations
- `combined_evaluation_results/combined_improvements.png`: Visualization comparing improvements across both evaluation methods

## Project Structure

The project is organized into modules:

- `modules/training/`: Model training functionality
- `modules/analysis/`: Influence analysis (factors and scores)
- `modules/evaluation/`: Evaluation framework (custom, DeepEval, reporting)
- `main.py`: Central orchestrator for all operations
- `run.py`: Simple wrapper for common operations
- `config.yaml`: Centralized configuration

## Customization

- Modify `config.yaml` to customize the behavior of the scripts
- Add new prompts to `prompts.json` to evaluate different queries
- Add new tasks to the DeepEval configuration in `config.yaml`

## Implementation Details

### Custom Task Definition

The project uses custom task definitions in `task.py` to tell Kronfluence how to:
- Calculate losses for the language model
- Measure influence on model outputs
- Track specific modules within the TinyLlama model's architecture

The implementation focuses on the MLP layers of the model, which are typically the most influential for language generation tasks.

### Visualization Tools

The `inspect_factors.py` script provides visualization tools to analyze:
- The lambda matrices that encode influence relationships
- The distribution of eigenvalues that determine influence strength

These visualizations can help identify patterns in how the model learns from different examples.

## Customization

You can customize the analysis by:

- Modifying `prompts.json` to analyze different queries
- Adjusting configuration variables in `run_all_analysis.sh`
- Trying different factor strategies (`ekfac`, `kfac`, or `diagfisher`)
- Experimenting with different rank values for the query gradient approximation

## Requirements

The main requirements are:

- PyTorch
- Transformers
- Datasets
- Kronfluence
- Accelerate
- DeepEval
- nvitop (for monitoring GPU usage) 