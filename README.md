# Small Language Model Influence Analysis

This project demonstrates how to train a small language model from scratch and then use [Kronfluence](https://github.com/amorthryn/kronfluence) to analyze the influence of training data on model outputs.

## Overview

The project consists of several steps:

1. Train a TinyLlama-1b model on a subset of OpenWebText
2. Compute influence factors for the trained model 
3. Calculate influence scores for specific prompts
4. Visualize and analyze the most influential training examples
5. Preview the dataset

## Hardware Requirements

This project is best run on a machine with at least one NVIDIA A100 GPU (or equivalent) due to the memory requirements of training and analyzing the TinyLlama-1b model.

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
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

5. To list all sessions:
```bash
tmux ls
```
### Monitoring with nvitop

To monitor GPU usage during your tasks, you can use nvitop. Run the following command in a separate terminal:

```bash
nvitop --colorful
```

## Running the Complete Analysis

The entire analysis pipeline can be run using a single script:

```bash
bash run_all_analysis.sh
```

This script will:
1. Train the TinyLlama-1b model
2. Compute influence factors
3. Visualize factor distributions
4. Calculate influence scores for provided prompts
5. Generate an analysis report

You can monitor the progress in real-time, and all outputs are logged to files for later review.

## Step-by-Step Process

The following sections detail what happens in each step of the analysis pipeline.

### Step 1: Train the Model

The script first trains the TinyLlama-1b model on a subset of OpenWebText. This is done by running `train.py`, which trains the model on examples from OpenWebText and saves it to the configured model path.

### Step 2: Compute Influence Factors

Next, the script computes the EKFAC influence factors for the trained model. This involves analyzing how different parts of the model contribute to its predictions.

### Step 3: Visualize Influence Factors

The script generates visualizations of the computed influence factors to help understand their distribution. This produces heatmaps and eigenvalue plots for selected layers' MLP modules.

### Step 4: Compute Influence Scores

The script then calculates how much each training example influenced the model's responses to the prompts defined in `prompts.json`.

### Step 5: Analyze the Results

Finally, the script generates a report showing the most influential training examples for each of your prompts, saved in Markdown format.

## Understanding the Results

For each prompt, the analysis will show:

- The top positive influences (training examples that helped the model produce the given response)
- The top negative influences (training examples that conflicted with the given response)

The influence score indicates the strength of the connection between a training example and the model's output for a given prompt.

## Implementation Details

### Custom Task Definition

The project uses custom task definitions in `task.py` to tell Kronfluence how to:
- Calculate losses for the language model
- Measure influence on model outputs
- Track specific modules within the TinyLlama-1b model's architecture

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
- nvitop (for monitoring GPU usage) 