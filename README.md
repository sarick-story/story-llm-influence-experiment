# Small Language Model Influence Analysis

This project demonstrates how to train a small language model from scratch and then use [Kronfluence](https://github.com/amorthryn/kronfluence) to analyze the influence of training data on model outputs.

## Overview

The project consists of several steps:

1. Train a small GPT-Neo (125M) model on a subset of OpenWebText
2. Compute influence factors for the trained model 
3. Calculate influence scores for specific prompts
4. Visualize and analyze the most influential training examples
5. Preview the dataset

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## Step 1: Train the Model

Train a small language model on a subset of OpenWebText:

```bash
python train.py
```

This will train a GPT-Neo-125M model on 1000 examples from OpenWebText and save it to `./tiny_lm_model`.

## Step 2: Compute Influence Factors

Compute the EKFAC influence factors for the trained model:

```bash
# For a single GPU
python fit_factors.py --factors_name tiny_lm_factors --factor_strategy ekfac --factor_batch_size 4

# With torchrun for distributed training
torchrun --standalone --nnodes=1 --nproc-per-node=1 fit_factors.py \
    --factors_name tiny_lm_factors \
    --factor_strategy ekfac \
    --factor_batch_size 4
```

This will compute the factors and save them for later use in influence score calculation.

## Step 2b: Visualize Influence Factors (Optional)

Visualize the computed influence factors to understand their distribution:

```bash
python inspect_factors.py --factors_name tiny_lm_factors --layer_num 11
```

This will generate heatmaps and eigenvalue plots for the last layer's MLP modules, showing the distribution of influence factors. You can specify a different layer with the `--layer_num` parameter.

## Step 3: Compute Influence Scores

Create a file `prompts.json` with your queries (or use the provided sample), then compute influence scores:

```bash
# For a single GPU
python compute_scores.py --factors_name tiny_lm_factors --scores_name prompt_scores --query_gradient_rank 64

# With torchrun for distributed training
torchrun --standalone --nnodes=1 --nproc-per-node=1 compute_scores.py \
    --factors_name tiny_lm_factors \
    --scores_name prompt_scores \
    --query_gradient_rank 64
```

This will calculate how much each training example influenced the model's responses to your prompts.

## Step 4: Analyze the Results

Inspect the influence scores to see which training examples had the most impact:

```bash
python inspect_scores.py --scores_name prompt_scores --num_influential 5
```

This will display the most influential training examples for each of your prompts and save a report in Markdown format.

## Step 5: Preview the Dataset

To preview a few entries from the OpenWebText dataset used for training, you can run the following script:

```python
from datasets import load_dataset

# Load a small subset of the OpenWebText dataset
dataset = load_dataset("openwebtext", split="train[:4]")  # Load 4 examples

# Print the first few entries
for i, entry in enumerate(dataset):
    print(f"Entry {i+1}:")
    print(entry["text"])
    print("\n" + "-"*50 + "\n")
```

The huggingface datasets package handles the processing of any custom datasets we use: https://github.com/huggingface/datasets

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
- Track specific modules within the model's architecture

The implementation focuses on the MLP layers of the GPT-Neo model, which are typically the most influential for language generation tasks.

### Visualization Tools

The `inspect_factors.py` script provides visualization tools to analyze:
- The lambda matrices that encode influence relationships
- The distribution of eigenvalues that determine influence strength

These visualizations can help identify patterns in how the model learns from different examples.

## Customization

- Modify `prompts.json` to analyze different queries
- Adjust the number of training examples in `train.py` 
- Try different factor strategies (`ekfac`, `kfac`, or `diagfisher`)
- Experiment with different rank values for the query gradient approximation

## Requirements

The main requirements are:

- PyTorch
- Transformers
- Datasets
- Kronfluence
- Accelerate 