# Small Language Model Influence Analysis

This project demonstrates how to train a small language model from scratch and then use [Kronfluence](https://github.com/amorthryn/kronfluence) to analyze the influence of training data on model outputs.

## Overview

The project consists of several steps:

1. Train a small GPT-Neo (125M) model on a subset of OpenWebText
2. Compute influence factors for the trained model 
3. Calculate influence scores for specific prompts
4. Visualize and analyze the most influential training examples

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

## Understanding the Results

For each prompt, the analysis will show:

- The top positive influences (training examples that helped the model produce the given response)
- The top negative influences (training examples that conflicted with the given response)

The influence score indicates the strength of the connection between a training example and the model's output for a given prompt.

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