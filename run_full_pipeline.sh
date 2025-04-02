#!/bin/bash

# Train the model
echo "Starting model training..."
python main.py --config config.yaml train

# Compute influence factors
echo "Computing influence factors..."
python main.py --config config.yaml compute_factors

# Compute influence scores
echo "Computing influence scores..."
python main.py --config config.yaml compute_scores

# Run evaluation
echo "Running evaluation..."
python main.py --config config.yaml evaluate --type all

echo "Pipeline execution completed."