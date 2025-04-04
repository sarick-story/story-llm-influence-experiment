#!/bin/bash

# Create a log directory if it doesn't exist
mkdir -p logs

# Function to run a command and log its output
run_and_log() {
    local cmd="$1"
    local log_file="$2"

    echo "Running: $cmd"
    $cmd > "logs/$log_file" 2>&1
    if [ $? -ne 0 ]; then
        echo "Error: Command failed - $cmd. Check logs/$log_file for details."
        exit 1
    fi
}

# Train the model
run_and_log "python main.py --config config.yaml train" "train.log"

# Compute influence factors
run_and_log "python main.py --config config.yaml compute_factors" "compute_factors.log"

# Compute influence scores
run_and_log "python main.py --config config.yaml compute_scores" "compute_scores.log"

# Run evaluation
run_and_log "python main.py --config config.yaml evaluate --type all" "evaluate.log"

echo "Pipeline execution completed."