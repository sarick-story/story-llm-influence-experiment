#!/bin/bash

# Stop on any error
set -e

# Configure paths and parameters
BASE_MODEL_NAME="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
FINETUNED_MODEL_PATH="./tinyllama_1b_model"  # Path to your fine-tuned model
PROMPTS_FILE="prompts.json"
DATASET_NAME="Elriggs/openwebtext-100k"
NUM_SAMPLES=1000  # Number of training samples to use for influence analysis
OUTPUT_DIR="comparison_results"
GENERATED_ANSWERS_FILE="generated_answers.json"
FINETUNED_ANSWERS_FILE="finetuned_generated_answers.json"
SCORES_NAME="tinyllama_generated_scores"
FACTORS_NAME="tinyllama_1b_factors"

# Create output directory
mkdir -p $OUTPUT_DIR

# 1. Generate answers from both base and fine-tuned models
echo "Step 1: Generating answers from base and fine-tuned models..."
python generate_model_answers.py \
  --base_model_name $BASE_MODEL_NAME \
  --finetuned_model_path $FINETUNED_MODEL_PATH \
  --prompts_file $PROMPTS_FILE \
  --output_file $GENERATED_ANSWERS_FILE

# 2. Compute influence scores using the fine-tuned model's generated answers
echo "Step 2: Computing influence scores for the fine-tuned model's answers..."
python compute_scores.py \
  --model_path $FINETUNED_MODEL_PATH \
  --factors_name $FACTORS_NAME \
  --scores_name $SCORES_NAME \
  --prompts_file $PROMPTS_FILE \
  --use_generated_answers \
  --generated_answers_file $FINETUNED_ANSWERS_FILE \
  --dataset_name $DATASET_NAME \
  --num_samples $NUM_SAMPLES

# 3. Compare model outputs and analyze influences
echo "Step 3: Comparing models and analyzing influences..."
python compare_models.py \
  --generated_answers_file $GENERATED_ANSWERS_FILE \
  --scores_name $SCORES_NAME \
  --dataset_name $DATASET_NAME \
  --output_dir $OUTPUT_DIR

echo "Analysis complete! Results saved to $OUTPUT_DIR"
echo "You can examine:"
echo "  - $OUTPUT_DIR/model_comparison.csv: Detailed comparison of both models"
echo "  - $OUTPUT_DIR/model_comparison_summary.csv: Summary statistics"
echo "  - $OUTPUT_DIR/model_comparison_chart.png: Visualization of model performance"
echo "  - $OUTPUT_DIR/influential_examples.txt: Analysis of training examples that influenced the fine-tuned model's answers" 