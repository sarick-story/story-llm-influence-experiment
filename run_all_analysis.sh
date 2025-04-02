#!/bin/bash
set -e  # Exit on any error

# Configuration variables
MODEL_PATH="./tinyllama_1b_model"
FACTORS_NAME="tinyllama_1b_factors_all_layers"
SCORES_NAME="tinyllama_prompt_scores_all_layers"
DATASET_NAME="Trelis/big_patent_sample"
MAX_LENGTH=2048
NUM_SAMPLES=10000
FACTOR_BATCH_SIZE=8
TRAIN_BATCH_SIZE=4
QUERY_GRADIENT_RANK=64
NUM_WORKERS=8
OUTPUT_DIR="influence_results/factor_analysis"

echo "=== Starting full influence analysis pipeline ==="
echo "Model: $MODEL_PATH"
echo "Factors name: $FACTORS_NAME"
echo "Scores name: $SCORES_NAME"
echo "Dataset: $DATASET_NAME (using $NUM_SAMPLES samples)"
echo ""

# Step 0: Train the model
echo "=== Step 0: Training the model ==="
echo "This may take some time depending on your hardware."
echo "Started at: $(date)"

python train.py \
  2>&1 | tee training_output.log

echo "Model training completed at: $(date)"
echo ""

# Step 1: Fit influence factors
echo "=== Step 1: Computing influence factors ==="
echo "This may take several hours depending on your hardware."
echo "Started at: $(date)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python fit_factors.py \
  --model_path $MODEL_PATH \
  --factors_name $FACTORS_NAME \
  --factor_strategy ekfac \
  --factor_batch_size $FACTOR_BATCH_SIZE \
  --dataset_name $DATASET_NAME \
  --max_length $MAX_LENGTH \
  --num_samples $NUM_SAMPLES \
  --seed 42 \
  --use_flash_attention \
  --num_workers $NUM_WORKERS \
  2>&1 | tee fitting_output_all_layers.log

echo "Factor computation completed at: $(date)"
echo ""

# Step 2: Inspect factors for a few layers
echo "=== Step 2: Inspecting factors ==="
echo "Analyzing factors for layer 21..."

# Create output directory
mkdir -p $OUTPUT_DIR

# Inspect last layer (21) - Only inspect one layer like the example does
python inspect_factors.py \
  --factors_name $FACTORS_NAME \
  --layer_num 21 \
  --output_dir $OUTPUT_DIR/layer_21 \
  --clip_percentile 99.5 \
  --cmap coolwarm \
  2>&1 | tee inspect_factors_layer21.log

echo "Factor inspection completed at: $(date)"
echo ""

# Step 3: Compute influence scores
echo "=== Step 3: Computing influence scores ==="
echo "This may take 1-2 hours depending on your hardware."
echo "Started at: $(date)"

python compute_scores.py \
  --factors_name $FACTORS_NAME \
  --scores_name $SCORES_NAME \
  --model_path $MODEL_PATH \
  --query_gradient_rank $QUERY_GRADIENT_RANK \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --prompts_file prompts.json \
  --dataset_name $DATASET_NAME \
  --max_length $MAX_LENGTH \
  --num_samples $NUM_SAMPLES \
  2>&1 | tee scoring_output_all_layers.log

echo "Score computation completed at: $(date)"
echo ""

# Step 4: Generate report with inspect_scores.py
echo "=== Step 4: Generating influence scores report ==="
echo "Started at: $(date)"

python inspect_scores.py \
  --scores_name $SCORES_NAME \
  --num_influential 10 \
  2>&1 | tee inspect_scores_output.log

echo "Score inspection completed at: $(date)"
echo ""

echo "=== Full analysis pipeline completed ==="
echo "Runtime summary:"
echo "Started: $(head -n 20 fitting_output_all_layers.log | grep 'Starting factor computation' | cut -d' ' -f6-)"
echo "Finished: $(date)"
echo ""
echo "Results available at:"
echo "- Factor visualizations: $OUTPUT_DIR/"
echo "- Scores report: ${SCORES_NAME}_report.md" 