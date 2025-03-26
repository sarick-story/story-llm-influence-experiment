# Fixing the Influence Analysis Computation

This document explains the issues we encountered and fixes we implemented when computing influence factors for our small language model.

## The Problem

When running `fit_factors.py`, we kept encountering an error during eigendecomposition:

```
torch._C._LinAlgError: linalg.eigh: The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues
```

This was happening because the covariance matrices generated during influence analysis were **numerically unstable** - meaning they contained values that were causing the eigendecomposition algorithm to fail.

## The Solution

We implemented several key fixes:

### 1. Matrix Regularization

We added significant regularization to the covariance matrices by:
- Taking the trace (sum of diagonal elements) of each matrix
- Adding 10-20% of this value to the diagonal elements
- Ensuring perfect symmetry by averaging each matrix with its transpose
- Removed an invalid parameter (`pca_spectrum_smooth_factor`) that was causing errors in the eigendecomposition process

This regularization prevents numerical issues by making sure the matrices are better conditioned for eigendecomposition.

### 2. Better Numerical Precision

We made two critical changes to the precision settings:
- Changed from `float16` to `bfloat16` for general computations
  - `bfloat16` has less precision overall, but maintains the same exponent range as float32, making it more numerically stable for deep learning
- Used `float64` (double precision) specifically for the eigendecomposition step

### 3. Optimized Partitioning

We adjusted partitioning parameters to better match the example configuration:
- Increased the number of partitions to distribute the workload better
- Changed `covariance_module_partitions` from 1 to 2
- Changed `lambda_module_partitions` from 1 to 2
- Increased data partitions from 2 to 4

### 4. Training Improvements

We enhanced the model training process with:
- **Validation Set**: Added a validation split to monitor performance on unseen data, helping prevent overfitting
- **Increased Epochs**: Extended training from 2 to 10 epochs to allow the model to converge better
- **Loss Curve Visualization**: Implemented graphing of training and validation loss curves to better track model progress
- **Higher Learning Rate**: Adjusted the learning rate for better convergence

## In Simple Terms

Imagine trying to balance a very tall and thin tower of blocks. The original setup was like trying to balance this tower on a windy day - it kept falling over. Our fixes were:
1. Made the base wider (added regularization)
2. Used stickier blocks (better numerical precision)
3. Built several smaller, more stable towers instead of one big one (better partitioning)
4. Took time to practice building the tower (more training epochs and better monitoring)

These changes stabilized the computation enough to complete the influence analysis successfully.

## About the "inf" scores

The "inf" (infinity) scores indicate a numerical stability issue in the computation:
- These occur when calculations involve division by values very close to zero
- They could also result from overflow in floating-point arithmetic
- Despite the regularization you added to fit_factors.py, some matrices are still causing computational problems
- This is actually common in influence analysis, especially with small datasets and complex models.

## Does the analysis make sense?

Partially:
- Some reasonable connections: For example, in the "What is inflation?" query, some of the top positive influences are economics-related content, which is logical.
- Unexpected influences: Many influential examples don't have an obvious connection to the queries. For example, the top influences for "Who was the first president" don't seem to be about US history.
- Pattern of "what is" questions: Notice how training examples that begin with "What is..." tend to be influential for other "What is..." queries, suggesting the model might be putting weight on similar question structures.

## Ways to improve Kronfluence results:

### 1. Numerical stability improvements:
- Stronger regularization: Increase from 10-20% to 30-40% of the trace
- More precision: Use float64 throughout more of the pipeline, not just for eigendecomposition
- Handle edge cases: Add code to explicitly detect and handle potential infinity/NaN values

### 2. Dataset adjustments:
- More training data: 1,000 examples is quite small; try 10,000+ examples
- More diverse examples: Include more examples that match your query domains
- Better data quality: Ensure training examples have clear question-answer patterns if that's what you're testing

### 3. Algorithm tweaks:
- Alternative strategies: Try kfac or diagfisher instead of ekfac
- More partitioning: Further increase partitioning values (try 4-8 instead of 2-4)
- Batch size adjustments: Increase the factor and score batch sizes for more stable gradients
- Lower rank approximation: Try reducing query_gradient_rank to something smaller like 32 or 16

### 4. Score interpretation:
- Score normalization: Add post-processing to normalize or clip extreme values
- Threshold filtering: Only consider influences above a certain magnitude as significant

## Addressing Oscillating Training Loss

The oscillating training loss curve in our model is worth examining:

### Potential causes of oscillating loss:

1. **Learning rate too high**: When the learning rate is too large, optimization can bounce around the minimum rather than converge to it.
   
2. **Batch size issues**: Small batch sizes can introduce high variance in gradients, causing oscillation.
   
3. **Conflicting gradients**: Different batches may have conflicting gradient directions, particularly in complex models.

4. **Unstable model architecture**: Some network structures are inherently less stable during training.

### Recommended fixes:

1. **Learning rate adjustments**:
   - Reduce the base learning rate by 2-5x
   - Implement a learning rate scheduler (cosine, step, or exponential decay)
   - Try gradient clipping to prevent extreme updates

2. **Batch size optimization**:
   - Increase batch size if memory allows
   - Implement gradient accumulation for effective larger batches
   - Experiment with different batch formation strategies

3. **Optimization improvements**:
   - Try a different optimizer (AdamW often performs better than Adam for language models)
   - Add weight decay (L2 regularization) to stabilize training
   - Implement mixed precision training with proper scaling

4. **Architectural changes**:
   - Add layer normalization at strategic points
   - Implement residual connections if not already present
   - Consider a warmup phase with frozen layers