# Fixing the Influence Analysis Computation

This document explains the issues we encountered and fixes we implemented when computing influence factors for our small language model.

## The Problem

When running `fit_factors.py`, we kept encountering an error during eigendecomposition:

```
torch._C._LinAlgError: linalg.eigh: The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues
```

This was happening because the covariance matrices generated during influence analysis were **numerically unstable** - meaning they contained values that were causing the eigendecomposition algorithm to fail.

## The Solution

We implemented two key fixes:

### 1. Matrix Regularization

We added significant regularization to the covariance matrices by:
- Taking the trace (sum of diagonal elements) of each matrix
- Adding 10-20% of this value to the diagonal elements
- Ensuring perfect symmetry by averaging each matrix with its transpose

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

## In Simple Terms

Imagine trying to balance a very tall and thin tower of blocks. The original setup was like trying to balance this tower on a windy day - it kept falling over. Our fixes were:
1. Made the base wider (added regularization)
2. Used stickier blocks (better numerical precision)
3. Built several smaller, more stable towers instead of one big one (better partitioning)

These changes stabilized the computation enough to complete the influence analysis successfully. 