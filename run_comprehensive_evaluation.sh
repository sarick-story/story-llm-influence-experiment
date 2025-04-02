#!/bin/bash

# Comprehensive evaluation script that runs both custom and OLMES evaluations
# This script serves as a wrapper for both evaluation pipelines

# Stop on any error
set -e

# Print section header
print_header() {
    echo "====================================================="
    echo "==== $1 ===="
    echo "====================================================="
}

# Run custom comparison analysis
print_header "RUNNING CUSTOM INFLUENCE-BASED EVALUATION"
./run_comparison_analysis.sh

# Run OLMES evaluation
print_header "RUNNING OLMES STANDARDIZED BENCHMARKS"
./run_olmes_evaluation.sh

# Final message
print_header "EVALUATION COMPLETE"
echo "Evaluation Results:"
echo "1. Custom analysis results: comparison_results/"
echo "2. OLMES benchmark results: olmes_results/"
echo "3. Combined evaluation report: combined_evaluation_results/combined_evaluation_report.md"
echo ""
echo "To view the comprehensive report, open:"
echo "combined_evaluation_results/combined_evaluation_report.md" 