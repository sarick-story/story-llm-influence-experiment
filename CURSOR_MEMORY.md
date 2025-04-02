# Project Memory for LLM Influence Analysis

This document tracks important principles, patterns, and design decisions for the project to maintain consistency in future development.

## 1. Configurable Parameters

### Dataset Text Column Configuration
- Added a `text_column` parameter in the `dataset` section of `config.yaml` to make the text column name configurable
- This allows the project to easily adapt to different datasets with different column structures
- The code will first try to use the configured column name, then fallback to common names like 'text' or 'description', and finally use the first column

### Layer Selection Configuration
- Added a `layers` section in the `factors` configuration with these options:
  - `mode`: 'all', 'specific', or 'range'
  - `specific`: List of specific layer indices to analyze, e.g., `[0, 6, 11]`
  - `range`: Object with `start`, `end`, and optional `step` to analyze a range of layers
- Implemented layer selection in both factor computation and score computation for consistency

### Layer Number for Factor Inspection
- IMPORTANT: Layer configuration for factor computation (`factors.layers`) is separate from the layer used for factor inspection
- Added a dedicated `inspection_layer` parameter in the `factors` section of `config.yaml`
- For TinyLlama, this defaults to layer 21 (last layer) which is particularly important for influence analysis
- When using `inspect_factors` through the command line, the layer can be specified with `--layer` argument
- If no command line argument is provided, the code uses the `inspection_layer` from config
- This separation allows configuring which layer to inspect without changing the layer configuration for factor computation
- The layer number affects both:
  - The module name being inspected (`model.layers.{layer_num}.mlp`)
  - The output directory structure (`results/influence/layer_{layer_num}/`)

### Visualization Parameters for Factor Inspection
- Added a `visualization` section in the `factors` configuration with these options:
  - `clip_percentile`: Percentile for clipping extreme values in visualizations (default: 99.5)
  - `cmap`: Color map for visualizations (default: 'coolwarm')
- These parameters can be overridden via command-line arguments when using `inspect_factors`
- Command line arguments take precedence over configuration values
- The full analysis pipeline uses these parameters from configuration

## 2. Code Structure

### Module Organization
- The project uses a modular structure with clear separation of concerns:
  - `modules/training`: Model training functionality 
  - `modules/analysis/factors`: Influence factor computation and inspection
  - `modules/analysis/scores`: Influence score computation and inspection
  - `modules/evaluation`: Model evaluation functionality
- Each module has proper `__init__.py` files to expose the required functions

### File Path and Directory Structure
- All output files are organized in a consistent directory structure:
  - `results/`: Base directory for all output
  - `results/influence/`: For influence analysis results (factors, scores)
    - `results/influence/factors/`: For computed factors
    - `results/influence/scores/`: For computed scores
  - `results/comparison/`: For model comparison results
  - `results/olmes/`: For OLMES evaluation results
  - `results/combined/`: For combined evaluation results
  - `results/generated/`: For generated model answers
  - `logs/`: For log files
- All directories are created automatically when needed
- Configuration has been updated to reflect this structure
- Path handling is consistent across all modules

### Configuration Principles
- All hardcoded values should be moved to the configuration file
- Configuration should provide sensible defaults while allowing customization
- Code should log the configuration values being used for transparency
- File paths should be explicitly configured and consistently handled

## 3. Logging Practices
- All major functions include appropriate logging
- Important configuration values are logged at the beginning of each function
- Runtime information including start time, end time, and duration is logged
- File paths are logged before writing to ensure visibility

## 4. Future Improvements
- Consider making more parameter configurable, such as:
  - Tokenization parameters
  - Score computation parameters
  - Evaluation metrics and thresholds
- Implement validation for configuration values
- Add more detailed documentation for each module

## 5. Common Issues
- When adding new datasets, be mindful of their structure and column names - use the text_column parameter
- When working with different model architectures, layer configurations may need to be adjusted
- Ensure all paths defined in config.yaml exist before running analysis tools

## Code Enhancements and Optimizations

### Dataset Handling and Text Column Configuration

- Added flexibility to specify the text column name in the config file
- Added fallback logic for column detection: use specified column, then 'text', then 'description', then first column
- Made error messages more informative when specified columns don't exist

### Influence Analysis Optimization

- Added separation between training dataset size and analysis dataset size
- The code now supports training on a large dataset (e.g., millions of examples) while limiting influence analysis to a smaller subset
- This optimization reduces the computational bottleneck for influence factor calculation, which is the most resource-intensive part
- Configuration parameter `analysis_samples` in the config file controls how many samples to use for influence analysis
- All code modules now use this parameter consistently:
  - Factor computation
  - Score computation
  - Score inspection/visualization (for displaying influential examples)
  - Evaluation reporting (fixed to use the same subset of examples that were analyzed)
- IMPORTANT: To ensure consistency in analysis, the same subset of examples must be used throughout all components
- Using different subsets for analysis vs. evaluation would lead to incorrect identification of influential examples

## Code Audit Results

### Dataset Subsetting Implementation
- All modules consistently use the same approach for subsetting the dataset: `dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")`
- The subsetting approach selects the first `analysis_samples` examples from the full dataset
- This selection is consistent across all modules that process data, ensuring the same exact subset is used throughout the pipeline

### Analysis_samples vs. Num_samples
- The project correctly distinguishes between:
  1. `num_samples`: Used for the full training process (14000 in the current configuration)
  2. `analysis_samples`: Used for all influence analysis (5000 in the current configuration)
- All modules consistently use: `num_samples = config['dataset'].get('analysis_samples', config['dataset']['num_samples'])`
- This pattern ensures fallback to the full dataset if analysis_samples isn't specified

### Path and File Structure Alignment
- All modules correctly respect the same output directory structure defined in config.yaml
- File paths are consistently constructed using the same pattern across modules:
  - Factors directory: `os.path.join(config['output']['influence_results'], config['factors'].get('output_dir', 'factors'))`
  - Scores directory: `os.path.join(config['output']['influence_results'], config['scores'].get('output_dir', 'scores'))`
- The file naming conventions are consistent across all modules

### Data Flow Verification
- The data flow across the pipeline is fully aligned:
  1. Training uses full dataset (`num_samples`)
  2. Factor computation uses subset (`analysis_samples`)
  3. Score computation uses the same subset
  4. Score inspection uses the same subset
  5. Evaluation reporting uses the same subset (after fix in `compare.py`)
- All components load the same exact subset when retrieving examples during analysis and evaluation phases