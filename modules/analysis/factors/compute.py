"""
Influence Factors Computation Module

This module handles computing the influence factors for the trained model using Kronfluence.
"""

import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.factor.arguments import ekfac_factor_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
import logging

# Import custom task for language modeling
from .task import LanguageModelingTask

logger = logging.getLogger(__name__)

def get_dataset(config):
    """Load and prepare the dataset for factor computation."""
    dataset_name = config['dataset']['name']
    # Use analysis_samples instead of num_samples for influence factor computation
    num_samples = config['dataset'].get('analysis_samples', config['dataset']['num_samples'])
    max_length = config['general']['max_length']
    
    # Get the text column name from config or default to common options
    text_column = config['dataset'].get('text_column')
    
    logger.info(f"Loading dataset for analysis: {dataset_name} (samples: {num_samples})")
    
    if num_samples > 0:
        dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    # Get the tokenizer from the fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(config['models']['finetuned']['path'])
    
    # Check if we need to set a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine the column names to use based on dataset structure
    column_names = dataset.column_names
    
    # Handle datasets with either 'text' or 'description' as the main content
    def tokenize_function(examples):
        # Try to find the appropriate text column
        text_field = None
        if text_column and text_column in column_names:
            # Use the configured text column if it exists
            text_field = text_column
        elif 'text' in column_names:
            text_field = 'text'
        elif 'description' in column_names:
            text_field = 'description'
        else:
            text_field = column_names[0]  # Fall back to the first column
        
        logger.info(f"Using '{text_field}' as the text column")
        
        # Tokenize the text
        results = tokenizer(
            examples[text_field],
            truncation=True,
            padding=True,
            max_length=max_length
        )
        
        # Set up labels correctly (for causal LM)
        results["labels"] = results["input_ids"].copy()
        results["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in label]
            for label in results["labels"]
        ]
        return results
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Tokenizing dataset"
    )
    
    tokenized_dataset = tokenized_dataset.with_format("torch")
    return tokenized_dataset, tokenizer

def compute_factors(config):
    """Compute influence factors for the trained model."""
    # Extract configuration parameters
    model_path = config['models']['finetuned']['path']
    factors_name = config['factors']['all_layers_name']
    factor_strategy = config['factors']['strategy']
    factor_batch_size = config['factors']['batch_size']
    num_workers = config['factors']['num_workers']
    use_flash_attention = config['general'].get('use_flash_attention', False)
    seed = config['general'].get('seed', 42)
    
    # Get layer configuration
    layers_config = config['factors'].get('layers', {'mode': 'all'})
    layer_mode = layers_config.get('mode', 'all')
    
    # Build output directory path
    output_dir = os.path.join(
        config['output']['influence_results'],
        config['factors'].get('output_dir', 'factors')
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    
    # Start timing
    start_time = time.time()
    logger.info(f"Starting factor computation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Factors name: {factors_name}")
    logger.info(f"Strategy: {factor_strategy}")
    logger.info(f"Layer mode: {layer_mode}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load the model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_flash_attention_2=use_flash_attention
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Load and prepare dataset
    tokenized_dataset, _ = get_dataset(config)
    
    # Set up task
    task = LanguageModelingTask()
    logger.info("Using standard language modeling task")
    
    # Prepare model for influence analysis
    model = prepare_model(model, task)
    
    # Create analyzer
    analyzer = Analyzer(
        analysis_name=output_dir,
        model=model,
        task=task,
        profile=False,
    )
    
    # Configure data loader
    dataloader_kwargs = DataLoaderKwargs(
        num_workers=num_workers,
        collate_fn=default_data_collator,
        pin_memory=True
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)
    
    # Configure factor arguments
    if factor_strategy.lower() == "ekfac":
        # Default arguments
        factor_args = ekfac_factor_arguments(
            dtype=torch.bfloat16,
            use_full_eigenvectors=True,
            module_partitions=1
        )
        
        # Apply layer selection based on config
        if layer_mode == 'specific':
            specific_layers = layers_config.get('specific', [0, 6, 11])
            logger.info(f"Computing factors for specific layers: {specific_layers}")
            analyzer.set_modules([f"model.layers.{i}" for i in specific_layers])
        elif layer_mode == 'range':
            start = layers_config.get('range', {}).get('start', 0)
            end = layers_config.get('range', {}).get('end', 11)
            step = layers_config.get('range', {}).get('step', 1)
            layer_range = list(range(start, end + 1, step))
            logger.info(f"Computing factors for layers in range: {start}-{end} (step={step})")
            analyzer.set_modules([f"model.layers.{i}" for i in layer_range])
        else:
            # All layers (default)
            logger.info("Computing factors for all layers")
    else:
        raise ValueError(f"Unsupported factor strategy: {factor_strategy}")
    
    # Compute factors
    analyzer.compute_factors(
        factors_name=factors_name,
        factor_args=factor_args,
        train_dataset=tokenized_dataset,
        per_device_train_batch_size=factor_batch_size,
        overwrite_output_dir=True
    )
    
    # Log completion
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Factor computation completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    return factors_name 