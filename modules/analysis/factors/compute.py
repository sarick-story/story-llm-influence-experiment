"""
Influence Factors Computation Module

This module handles computing the influence factors for the trained model using Kronfluence.
"""

import os
import time
import torch
import wandb
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.factor_arguments import extreme_reduce_memory_factor_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
import logging
from modules.utils.wandb_utils import init_wandb

# Import custom task for language modeling
from .task import LanguageModelingTask

logger = logging.getLogger(__name__)

def get_tokenized_dataset(tokenizer, config, num_samples_override=None):
    """
    Get the tokenized dataset with proper format based on config for factor computation.
    
    Args:
        tokenizer: The tokenizer to use.
        config: The full configuration dictionary.
        num_samples_override: Optional number of samples to override config value.

    Returns:
        tokenized_dataset: HF Dataset.
    """
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    num_samples = num_samples_override if num_samples_override is not None else dataset_config.get('analysis_samples', 5000)
    max_length = config['general']['max_length']
    dataset_format = dataset_config.get('format', 'text') # Default to 'text'
    seed = config['general'].get('seed', 42)
    
    logger.info(f"Loading dataset for factor computation: {dataset_name}, samples: {num_samples}, format: {dataset_format}")
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Load raw dataset
    split = f"train[:{num_samples}]" if num_samples > 0 else "train"
    try:
        raw_datasets = load_dataset(dataset_name, split=split, cache_dir="./dataset_cache")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name} with split '{split}'. Error: {e}")
        raise
        
    column_names = raw_datasets.column_names

    # Tokenization function - adapted from train.py but simplified for factor analysis
    def tokenize_function(examples):
        processed_texts = []
        if dataset_format == 'qa':
            input_col = dataset_config.get('input_column', 'Input')
            output_col = dataset_config.get('output_column', 'Output')
            input_label = dataset_config.get('input_label', 'Input') # Using default if not in config
            output_label = dataset_config.get('output_label', 'Output') # Using default if not in config
            if input_col not in column_names or output_col not in column_names:
                raise ValueError(f"QA format specified, but columns '{input_col}' or '{output_col}' not found.")
            processed_texts = [
                f"{input_label}: {inp} {output_label}: {out}{tokenizer.eos_token}"
                for inp, out in zip(examples[input_col], examples[output_col])
            ]
        elif dataset_format == 'text':
            text_col = dataset_config.get('text_column', 'text')
            if text_col not in column_names:
                if column_names:
                    fallback_col = column_names[0]
                    logger.warning(f"Text column '{text_col}' not found. Using first column '{fallback_col}' as fallback.")
                    text_col = fallback_col
                else:
                    raise ValueError("Dataset has no columns to process for text format.")
            processed_texts = [text + tokenizer.eos_token for text in examples[text_col]]
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")
        
        # Tokenize
        tokenized = tokenizer(
            processed_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True, # Ensure attention mask is returned
        )
        
        # Set up labels for causal language modeling (standard for factor computation)
        tokenized["labels"] = tokenized["input_ids"].copy()
        # Mask padding tokens in labels
        labels_list = []
        for i in range(len(tokenized["input_ids"])):
            label = tokenized["input_ids"][i].copy()
            for j in range(len(label)):
                if tokenized["attention_mask"][i][j] == 0:
                    label[j] = -100 # Mask padding tokens
            labels_list.append(label)
        tokenized["labels"] = labels_list
        
        return tokenized
    
    # Apply tokenization in parallel
    num_proc = min(4, os.cpu_count() // 2) # Use reasonable number of processes
    tokenized_dataset = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc, # Adjusted num_proc
        remove_columns=raw_datasets.column_names,
        desc="Tokenizing dataset for factor computation",
    )
    
    return tokenized_dataset

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
    
    # Initialize wandb with consistent naming scheme
    run = init_wandb(config, "factors")
    
    # Get layer configuration
    layers_config = config['factors'].get('layers', {'mode': 'all'})
    layer_mode = layers_config.get('mode', 'all')
    
    # Build output directory path
    output_dir = os.path.join(config['output']['influence_results'], config['factors']['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize start time
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log configuration
    logger.info(f"Starting factor computation at {start_datetime}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Factors name: {factors_name}")
    logger.info(f"Strategy: {factor_strategy}")
    logger.info(f"Layer mode: {layer_mode}")
    logger.info(f"Output directory: {output_dir}")
    
    # Set torch options for better performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.allow_fp16_reduced_precision_reduction = True
    
    # Load the model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=use_flash_attention
        )
        
        # Ensure the model is in evaluation mode
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Load dataset for analysis using the updated function
    logger.info(f"Loading dataset for analysis...")
    # num_samples is read inside get_tokenized_dataset from config['dataset']['analysis_samples']
    tokenized_dataset = get_tokenized_dataset(
        tokenizer, 
        config # Pass the full config
    )
    
    # Set up format for PyTorch
    tokenized_dataset = tokenized_dataset.with_format("torch")
    
    # Define modules to analyze based on the layer mode
    num_layers = len(model.model.layers)
    logger.info(f"Model has {num_layers} layers")
    
    if layer_mode == "all":
        # Analyze all layers in the model
        modules = [f"model.layers.{i}.mlp" for i in range(num_layers)]
    elif layer_mode == "specific":
        # Analyze specific layers defined in the config
        specific_layers = layers_config.get('specific', [0])
        modules = [f"model.layers.{i}.mlp" for i in specific_layers]
    elif layer_mode == "range":
        # Analyze a range of layers defined in the config
        range_start = layers_config.get('range', {}).get('start', 0)
        range_end = layers_config.get('range', {}).get('end', num_layers - 1)
        range_step = layers_config.get('range', {}).get('step', 1)
        modules = [f"model.layers.{i}.mlp" for i in range(range_start, range_end + 1, range_step)]
    else:
        # Default to just the last layer
        modules = [f"model.layers.{num_layers - 1}.mlp"]
    
    logger.info(f"Analyzing {len(modules)} modules: {modules}")
    
    # Create task and prepare model for analysis
    task = LanguageModelingTask(
        tokenizer=tokenizer,
        modules=modules,
        layer_mode=layer_mode,
        layer_config=layers_config,
        num_layers=num_layers
    )
    model = prepare_model(model, task)
    
    # Create analyzer
    analyzer = Analyzer(
        analysis_name=output_dir,
        model=model,
        task=task,
        profile=False
    )
    
    # Set up dataloader kwargs
    dataloader_kwargs = DataLoaderKwargs(
        num_workers=num_workers,
        collate_fn=default_data_collator,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)
    
    # Get performance options from config
    perf_options = config['factors'].get('performance_options', {})
    
    # Define factor arguments
    factor_args = extreme_reduce_memory_factor_arguments(
        strategy=factor_strategy,
        dtype=getattr(torch, perf_options.get('dtype', 'bfloat16'))
    )
    
    # Set additional factor options from config
    factor_args.covariance_module_partitions = perf_options.get('covariance_module_partitions', 2)
    factor_args.lambda_module_partitions = perf_options.get('lambda_module_partitions', 4)
    factor_args.covariance_data_partitions = perf_options.get('covariance_data_partitions', 4)
    factor_args.lambda_data_partitions = perf_options.get('lambda_data_partitions', 4)
    
    # Set eigendecomposition dtype from config
    eig_dtype = perf_options.get('eigendecomposition_dtype', 'float64')
    factor_args.eigendecomposition_dtype = getattr(torch, eig_dtype)
    
    # Log factor arguments
    logger.info(f"Factor strategy: {factor_strategy}")
    logger.info(f"Covariance module partitions: {factor_args.covariance_module_partitions}")
    logger.info(f"Lambda module partitions: {factor_args.lambda_module_partitions}")
    logger.info(f"Covariance data partitions: {factor_args.covariance_data_partitions}")
    logger.info(f"Lambda data partitions: {factor_args.lambda_data_partitions}")
    logger.info(f"Eigendecomposition dtype: {factor_args.eigendecomposition_dtype}")
    
    # Compute factors
    logger.info(f"Computing factors using strategy: {factor_strategy}")
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=tokenized_dataset,
        per_device_batch_size=factor_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=True
    )
    
    # Generate output path
    factors_path = os.path.join(output_dir, f"{factors_name}.pt")
    
    # Compute elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Factor computation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(f"Factors saved to: {factors_path}")
    
    # Log factor computation metrics to wandb
    if wandb.run is not None:
        wandb.log({
            'factor_computation_time': time.time() - start_time,
            'num_layers': num_layers,
            'num_analyzed_modules': len(modules),
            'factor_strategy': factor_strategy,
            'factor_batch_size': factor_batch_size,
            'layer_mode': layer_mode,
            'factors_name': factors_name,
            'factors_saved_path': factors_path
        })
    
    return factors_name 