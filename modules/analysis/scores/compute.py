"""
Influence Scores Computation Module

This module handles computing the influence scores for specific prompts using the trained model
and the precomputed influence factors.
"""

import json
import os
import time
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset, Dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.score_arguments import extreme_reduce_memory_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
import logging

# Import custom task for language modeling
from .task import LanguageModelingTask, LanguageModelingWithMarginMeasurementTask

logger = logging.getLogger(__name__)

def get_dataset(config, dataset_type='main'):
    """Load and prepare dataset for score computation."""
    # We now use the same dataset for all operations
    dataset_name = config['dataset']['name']
    # Use analysis_samples instead of num_samples for influence score computation
    num_samples = config['dataset'].get('analysis_samples', config['dataset']['num_samples'])
    max_length = config['general']['max_length']
    
    # Get the text column name from config or default to common options
    text_column = config['dataset'].get('text_column')
    
    logger.info(f"Loading dataset for analysis: {dataset_name} (samples: {num_samples})")
    
    if num_samples > 0:
        dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    # Load tokenizer from fine-tuned model
    model_path = config['models']['finetuned']['path']
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure we have a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine column names for tokenization
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

def create_prompt_dataset(config, tokenizer, use_generated_answers=False):
    """Create dataset from prompts in the config."""
    prompts_file = config['general']['prompts_file']
    max_length = config['general']['max_length']
    
    # If using generated answers, load them instead
    if use_generated_answers:
        generated_answers_file = config['evaluation']['finetuned_answers_file']
        if not os.path.exists(generated_answers_file):
            raise FileNotFoundError(f"Generated answers file {generated_answers_file} not found")
        
        logger.info(f"Using generated answers from {generated_answers_file}")
        with open(generated_answers_file, 'r') as f:
            prompts_data = json.load(f)
    else:
        # Load prompts from file
        if not os.path.exists(prompts_file):
            raise FileNotFoundError(f"Prompts file {prompts_file} not found")
        
        logger.info(f"Using prompts from {prompts_file}")
        with open(prompts_file, 'r') as f:
            prompts_data = json.load(f)
    
    # Process prompts into a format for tokenization
    processed_prompts = []
    for item in prompts_data:
        # We'll use prompt + completion for proper evaluation
        full_text = item["prompt"] + item["completion"]
        processed_prompts.append({"text": full_text})
    
    # Create dataset and tokenize
    dataset = Dataset.from_list(processed_prompts)
    
    def tokenize_function(examples):
        results = tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=max_length
        )
        # Set up labels correctly
        results["labels"] = results["input_ids"].copy()
        results["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in label]
            for label in results["labels"]
        ]
        return results
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    tokenized_dataset = tokenized_dataset.with_format("torch")
    return tokenized_dataset

def compute_scores(config, use_generated_answers=False):
    """Compute influence scores for prompts."""
    # Extract configuration
    model_path = config['models']['finetuned']['path']
    
    # Determine which factors and scores to use
    if use_generated_answers:
        factors_name = config['factors']['name']
        scores_name = config['scores']['generated_name']
    else:
        factors_name = config['factors']['all_layers_name']
        scores_name = config['scores']['all_layers_name']
    
    # Get layer configuration
    layers_config = config['factors'].get('layers', {'mode': 'all'})
    layer_mode = layers_config.get('mode', 'all')
    
    # Build output directory paths
    factors_dir = os.path.join(
        config['output']['influence_results'], 
        config['factors'].get('output_dir', 'factors')
    )
    scores_dir = os.path.join(
        config['output']['influence_results'], 
        config['scores'].get('output_dir', 'scores')
    )
    os.makedirs(scores_dir, exist_ok=True)
    
    query_gradient_rank = config['scores']['query_gradient_rank']
    train_batch_size = config['scores']['train_batch_size']
    use_margin = False  # Default to standard cross-entropy
    
    # Start timing
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(f"Starting score computation at {start_datetime}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Factors: {factors_name} (in {factors_dir})")
    logger.info(f"Scores: {scores_name} (will save to {scores_dir})")
    logger.info(f"Layer mode: {layer_mode}")
    logger.info(f"Using generated answers: {use_generated_answers}")
    
    # Load the model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_flash_attention_2=False  # Disable flash attention for compatibility with Kronfluence
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare datasets
    train_dataset, _ = get_dataset(config)
    
    query_dataset = create_prompt_dataset(
        config, 
        tokenizer, 
        use_generated_answers
    )
    
    # Define task
    if use_margin:
        logger.info("Using margin-based measurement task")
        task = LanguageModelingWithMarginMeasurementTask()
    else:
        logger.info("Using standard language modeling task")
        task = LanguageModelingTask()
    
    # Prepare model
    model = prepare_model(model, task)
    
    # Create analyzer
    analyzer = Analyzer(
        analysis_name=scores_dir,
        model=model,
        task=task,
        profile=False,
    )
    
    # Configure data loader
    dataloader_kwargs = DataLoaderKwargs(
        num_workers=config['factors']['num_workers'],
        collate_fn=default_data_collator,
        pin_memory=True
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)
    
    # Apply layer selection based on config
    if layer_mode == 'specific':
        specific_layers = layers_config.get('specific', [0, 6, 11])
        logger.info(f"Computing scores for specific layers: {specific_layers}")
        analyzer.set_modules([f"model.layers.{i}" for i in specific_layers])
    elif layer_mode == 'range':
        start = layers_config.get('range', {}).get('start', 0)
        end = layers_config.get('range', {}).get('end', 11)
        step = layers_config.get('range', {}).get('step', 1)
        layer_range = list(range(start, end + 1, step))
        logger.info(f"Computing scores for layers in range: {start}-{end} (step={step})")
        analyzer.set_modules([f"model.layers.{i}" for i in layer_range])
    else:
        # All layers (default)
        logger.info("Computing scores for all layers")
    
    # Configure score arguments
    rank = query_gradient_rank if query_gradient_rank != -1 else None
    score_args = extreme_reduce_memory_score_arguments(
        damping_factor=None,
        module_partitions=1,
        query_gradient_low_rank=rank,
        dtype=torch.bfloat16
    )
    
    # Refine score arguments
    score_args.query_gradient_accumulation_steps = 10
    score_args.use_full_svd = True
    score_args.precondition_dtype = torch.bfloat16
    score_args.per_sample_gradient_dtype = torch.bfloat16
    
    logger.info(f"Computing influence scores using factors: {factors_name}")
    
    # Compute pairwise influence scores
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        score_args=score_args,
        factors_name=factors_name,
        query_dataset=query_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=1,
        per_device_train_batch_size=train_batch_size,
        overwrite_output_dir=True,
    )
    
    # Load and log scores information
    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    logger.info(f"Scores shape: {scores.shape}")
    logger.info(f"Influence scores computed and saved with name: {scores_name}")
    
    # Log runtime information
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(f"Started at: {start_datetime}")
    logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return scores_name 