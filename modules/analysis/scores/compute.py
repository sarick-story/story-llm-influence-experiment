"""
Influence Scores Computation Module

This module handles computing the influence scores for specific prompts using the trained model
and the precomputed influence factors.
"""

import json
import os
import time
import torch
import wandb
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset, Dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.score_arguments import extreme_reduce_memory_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
import logging
from modules.utils.wandb_utils import init_wandb

# Import custom task for language modeling - from factors module
from modules.analysis.factors.task import LanguageModelingTask, LanguageModelingWithMarginMeasurementTask

logger = logging.getLogger(__name__)

def get_dataset(config, dataset_type='main'):
    """Load and prepare dataset for score computation."""
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    # Use analysis_samples for influence score computation
    num_samples = dataset_config.get('analysis_samples', dataset_config['num_samples'])
    max_length = config['general']['max_length']
    dataset_format = dataset_config.get('format', 'text') # Default to 'text'
    
    logger.info(f"Loading dataset for score analysis: {dataset_name} (samples: {num_samples}, format: {dataset_format})")
    
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
    
    # Tokenization function adapted for different formats
    def tokenize_function(examples):
        processed_texts = []
        if dataset_format == 'qa':
            input_col = dataset_config.get('input_column', 'Input')
            output_col = dataset_config.get('output_column', 'Output')
            input_label = dataset_config.get('input_label', 'Input') # Using default if not in config
            output_label = dataset_config.get('output_label', 'Output') # Using default if not in config
            if input_col not in column_names or output_col not in column_names:
                raise ValueError(f"QA format specified, but columns '{input_col}' or '{output_col}' not found.")
            logger.debug(f"Processing QA format with columns: {input_col}, {output_col}")
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
            logger.debug(f"Processing text format with column: {text_col}")
            processed_texts = [text + tokenizer.eos_token for text in examples[text_col]]
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")
        
        # Tokenize
        results = tokenizer(
            processed_texts,
            truncation=True,
            padding="max_length", # Use max_length padding
            max_length=max_length,
            return_attention_mask=True, # Ensure attention mask is returned
        )
        
        # Set up labels correctly (for causal LM)
        results["labels"] = results["input_ids"].copy()
        # Mask padding tokens in labels
        labels_list = []
        for i in range(len(results["input_ids"])):
            label = results["input_ids"][i].copy()
            for j in range(len(label)):
                if results["attention_mask"][i][j] == 0:
                    label[j] = -100 # Mask padding tokens
            labels_list.append(label)
        results["labels"] = labels_list
        
        return results
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=min(4, os.cpu_count() // 2), # Use reasonable number of processes
        remove_columns=column_names,
        load_from_cache_file=True,
        desc=f"Tokenizing dataset ({dataset_format} format)"
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
    
    # Initialize wandb with a unique run name including generated flag if used
    run_prefix = "scores_gen" if use_generated_answers else "scores"
    run = init_wandb(config, run_prefix)
    
    # Determine which factors and scores to use
    if use_generated_answers:
        factors_name = config['factors']['all_layers_name']
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
    
    # Set performance options for better throughput (from original code)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    
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
    
    # MONKEY-PATCH the factors_output_dir method to point to the correct location
    import types
    from pathlib import Path
    
    original_factors_output_dir = analyzer.factors_output_dir
    
    def patched_factors_output_dir(self, factors_name: str) -> Path:
        # Construct the path to the saved factors using relative paths
        # This matches how we observed factors are stored from our inspection
        correct_path = Path(f"influence_results/results/influence/factors/factors_{factors_name}").resolve()
        self.logger.info(f"[Patched] Factors output dir pointing to: {correct_path}")
        
        if not correct_path.exists():
            # Try alternative locations if the primary path doesn't exist
            alt_paths = [
                Path(f"influence_results/factors_{factors_name}").resolve(),
                Path(os.path.join(factors_dir, f"factors_{factors_name}")).resolve(),
                Path(f"./influence_results/influence_results/factors_{factors_name}").resolve(),
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    self.logger.info(f"[Patched] Found factors at alternative location: {alt_path}")
                    return alt_path
                    
            self.logger.error(f"[Patched] Factors directory does not exist at: {correct_path}")
            self.logger.error(f"[Patched] Alternative paths checked: {alt_paths}")
            raise FileNotFoundError(f"[Patched] Factors directory not found: {correct_path}")
            
        # Check if the arguments file exists within this correct path
        args_file = correct_path / "factor_arguments.json"
        if not args_file.exists():
            self.logger.error(f"[Patched] Factor arguments file not found at: {args_file}")
            # Check for other files to help debug
            if os.path.exists(correct_path):
                self.logger.info(f"[Patched] Files in factors directory: {os.listdir(correct_path)}")
            raise FileNotFoundError(f"[Patched] Factor arguments file not found: {args_file}")
            
        return correct_path
    
    # Apply the monkey patch
    analyzer.factors_output_dir = types.MethodType(patched_factors_output_dir, analyzer)
    logger.info("Applied patch to analyzer.factors_output_dir")
    
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
    
    # Log score computation metrics to wandb
    if wandb.run is not None:
        wandb.log({
            'score_computation_time': time.time() - start_time,
            'num_queries': len(query_dataset),
            'num_train_samples': len(train_dataset),
            'use_generated_answers': use_generated_answers,
            'scores_name': scores_name,
            'factors_name': factors_name,
            'scores_shape': list(scores.shape),
            'layer_mode': layer_mode
        })
    
    return scores_name 