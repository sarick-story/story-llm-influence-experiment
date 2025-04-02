import os
import sys
import pickle
import logging
import time
from datetime import timedelta, datetime
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader, DistributedSampler
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed, DeepSpeedPlugin
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.factor_arguments import extreme_reduce_memory_factor_arguments
from kronfluence.utils.dataset import DataLoaderKwargs

# Set aggressive performance options
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.allow_fp16_reduced_precision_reduction = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./olmoe_1b_model")
    parser.add_argument("--factors_name", type=str, default="olmoe_1b_factors")
    parser.add_argument("--factor_strategy", type=str, default="ekfac", choices=["ekfac", "kfac", "diagfisher"])
    parser.add_argument("--factor_batch_size", type=int, default=4)  # Increased default batch size
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="Elriggs/openwebtext-100k")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--use_flash_attention", action="store_true", default=False)
    parser.add_argument("--cpu_offload", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=8)  # Increased workers
    # Distributed training args
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()  

def get_tokenized_dataset(tokenizer, dataset_name, max_length, num_samples, seed=42):
    """Get the tokenized dataset with proper format matching original implementation"""
    logger.info(f"Loading dataset: {dataset_name}, samples: {num_samples}")
    
    # Load raw dataset with caching enabled
    if num_samples > 0:
        raw_datasets = load_dataset(dataset_name, split=f"train[:{num_samples}]", cache_dir="./dataset_cache")
    else:
        raw_datasets = load_dataset(dataset_name, split="train", cache_dir="./dataset_cache")
    
    # Shuffle dataset for better training
    raw_datasets = raw_datasets.shuffle(seed=seed)
    
    column_names = raw_datasets.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    # Tokenize function that matches the original implementation
    def tokenize_function(examples):
        results = tokenizer(
            examples[text_column_name], 
            truncation=True, 
            padding="max_length",  # Use max_length padding instead of just "True"
            max_length=max_length,
            return_tensors=None  # Return lists instead of tensors
        )
        # Set up labels correctly
        results["labels"] = results["input_ids"].copy()
        results["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in label] 
            for label in results["labels"]
        ]
        return results
    
    # Map the tokenization function with more processes
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=min(os.cpu_count(), 16),  # Utilize more cores but cap at 16
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    
    # Format correctly
    tokenized_datasets = tokenized_datasets.with_format("torch")
    
    return tokenized_datasets

def main():
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting factor computation at {start_datetime}")
    
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Handle distributed training setup
    env_local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if env_local_rank != -1:
        args.local_rank = env_local_rank
    
    # Set up distributed environment if needed
    if args.local_rank != -1:
        if not dist.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        logger.info(f"Initialized distributed training on rank {args.local_rank}")
    
    # Prepare model loading kwargs
    model_kwargs = {
        "device_map": "auto",
        # Remove bfloat16 to ensure gradient tracking works properly
        # "torch_dtype": torch.bfloat16,
    }
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        **model_kwargs
    )
    
    # Print model information for debugging
    logger.info(f"Model type: {type(model).__name__}")
    
    # Make sure the model parameters require gradients
    for name, param in model.named_parameters():
        param.requires_grad = True
        if "weight" in name:
            logger.info(f"Param {name} requires_grad set to {param.requires_grad}")
    
    # Set model to train mode to ensure gradients flow
    model.train()
    logger.info("Model set to train mode")
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Model doesn't support gradient checkpointing")
    
    # Use flash attention if available and requested
    if args.use_flash_attention:
        if hasattr(model, "config") and hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "flash_attention_2"
            logger.info("Flash Attention 2 enabled")
        else:
            logger.warning("Model doesn't support Flash Attention configuration")
    else:
        logger.info("Flash Attention disabled")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare the dataset
    train_dataset = get_tokenized_dataset(
        tokenizer, 
        args.dataset_name, 
        args.max_length, 
        args.num_samples,
        args.seed
    )
    
    # Import task
    try:
        from task import LanguageModelingTask
        logger.info("Using custom LanguageModelingTask from task.py")
        task = LanguageModelingTask()
    except ImportError:
        logger.error("Failed to import task.py. Please ensure it exists in the current directory.")
        sys.exit(1)
    
    # Prepare model for influence analysis
    model = prepare_model(model, task)
    
    # Set up accelerator with more optimizations
    accelerator_kwargs = {
        # Use fp32 precision instead of bf16 to ensure gradient flow
        # "mixed_precision": "bf16",  # Use bfloat16 mixed precision
        "mixed_precision": "no",  # Use full precision for gradient stability
    }
    
    # Configure DeepSpeed if CPU offloading is requested
    if args.cpu_offload:
        logger.info("CPU offloading enabled via DeepSpeed")
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=2,
            offload_optimizer=True,
            offload_param=False,
        )
        accelerator_kwargs["deepspeed_plugin"] = deepspeed_plugin
    
    # Add timeout for distributed setup
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # 2 hour timeout
    accelerator = Accelerator(kwargs_handlers=[kwargs], **accelerator_kwargs)
    model = accelerator.prepare_model(model)
    
    # Create the analyzer with correct analysis_name
    analyzer = Analyzer(
        analysis_name=f"influence_results",
        model=model,
        task=task,
        profile=args.profile,
    )
    
    # Configure parameters for DataLoader with better performance
    dataloader_kwargs = DataLoaderKwargs(
        num_workers=args.num_workers, 
        collate_fn=default_data_collator, 
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)
    
    # Configure factor arguments - use settings from the example
    factor_args = extreme_reduce_memory_factor_arguments(
        strategy=args.factor_strategy,
        module_partitions=1,  # Match example setting
        dtype=torch.bfloat16  # Use bfloat16 as in the example
    )
    
    # Match partitioning settings from the example
    factor_args.covariance_module_partitions = 2
    factor_args.lambda_module_partitions = 4
    factor_args.covariance_data_partitions = 4
    factor_args.lambda_data_partitions = 4
    
    # Set eigendecomposition to float64 for numerical stability
    factor_args.eigendecomposition_dtype = torch.float64
    
    logger.info(f"Computing influence factors with strategy: {args.factor_strategy}")
    
    # Fit the factors
    analyzer.fit_all_factors(
        factors_name=args.factors_name,
        dataset=train_dataset,
        per_device_batch_size=args.factor_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    
    logger.info(f"Influence factors computed and saved with name: {args.factors_name}")
    
    # Log runtime information
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logger.info(f"Started at: {start_datetime}")
    logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 