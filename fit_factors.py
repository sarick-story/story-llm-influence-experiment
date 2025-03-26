import os
import sys
import pickle
import logging
from datetime import timedelta
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.factor_arguments import extreme_reduce_memory_factor_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.utils.constants import ACTIVATION_COVARIANCE_MATRIX_NAME, GRADIENT_COVARIANCE_MATRIX_NAME
from kronfluence.factor.eigen import perform_eigendecomposition

# Set some performance options
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set global tokenizer
tokenizer = None

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./tiny_lm_model")
    parser.add_argument("--factors_name", type=str, default="tiny_lm_factors")
    parser.add_argument("--factor_strategy", type=str, default="ekfac", choices=["ekfac", "kfac", "diagfisher"])
    parser.add_argument("--factor_batch_size", type=int, default=4)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cached_dataset_path", type=str, default="./cached_tokenized_dataset.pkl")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--covariance_data_partitions", type=int, default=8)
    parser.add_argument("--lambda_data_partitions", type=int, default=8)
    parser.add_argument("--eigendecomposition_dtype", type=str, default="float64", 
                        choices=["float32", "float64"])
    parser.add_argument("--regularization_factor", type=float, default=0.25, 
                        help="Fraction of trace to add to diagonal (0.0-1.0)")
    # Distributed training args
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    return parser.parse_args()

def get_tokenized_dataset(tokenizer, num_samples=10000):
    """Get the tokenized dataset from cache or create it"""
    cache_file = f"./cached_tokenized_dataset_{num_samples}.pkl"
    if os.path.exists(cache_file):
        logger.info(f"Loading tokenized dataset from cache ({num_samples} samples)")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    else:
        logger.info(f"Creating tokenized dataset from scratch ({num_samples} samples)")
        dataset = load_dataset("openwebtext", split=f"train[:{num_samples}]")
        
        def tokenize_function(examples):
            outputs = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
            outputs["labels"] = outputs["input_ids"].copy()
            return outputs

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_dataset = tokenized_dataset.with_format("torch")
        
        with open(cache_file, "wb") as f:
            pickle.dump(tokenized_dataset, f)
        
        return tokenized_dataset

def initialize_distributed(args):
    """Initialize distributed training if needed"""
    # Check if we're explicitly running in distributed mode
    if args.local_rank != -1 and int(os.environ.get("WORLD_SIZE", "0")) > 1:
        logger.info("Initializing distributed training")
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        logger.info(f"Initialized distributed training on rank {args.local_rank}")
    else:
        logger.info("Running in non-distributed mode")
        # For single GPU, set device to cuda:0 if available
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            logger.info("Using single GPU (cuda:0)")
        else:
            logger.info("Using CPU")

class LanguageModelingTask:
    """Temporary task class that will be replaced by imported task.py"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    # Placeholder methods that will be replaced

def add_regularization_to_covariance(covmat, reg_factor):
    """
    Add regularization to a covariance matrix to improve numerical stability.
    
    Args:
        covmat: The covariance matrix tensor
        reg_factor: Value between 0 and 1, representing the fraction of the trace to add to the diagonal
    
    Returns:
        Regularized covariance matrix
    """
    # Compute the trace
    trace = torch.trace(covmat)
    
    # Compute the diagonal increment
    avg_diag_value = trace / covmat.shape[0]
    increment = reg_factor * avg_diag_value
    
    # Add to diagonal (in-place operation)
    diag_indices = torch.arange(covmat.shape[0], device=covmat.device)
    covmat[diag_indices, diag_indices] += increment
    
    logger.info(f"Added regularization: {reg_factor:.2f} * trace/dim = {increment:.6f}")
    
    return covmat

def main():
    args = parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Determine if we're using distributed mode
    use_distributed = args.local_rank != -1 and int(os.environ.get("WORLD_SIZE", "0")) > 1
    
    # Start initializing distributed (or single device)
    initialize_distributed(args)

    # Set manual seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    
    # Handle local paths correctly
    model_path = args.model_path
    if model_path.startswith('./'):
        model_path = os.path.abspath(model_path)
    logger.info(f"Resolved model path: {model_path}")
    
    # Load the trained model
    global tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare the dataset
    train_dataset = get_tokenized_dataset(tokenizer, args.num_samples)
    
    # Load the custom task or fall back to default
    try:
        from task import LanguageModelingTask
        logger.info("Using custom LanguageModelingTask from task.py")
        task = LanguageModelingTask()
    except ImportError:
        logger.warning("task.py not found, using default LanguageModelingTask")
        task = LanguageModelingTask(tokenizer)
    
    # Define task and prepare model
    model = prepare_model(model, task)
    
    # Set up accelerator with different settings for distributed vs single-device
    if use_distributed:
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))  # 1 hour timeout
        accelerator = Accelerator(kwargs_handlers=[kwargs])
    else:
        # Simpler accelerator setup for single device
        accelerator = Accelerator()
    
    model = accelerator.prepare_model(model)
    
    # Create the analyzer
    analyzer = Analyzer(
        analysis_name=f"influence_results/{args.factors_name}",
        model=model,
        task=task,
        profile=args.profile,
    )
    
    # Configure parameters for DataLoader
    dataloader_kwargs = DataLoaderKwargs(
        num_workers=2, 
        collate_fn=default_data_collator, 
        pin_memory=True
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)
    
    # Configure factor arguments
    factor_args = extreme_reduce_memory_factor_arguments(
        strategy=args.factor_strategy, 
        module_partitions=1, 
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    # Apply the user's partitioning settings for improved numerical stability
    factor_args.covariance_module_partitions = 2
    factor_args.lambda_module_partitions = 2
    factor_args.covariance_data_partitions = args.covariance_data_partitions
    factor_args.lambda_data_partitions = args.lambda_data_partitions
    
    # Setting eigendecomposition to double precision for numerical stability
    if args.eigendecomposition_dtype == "float64":
        factor_args.eigendecomposition_dtype = torch.float64
    else:
        factor_args.eigendecomposition_dtype = torch.float32
    
    # Add custom regularization function to improve numerical stability
    if args.regularization_factor > 0:
        logger.info(f"Setting eigenvalue regularization factor: {args.regularization_factor}")
    
    # Custom approach to add regularization by manually adjusting matrices after they're computed
    # but before eigendecomposition
    if args.regularization_factor > 0:
        logger.info(f"Using covariance matrix regularization factor: {args.regularization_factor}")
        
        # We'll implement our regularization by using a custom function in the eigendecomposition step
        orig_perform_eigendecomposition = torch.no_grad()(perform_eigendecomposition)
        
        @torch.no_grad()
        def regularized_eigendecomposition(covariance_factors, model, state, factor_args, disable_tqdm=False):
            # Add regularization to covariance matrices before eigendecomposition
            for module_name in covariance_factors[ACTIVATION_COVARIANCE_MATRIX_NAME]:
                act_covmat = covariance_factors[ACTIVATION_COVARIANCE_MATRIX_NAME][module_name]
                grad_covmat = covariance_factors[GRADIENT_COVARIANCE_MATRIX_NAME][module_name]
                
                # Compute trace and add regularization
                act_covmat = add_regularization_to_covariance(act_covmat, args.regularization_factor)
                grad_covmat = add_regularization_to_covariance(grad_covmat, args.regularization_factor)
                
                # Update the covariance matrices
                covariance_factors[ACTIVATION_COVARIANCE_MATRIX_NAME][module_name] = act_covmat
                covariance_factors[GRADIENT_COVARIANCE_MATRIX_NAME][module_name] = grad_covmat
            
            # Continue with the original eigendecomposition implementation
            return orig_perform_eigendecomposition(covariance_factors, model, state, factor_args, disable_tqdm)
        
        # Replace the eigendecomposition function in the imported namespace
        sys.modules['kronfluence.factor.eigen'].perform_eigendecomposition = regularized_eigendecomposition
    
    logger.info(f"Computing influence factors with strategy: {args.factor_strategy}")
    logger.info(f"Using data partitions: covariance={args.covariance_data_partitions}, lambda={args.lambda_data_partitions}")
    logger.info(f"Using eigendecomposition dtype: {args.eigendecomposition_dtype}")
    
    # Fit the factors
    analyzer.fit_all_factors(
        factors_name=args.factors_name,
        dataset=train_dataset,
        per_device_batch_size=args.factor_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    
    logger.info(f"Influence factors computed and saved with name: {args.factors_name}")

if __name__ == "__main__":
    main() 