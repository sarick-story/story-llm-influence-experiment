import argparse
import logging
import json
import os
from datetime import timedelta

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset, Dataset

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.score_arguments import extreme_reduce_memory_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs

# Import our custom task from task.py instead of fit_factors.py
from task import LanguageModelingTask

torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Influence score computation for small language model.")

    parser.add_argument(
        "--factors_name",
        type=str,
        default="tiny_lm_factors",
        help="Name of the factors to use.",
    )
    parser.add_argument(
        "--scores_name",
        type=str,
        default="prompt_scores",
        help="Name to save the scores under.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./tiny_lm_model",
        help="Path to the trained model.",
    )
    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=64,
        help="Rank for the low-rank query gradient approximation. Use -1 for full rank.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="prompts.json",
        help="JSON file with prompts to compute influence for.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of training examples to use."
    )
    parser.add_argument(
        "--damping_factor",
        type=float,
        default=1e-5,
        help="Damping factor for numerical stability."
    )
    parser.add_argument(
        "--per_sample_gradient_dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32", "float64"],
        help="Precision for per-sample gradient computation."
    )
    parser.add_argument(
        "--precondition_dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32", "float64"],
        help="Precision for preconditioning operations."
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=5,
        help="Gradient accumulation steps for query gradients."
    )
    args = parser.parse_args()

    return args


def get_tokenized_dataset(tokenizer, num_samples=10000):
    """Get a tokenized dataset for training examples."""
    split = f"train[:{num_samples}]"
    logger.info(f"Loading dataset with {num_samples} examples")
    
    dataset = load_dataset("openwebtext", split=split)
    
    # Tokenize
    def tokenize_function(examples):
        outputs = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
        # Add labels for causal language modeling (same as input_ids)
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.with_format("torch")
    
    return tokenized_dataset


def create_prompt_dataset(tokenizer, prompts_file):
    """Create a dataset from prompts in a JSON file."""
    if not os.path.exists(prompts_file):
        # Create a default prompts file with a few examples
        prompts = [
            {"prompt": "What is inflation?", "completion": " Inflation is the rate at which prices increase over time."},
            {"prompt": "Who was the first president of the United States?", "completion": " George Washington was the first president."},
            {"prompt": "Explain quantum computing", "completion": " Quantum computing uses quantum bits to perform computations."}
        ]
        with open(prompts_file, 'w') as f:
            json.dump(prompts, f, indent=2)
        logger.info(f"Created default prompts file: {prompts_file}")
    
    # Load prompts from file
    with open(prompts_file, 'r') as f:
        prompts_data = json.load(f)
    
    # Process prompts into a dataset
    prompt_texts = []
    for item in prompts_data:
        # Combine prompt and completion for causal language modeling
        full_text = item["prompt"] + item["completion"]
        prompt_texts.append({"text": full_text})
    
    # Create a dataset from the prompts
    dataset = Dataset.from_list(prompt_texts)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        outputs = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
        # Add labels for causal language modeling (same as input_ids)
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.with_format("torch")
    
    return tokenized_dataset


def get_dtype_from_string(dtype_str):
    """Convert a string dtype to a torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    return dtype_map.get(dtype_str, torch.float32)


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    logger.info(f"Loading model from {args.model_path}")
    
    # Load the trained model
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare datasets
    logger.info("Loading datasets")
    train_dataset = get_tokenized_dataset(tokenizer, args.num_samples)
    query_dataset = create_prompt_dataset(tokenizer, args.prompts_file)
    
    # Define task and prepare model
    task = LanguageModelingTask()
    model = prepare_model(model, task)
    
    # Set up accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))  # 1 hour timeout
    accelerator = Accelerator(kwargs_handlers=[kwargs])
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
    
    # Configure score arguments
    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    
    # Convert string dtypes to torch dtypes
    per_sample_gradient_dtype = get_dtype_from_string(args.per_sample_gradient_dtype)
    precondition_dtype = get_dtype_from_string(args.precondition_dtype)
    
    # Log our configuration
    logger.info(f"Using damping factor: {args.damping_factor}")
    logger.info(f"Using per-sample gradient dtype: {args.per_sample_gradient_dtype}")
    logger.info(f"Using precondition dtype: {args.precondition_dtype}")
    logger.info(f"Using query gradient rank: {rank}")
    logger.info(f"Using accumulation steps: {args.accumulation_steps}")
    
    score_args = extreme_reduce_memory_score_arguments(
        damping_factor=args.damping_factor, 
        module_partitions=1, 
        query_gradient_low_rank=rank, 
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    # Apply stability improvements
    score_args.query_gradient_accumulation_steps = args.accumulation_steps
    score_args.use_full_svd = True  # More accurate but slower SVD
    score_args.precondition_dtype = precondition_dtype
    score_args.per_sample_gradient_dtype = per_sample_gradient_dtype
    
    logger.info(f"Computing influence scores using factors: {args.factors_name}")
    
    # Compute pairwise influence scores
    analyzer.compute_pairwise_scores(
        scores_name=args.scores_name,
        score_args=score_args,
        factors_name=args.factors_name,
        query_dataset=query_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=1,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=True,
    )
    
    # Load and print scores information
    scores = analyzer.load_pairwise_scores(args.scores_name)["all_modules"]
    logger.info(f"Scores shape: {scores.shape}")
    
    # Check for any numerical issues like NaN or inf
    num_inf = torch.isinf(scores).sum().item()
    num_nan = torch.isnan(scores).sum().item()
    if num_inf > 0 or num_nan > 0:
        logger.warning(f"Found {num_inf} inf values and {num_nan} NaN values in scores")
        
        # Suggest potential fixes
        if num_inf > 0 or num_nan > 0:
            logger.warning("Consider increasing the damping factor if numerical issues persist")
    
    logger.info(f"Influence scores computed and saved with name: {args.scores_name}")


if __name__ == "__main__":
    main() 