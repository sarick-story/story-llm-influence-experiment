import argparse
import logging
import json
import os
import time
from datetime import timedelta, datetime

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset, Dataset

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.score_arguments import extreme_reduce_memory_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs

# Import our custom task from task.py
from task import LanguageModelingTask, LanguageModelingWithMarginMeasurementTask

torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Influence score computation for language model.")

    parser.add_argument(
        "--factors_name",
        type=str,
        default="olmoe_1b_factors",
        help="Name of the factors to use.",
    )
    parser.add_argument(
        "--scores_name",
        type=str,
        default="olmoe_prompt_scores",
        help="Name to save the scores under.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./olmoe_1b_model",
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
        default=1,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="prompts.json",
        help="JSON file with prompts to compute influence for.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Elriggs/openwebtext-100k",
        help="Dataset name to use for training examples.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of training samples to use.",
    )
    parser.add_argument(
        "--use_margin_for_measurement",
        action="store_true",
        default=False,
        help="Use margin for measurement instead of cross-entropy.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )
    return parser.parse_args()


def get_tokenized_dataset(tokenizer, dataset_name, max_length, num_samples):
    """Get a tokenized dataset for training examples."""
    if num_samples > 0:
        dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    column_names = dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    
    # Tokenize function that matches the original implementation
    def tokenize_function(examples):
        results = tokenizer(
            examples[text_column_name],
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
    
    # Map the tokenization function
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    
    tokenized_dataset = tokenized_dataset.with_format("torch")
    return tokenized_dataset


def create_prompt_dataset(tokenizer, prompts_file, max_length):
    """Create a dataset from prompts in a JSON file."""
    # Load prompts from file
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file {prompts_file} not found")
    
    with open(prompts_file, 'r') as f:
        prompts_data = json.load(f)
    
    # Process prompts into a format for tokenization
    processed_prompts = []
    for item in prompts_data:
        # We'll use prompt + completion for proper evaluation
        full_text = item["prompt"] + item["completion"]
        processed_prompts.append({"text": full_text})
    
    # Create a dataset from the processed prompts
    dataset = Dataset.from_list(processed_prompts)
    
    # Tokenize the dataset
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


def main():
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.info(f"Starting score computation at {start_datetime}")
    
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    
    logger.info(f"Loading model from {args.model_path}")
    
    # Use our fine-tuned model - this is important for influence analysis
    logger.info(f"Loading fine-tuned model from {args.model_path}")
    
    # For better debugging
    os.environ["TRANSFORMERS_VERBOSITY"] = "info"
    
    try:
        # Try to load the model with settings matching the example
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,  # Use bfloat16 as in the example
            device_map="auto",
            use_flash_attention_2=False  # Disable flash attention which can cause issues with tracking
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading fine-tuned model: {e}")
        logger.info("Trying to load the base model as fallback...")
        model = AutoModelForCausalLM.from_pretrained(
            "allenai/OLMoE-1B-7B-0125",
            torch_dtype=torch.bfloat16,  # Use bfloat16 consistently
            device_map="auto"
        )
    
    # Load the tokenizer from the fine-tuned model path
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125")
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad token to eos token")
    
    # Prepare datasets
    logger.info("Loading datasets")
    train_dataset = get_tokenized_dataset(
        tokenizer, 
        args.dataset_name, 
        args.max_length, 
        args.num_samples
    )
    
    query_dataset = create_prompt_dataset(
        tokenizer, 
        args.prompts_file, 
        args.max_length
    )
    
    # Define task
    if args.use_margin_for_measurement:
        logger.info("Using margin-based measurement task")
        task = LanguageModelingWithMarginMeasurementTask()
    else:
        logger.info("Using standard language modeling task")
        task = LanguageModelingTask()
    
    # Prepare model
    model = prepare_model(model, task)
    
    # Set up accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))  # 1 hour timeout
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    model = accelerator.prepare_model(model)
    
    # Create the analyzer
    analyzer = Analyzer(
        analysis_name=f"influence_results",
        model=model,
        task=task,
        profile=args.profile,
    )
    
    # Configure parameters for DataLoader
    dataloader_kwargs = DataLoaderKwargs(
        num_workers=4, 
        collate_fn=default_data_collator, 
        pin_memory=True
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)
    
    # Configure score arguments - match example settings
    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    score_args = extreme_reduce_memory_score_arguments(
        damping_factor=None, 
        module_partitions=1, 
        query_gradient_low_rank=rank, 
        dtype=torch.bfloat16  # Always use bfloat16 as in the example
    )
    
    # Match original implementation settings
    score_args.query_gradient_accumulation_steps = 10
    score_args.use_full_svd = True
    # Use bfloat16 for all operations to match example
    score_args.precondition_dtype = torch.bfloat16
    score_args.per_sample_gradient_dtype = torch.bfloat16
    
    logger.info(f"Computing influence scores using factors: {args.factors_name}")
    
    # MONKEY-PATCH the factors_output_dir method to point to the correct location
    import types
    from pathlib import Path
    
    original_factors_output_dir = analyzer.factors_output_dir
    
    def patched_factors_output_dir(self, factors_name: str) -> Path:
        # Construct the known correct path to the saved factors
        correct_path = Path(f"./influence_results/influence_results/factors_{factors_name}").resolve()
        self.logger.info(f"[Patched] Factors output dir pointing to: {correct_path}")
        if not correct_path.exists():
             self.logger.error(f"[Patched] Factors directory does not exist at: {correct_path}")
             raise FileNotFoundError(f"[Patched] Factors directory not found: {correct_path}")
        # Check if the arguments file exists within this correct path
        args_file = correct_path / "factor_arguments.json"  # Use the actual filename being produced
        if not args_file.exists():
             self.logger.error(f"[Patched] Factor arguments file not found at: {args_file}")
             raise FileNotFoundError(f"[Patched] Factor arguments file not found: {args_file}")
        return correct_path
    
    # Apply the monkey patch
    analyzer.factors_output_dir = types.MethodType(patched_factors_output_dir, analyzer)
    logger.info("Applied patch to analyzer.factors_output_dir")
    
    # Compute pairwise influence scores
    analyzer.compute_pairwise_scores(
        scores_name=args.scores_name,
        score_args=score_args,
        factors_name=args.factors_name,
        query_dataset=query_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=1,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=True,  # Don't overwrite by default
    )
    
    # Load and print scores information
    scores = analyzer.load_pairwise_scores(args.scores_name)["all_modules"]
    logger.info(f"Scores shape: {scores.shape}")
    logger.info(f"Influence scores computed and saved with name: {args.scores_name}")
    
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