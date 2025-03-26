import argparse
import logging
from datetime import timedelta
import os
import sys

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from pathlib import Path
import numpy as np

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.utils.common.factor_arguments import extreme_reduce_memory_factor_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.task import Task

torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

# Monkey-patch the eigendecomposition function to add regularization
import kronfluence.factor.eigen
original_perform_eigendecomposition = kronfluence.factor.eigen.perform_eigendecomposition

@torch.no_grad()
def perform_eigendecomposition_with_reg(covariance_factors, model, state, factor_args, disable_tqdm=False):
    """Wrapper for perform_eigendecomposition that adds regularization to the covariance matrices"""
    logger = logging.getLogger(__name__)
    logger.info("Using regularized eigendecomposition")
    
    # Process covariance matrices with regularization
    for factor_name in ['activation_covariance', 'gradient_covariance']:
        for module_name in covariance_factors[factor_name]:
            # Add regularization to the diagonal before decomposition
            matrix = covariance_factors[factor_name][module_name]
            
            # Get the trace of the matrix (sum of diagonal elements)
            trace = torch.trace(matrix)
            
            # Add extremely strong regularization (10% of trace) - much more aggressive
            reg_value = trace * 0.1 / matrix.size(0)
            
            # For very ill-conditioned matrices, use even stronger regularization
            if matrix.size(0) > 1000:  # Large matrices need more regularization
                reg_value = trace * 0.2 / matrix.size(0)
                
            logger.info(f"Adding regularization {reg_value:.6f} to {module_name} {factor_name}")
            
            # Add to diagonal
            eye_matrix = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)
            covariance_factors[factor_name][module_name] = matrix + reg_value * eye_matrix
            
            # Force symmetry (sometimes numerical issues can cause tiny asymmetries)
            matrix = covariance_factors[factor_name][module_name]
            covariance_factors[factor_name][module_name] = 0.5 * (matrix + matrix.T)
    
    # Call the original function with our regularized matrices
    return original_perform_eigendecomposition(covariance_factors, model, state, factor_args, disable_tqdm)

# Replace the original function with our regularized version
kronfluence.factor.eigen.perform_eigendecomposition = perform_eigendecomposition_with_reg

# Define the Language Modeling Task
class LanguageModelingTask(Task):
    def __init__(self, tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer
        
    def compute_train_loss(
        self,
        batch,
        model,
        sample: bool = False,
    ) -> torch.Tensor:
        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits.float()
        
        # Shift logits and labels for next token prediction
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))
        labels = batch["input_ids"][..., 1:].contiguous()
        
        # Compute loss
        if not sample:
            summed_loss = torch.nn.functional.cross_entropy(
                logits, 
                labels.view(-1), 
                reduction="sum", 
                ignore_index=self.tokenizer.pad_token_id if self.tokenizer else -100
            )
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
                masks = labels.view(-1) == (self.tokenizer.pad_token_id if self.tokenizer else -100)
                sampled_labels[masks] = self.tokenizer.pad_token_id if self.tokenizer else -100
            summed_loss = torch.nn.functional.cross_entropy(
                logits, 
                sampled_labels, 
                ignore_index=self.tokenizer.pad_token_id if self.tokenizer else -100, 
                reduction="sum"
            )
        return summed_loss

    def compute_measurement(
        self,
        batch,
        model,
    ) -> torch.Tensor:
        # Similar to compute_train_loss but for evaluation
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits.float()
        
        shift_labels = batch["input_ids"][..., 1:].contiguous().view(-1)
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        
        return torch.nn.functional.cross_entropy(
            logits, 
            shift_labels, 
            ignore_index=self.tokenizer.pad_token_id if self.tokenizer else -100, 
            reduction="sum"
        )

    def get_influence_tracked_modules(self) -> list:
        # Track transformer blocks in the model
        total_modules = []
        
        # For GPT-Neo-125M, track the MLP blocks
        for i in range(12):  # 12 layers in gpt-neo-125m
            total_modules.append(f"transformer.h.{i}.mlp.c_fc")
            total_modules.append(f"transformer.h.{i}.mlp.c_proj")
            
        return total_modules

    def get_attention_mask(self, batch):
        return batch["attention_mask"]


def parse_args():
    parser = argparse.ArgumentParser(description="Influence factor computation for small language model.")

    parser.add_argument(
        "--factors_name",
        type=str,
        default="tiny_lm_factors",
        help="Name of the factor.",
    )
    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors. Options: ekfac, kfac, diagfisher",
    )
    parser.add_argument(
        "--factor_batch_size",
        type=int,
        default=4,
        help="Batch size for computing influence factors.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./tiny_lm_model",
        help="Path to the trained model.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )
    args = parser.parse_args()

    return args


def get_tokenized_dataset(tokenizer):
    # Load dataset
    dataset = load_dataset("openwebtext", split="train[:1000]")  # Use the same dataset as training
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.with_format("torch")
    
    return tokenized_dataset


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
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
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare the dataset
    train_dataset = get_tokenized_dataset(tokenizer)
    
    # Define task and prepare model
    task = LanguageModelingTask(tokenizer)
    model = prepare_model(model, task)
    
    # Set up accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))  # 1 hour timeout
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    model = accelerator.prepare_model(model)
    
    # Create the analyzer
    analyzer = Analyzer(
        analysis_name="tiny_lm_influence",
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
    
    # For smaller models, we can use simpler partitioning
    factor_args.covariance_module_partitions = 2
    factor_args.lambda_module_partitions = 2
    factor_args.covariance_data_partitions = 4
    factor_args.lambda_data_partitions = 4
    
    # Setting eigendecomposition to double precision for numerical stability
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


if __name__ == "__main__":
    main() 