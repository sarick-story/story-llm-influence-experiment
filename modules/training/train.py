"""
Model Training Module

This module handles training the language model using the dataset and parameters
specified in the configuration.
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    default_data_collator
)
from datasets import load_dataset
import logging
import pickle
import yaml

logger = logging.getLogger(__name__)

def get_dataset(config):
    """Load and prepare the dataset for training."""
    dataset_name = config['dataset']['name']
    num_samples = config['dataset']['num_samples']
    max_length = config['general']['max_length']
    
    logger.info(f"Loading dataset: {dataset_name} (samples: {num_samples})")
    
    if num_samples > 0:
        dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    # Get the tokenizer from the base model
    tokenizer = AutoTokenizer.from_pretrained(config['models']['base']['name'])
    
    # Check if we need to set a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine the column names to use based on dataset structure
    column_names = dataset.column_names
    
    # Handle datasets with either 'text' or 'description' as the main content
    def tokenize_function(examples):
        # Try to find the appropriate text column
        text_field = None
        if 'text' in column_names:
            text_field = 'text'
        elif 'description' in column_names:
            text_field = 'description'
        else:
            text_field = column_names[0]  # Fall back to the first column
        
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

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_model(config):
    """Train the language model and save it to the specified path."""
    model_output_path = config['models']['finetuned']['path']
    base_model_name = config['models']['base']['name']
    use_flash_attention = config['general'].get('use_flash_attention', False)
    seed = config['general'].get('seed', 42)
    
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    
    logger.info(f"Loading base model: {base_model_name}")
    
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for training
        device_map="auto",
        use_flash_attention_2=use_flash_attention
    )
    
    # Get the tokenized dataset and tokenizer
    tokenized_dataset, tokenizer = get_dataset(config)
    
    # Setup training arguments - adjusted for full precision
    logger.info(f"Setting up training with output directory: {model_output_path}")

    # Determine mixed-precision settings
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    training_args = TrainingArguments(
        output_dir=model_output_path,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=100,
        logging_steps=20,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        lr_scheduler_type="cosine_with_restarts",
        gradient_accumulation_steps=1,
        # When using BFloat16, we need to disable gradient clipping to avoid the error
        max_grad_norm=None if use_bf16 else 1.0,
        # Use bfloat16 if available
        bf16=use_bf16,
        # Disable gradient checkpointing to speed up training since we have memory headroom
        gradient_checkpointing=False,
        # Optimize memory usage
        optim="adamw_torch_fused",
        # Data loading settings
        dataloader_num_workers=4,
        # Avoid certain warnings
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,
        remove_unused_columns=True,
    )
    
    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    logger.info("Starting model training...")
    trainer.train()
    
    # Save the model and tokenizer
    logger.info(f"Saving model to {model_output_path}")
    trainer.save_model(model_output_path)
    tokenizer.save_pretrained(model_output_path)
    
    return model_output_path 

if __name__ == "__main__":
    main() 