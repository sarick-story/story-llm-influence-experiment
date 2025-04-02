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
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=model_output_path,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=500,
        fp16=True,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        seed=seed,
        report_to="none"  # Disable wandb or other reporting
    )
    
    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )
    
    logger.info("Starting model training...")
    trainer.train()
    
    # Save the model and tokenizer
    logger.info(f"Saving model to {model_output_path}")
    trainer.save_model(model_output_path)
    tokenizer.save_pretrained(model_output_path)
    
    return model_output_path 