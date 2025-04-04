"""
Model Training Module

This module handles training the language model using the dataset and parameters
specified in the configuration.
"""

import os
import torch
import wandb
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
from modules.utils.wandb_utils import init_wandb

logger = logging.getLogger(__name__)

def get_dataset(config):
    """Load and prepare the dataset for training."""
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    num_samples = dataset_config['num_samples']
    max_length = config['general']['max_length']
    dataset_format = dataset_config.get('format', 'text') # Default to 'text' if not specified
    
    logger.info(f"Loading dataset: {dataset_name} (samples: {num_samples}, format: {dataset_format})")
    
    if num_samples > 0:
        dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    # Get the tokenizer from the base model
    tokenizer = AutoTokenizer.from_pretrained(config['models']['base']['name'])
    
    # Check if we need to set a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    column_names = dataset.column_names
    
    # Define the tokenization function based on dataset structure
    def tokenize_function(examples):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        if dataset_format == 'qa':
            input_col = dataset_config.get('input_column', 'Input')
            output_col = dataset_config.get('output_column', 'Output')
            input_label = dataset_config.get('input_label', 'Input') # Using default if not in config
            output_label = dataset_config.get('output_label', 'Output') # Using default if not in config

            if input_col not in column_names or output_col not in column_names:
                raise ValueError(f"QA format specified, but columns '{input_col}' or '{output_col}' not found.")

            # Construct the prompt marker sequence that ends the input part
            # Ensure there's a space before the output label if it doesn't start with one
            prompt_marker = f" {output_label}: " if not output_label.startswith(' ') else f"{output_label}: "
            logger.info(f"Using prompt marker for label masking: '{prompt_marker}'")
            # Tokenize the marker sequence *without* adding special tokens
            marker_tokens = tokenizer(prompt_marker, add_special_tokens=False)["input_ids"]
            marker_len = len(marker_tokens)
            if marker_len == 0:
                 logger.warning(f"Prompt marker '{prompt_marker}' tokenized to empty sequence. Check marker/tokenizer.")

            # Construct proper prompts and full texts
            # FIXED: Use a format that doesn't include output_label in the prompt to avoid confusion
            prompts = [f"{inp}" for inp in examples[input_col]]
            outputs = [f"{out}{tokenizer.eos_token}" for out in examples[output_col]]
            full_texts = [p + o for p, o in zip(prompts, outputs)]

            # Tokenize full texts with padding and truncation
            full_tokenized = tokenizer(
                full_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt" # Return PyTorch tensors temporarily for easier manipulation
            )
            # Convert back to lists for processing
            input_ids_list = full_tokenized['input_ids'].tolist()
            attention_mask_list = full_tokenized['attention_mask'].tolist()

            # --- Robust Label Masking --- 
            for i in range(len(input_ids_list)):
                current_input_ids = input_ids_list[i]
                labels = current_input_ids.copy()
                
                # FIXED: Rather than looking for a marker that may not exist in the tokenized form,
                # we'll tokenize the prompt separately and use its length
                prompt_tokenized = tokenizer(
                    prompts[i], 
                    add_special_tokens=True, 
                    return_tensors="pt"
                )
                prompt_length = prompt_tokenized['input_ids'].shape[1]
                
                # Mask everything up to the end of the prompt (everything up to prompt_length)
                # This will train the model to predict only the completion, not repeat the prompt
                if prompt_length > 0:
                    labels[:prompt_length-1] = [-100] * (prompt_length-1)
                    
                    # Log some examples of the label masking for debugging
                    if i < 3:
                        logger.info(f"Example {i}:")
                        logger.info(f"  Prompt: {prompts[i]}")
                        logger.info(f"  Output: {outputs[i]}")
                        logger.info(f"  Prompt length (tokens): {prompt_length}")
                        logger.info(f"  Masked labels: {labels[:prompt_length+10]}...")
                else:
                    logger.warning(f"Zero prompt length for example {i}. Check tokenization.")
                    
                # Also mask padding tokens using the attention mask (essential)
                for j in range(len(labels)):
                    if attention_mask_list[i][j] == 0:
                        labels[j] = -100
                
                labels_list.append(labels)

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
            
            # Tokenize with padding and truncation
            tokenized_outputs = tokenizer(
                processed_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            
            input_ids_list = tokenized_outputs['input_ids']
            attention_mask_list = tokenized_outputs['attention_mask']
            
            # Standard Causal LM: labels are input_ids, mask padding
            labels_list = [list(ids) for ids in input_ids_list] # Make copies
            for i in range(len(labels_list)):
                 for j in range(len(labels_list[i])):
                     if attention_mask_list[i][j] == 0:
                         labels_list[i][j] = -100

        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list
        }
    
    # Tokenize the dataset
    num_proc = min(4, os.cpu_count() // 2)  # Use half of available CPU cores, max 4
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=column_names, # Remove original columns after tokenization
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
    
    # Initialize wandb with a consistent naming scheme
    run = init_wandb(config, "train")
    
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
        # Use wandb for logging if configured
        report_to="wandb" if wandb.run is not None else None,
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
    
    # Log final training metrics to wandb
    if wandb.run is not None:
        wandb.log({
            "training_completed": True,
            "model_saved_path": model_output_path
        })
    
    return model_output_path 