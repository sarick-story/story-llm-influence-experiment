import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import logging
import os
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the TinyLlama 1.1B model
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
logger.info(f"Loading model: {model_name}")

# Load model with bfloat16 precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

logger.info("Model loaded with bfloat16 precision")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Check if we have a cached dataset
cache_file = "./cached_tokenized_dataset_tinyllama.pkl"
if os.path.exists(cache_file):
    logger.info("Loading tokenized dataset from cache...")
    with open(cache_file, "rb") as f:
        tokenized_dataset = pickle.load(f)
else:
    # Load a small dataset (subset of OpenWebText)
    num_samples = 10000
    logger.info("Loading dataset from scratch")
    dataset = load_dataset("Elriggs/openwebtext-100k", split=f"train[:{num_samples}]")

    # Tokenize dataset
    def tokenize_function(examples):
        outputs = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
        # Set labels equal to input_ids for causal language modeling
        outputs["labels"] = outputs["input_ids"].copy()
        # Replace padding token IDs with -100 in labels
        outputs["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in label] 
            for label in outputs["labels"]
        ]
        return outputs

    logger.info("Tokenizing dataset")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.with_format("torch")
    
    # Cache the tokenized dataset
    logger.info("Caching tokenized dataset")
    with open(cache_file, "wb") as f:
        pickle.dump(tokenized_dataset, f)

# Setup training arguments - adjusted for full precision
output_dir = "./tinyllama_1b_model"
logger.info(f"Setting up training with output directory: {output_dir}")

# Determine mixed-precision settings
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=32,  # Increased from 8 to better utilize GPU memory
    save_steps=100,
    logging_steps=20,
    learning_rate=2e-5,  # Slightly increased learning rate
    weight_decay=0.01,
    logging_dir="./logs",
    lr_scheduler_type="cosine_with_restarts",
    gradient_accumulation_steps=1,  # Reduced from 4 since we have plenty of memory
    max_grad_norm=1.0,
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

# We can skip enabling gradient checkpointing since we have adequate memory
# model.gradient_checkpointing_enable()  # Commented out

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train model
logger.info("Starting training")
trainer.train()

# Save model
logger.info(f"Saving model to {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

logger.info("Training complete!") 