import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import logging
import os
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load a small model
model_name = "EleutherAI/gpt-neo-125m"
logger.info(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Check if we have a cached dataset
cache_file = "./cached_tokenized_dataset.pkl"
if os.path.exists(cache_file):
    logger.info("Loading tokenized dataset from cache...")
    with open(cache_file, "rb") as f:
        tokenized_dataset = pickle.load(f)
else:
    # Load a small dataset (subset of OpenWebText)
    logger.info("Loading dataset from scratch")
    dataset = load_dataset("openwebtext", split="train[:1000]")  # Just 1000 examples

    # Tokenize dataset
    def tokenize_function(examples):
        outputs = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
        # Set labels equal to input_ids for causal language modeling
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    logger.info("Tokenizing dataset")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.with_format("torch")
    
    # Cache the tokenized dataset
    logger.info("Caching tokenized dataset")
    with open(cache_file, "wb") as f:
        pickle.dump(tokenized_dataset, f)

# Setup training arguments
output_dir = "./tiny_lm_model"
logger.info(f"Setting up training with output directory: {output_dir}")

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=100,
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
)

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