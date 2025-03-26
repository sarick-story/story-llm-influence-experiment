import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
import logging
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

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
    dataset = load_dataset("openwebtext", split="train[:10000]")  # Just 10000 examples

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

# Split dataset into train and validation
logger.info("Creating train/validation split")
train_val_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_val_split["train"]
validation_dataset = train_val_split["test"]
logger.info(f"Training on {len(train_dataset)} examples, validating on {len(validation_dataset)} examples")

# Setup training arguments
output_dir = "./tiny_lm_model"
logger.info(f"Setting up training with output directory: {output_dir}")

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    evaluation_strategy="steps",
    eval_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=100,
    save_total_limit=2,  # Only keep the 2 best checkpoints
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    load_best_model_at_end=True,  # Load the best model when training ends
    metric_for_best_model="eval_loss",  # Use eval loss to determine the best model
    greater_is_better=False,  # Lower eval loss is better
    report_to="none"
)

# Initialize trainer with validation dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop if no improvement for 3 eval calls
)

# Train model
logger.info("Starting training")
train_output = trainer.train()

# Save model
logger.info(f"Saving model to {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Plotting loss curves
logger.info("Plotting training and validation loss curves...")
log_history = trainer.state.log_history
train_loss_values = []
train_steps = []
eval_loss_values = []
eval_steps = []

for log_entry in log_history:
    if 'loss' in log_entry:
        train_loss_values.append(log_entry['loss'])
        train_steps.append(log_entry.get('step', 0))
    elif 'eval_loss' in log_entry:
        eval_loss_values.append(log_entry['eval_loss'])
        eval_steps.append(log_entry.get('step', 0))

plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_loss_values, label='Training Loss', color='blue')
if eval_loss_values:  # Only plot if we have evaluation data
    plt.plot(eval_steps, eval_loss_values, label='Validation Loss', color='red')
plt.title('Training and Validation Loss Curves')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
logger.info(f"Loss curves plot saved to {output_dir}/loss_curves.png")
plt.close()

logger.info("Training complete!") 