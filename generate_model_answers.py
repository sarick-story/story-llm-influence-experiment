import argparse
import json
import logging
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate answers from base and fine-tuned models.")
    
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        help="Name or path of the base model."
    )
    parser.add_argument(
        "--finetuned_model_path",
        type=str,
        default="./tinyllama_1b_model",
        help="Path to the fine-tuned model."
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="prompts.json",
        help="JSON file with prompts to generate answers for."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="generated_answers.json",
        help="File to save generated answers to."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate."
    )
    
    return parser.parse_args()

def load_model_and_tokenizer(model_path_or_name):
    """Load model and tokenizer from path or name."""
    logger.info(f"Loading model from {model_path_or_name}")
    
    # Load model with bfloat16 precision
    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

def generate_answer(model, tokenizer, prompt, max_new_tokens=50):
    """Generate an answer for a prompt using the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate completion
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,    # Use greedy decoding for deterministic output
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the completion (remove the original prompt)
    completion = generated_text[len(prompt):]
    
    return completion

def main():
    args = parse_args()
    
    # Load prompts
    with open(args.prompts_file, 'r') as f:
        prompts_data = json.load(f)
    
    # Load base model
    base_model, base_tokenizer = load_model_and_tokenizer(args.base_model_name)
    
    # Load fine-tuned model
    finetuned_model, finetuned_tokenizer = load_model_and_tokenizer(args.finetuned_model_path)
    
    # Generate answers for each prompt
    results = []
    for item in prompts_data:
        prompt = item["prompt"]
        original_completion = item.get("completion", "")
        
        logger.info(f"Generating answers for prompt: {prompt}")
        
        # Generate answers using both models
        base_completion = generate_answer(base_model, base_tokenizer, prompt, args.max_new_tokens)
        finetuned_completion = generate_answer(finetuned_model, finetuned_tokenizer, prompt, args.max_new_tokens)
        
        # Add to results
        results.append({
            "prompt": prompt,
            "original_completion": original_completion,
            "base_model_completion": base_completion,
            "finetuned_model_completion": finetuned_completion
        })
    
    # Save the results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Generated answers saved to {args.output_file}")
    
    # Also create a file with just the fine-tuned model completions in the original format
    finetuned_prompts = []
    for item in results:
        finetuned_prompts.append({
            "prompt": item["prompt"],
            "completion": item["finetuned_model_completion"]
        })
    
    finetuned_output_file = f"finetuned_{os.path.basename(args.output_file)}"
    with open(finetuned_output_file, 'w') as f:
        json.dump(finetuned_prompts, f, indent=2)
    
    logger.info(f"Fine-tuned model prompts saved to {finetuned_output_file}")

if __name__ == "__main__":
    main() 