"""
Model Answer Generation Module

This module generates answers from both the base and fine-tuned models for the prompts
in the configuration.
"""

import os
import json
import torch
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

def load_prompts(config):
    """Load prompts from the configuration file."""
    prompts_file = config['general']['prompts_file']
    
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file {prompts_file} not found")
    
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    return prompts

def generate_completion(model, tokenizer, prompt, max_length=100):
    """Generate a completion for a prompt using the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate output (disable grad for efficiency)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,  # Enable sampling for more diverse outputs
            temperature=0.8,  # Add some randomness (but not too much)
            top_p=0.95,      # Nucleus sampling
            repetition_penalty=1.2,  # Add repetition penalty to reduce loops
            no_repeat_ngram_size=3,  # Prevent repeating of 3-grams
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract just the completion (remove the prompt)
    completion = generated_text[len(prompt):]
    
    return completion.strip()

def generate_model_answers(config):
    """Generate answers from both base and fine-tuned models for evaluation."""
    base_model_name = config['models']['base']['name']
    finetuned_model_path = config['models']['finetuned']['path']
    output_file = config['evaluation']['generated_answers_file']
    finetuned_output_file = config['evaluation']['finetuned_answers_file']
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(finetuned_output_file), exist_ok=True)
    
    logger.info(f"Generating answers from base model ({base_model_name}) and fine-tuned model ({finetuned_model_path})")
    logger.info(f"Will save output to: {output_file} and {finetuned_output_file}")
    
    # Load prompts
    prompts = load_prompts(config)
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Load base model
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load fine-tuned model
    logger.info(f"Loading fine-tuned model: {finetuned_model_path}")
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        finetuned_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    
    # Generate answers
    results = []
    
    logger.info("Generating answers...")
    for prompt_data in tqdm(prompts, desc="Generating answers"):
        prompt = prompt_data["prompt"]
        expected_completion = prompt_data["completion"].strip()
        
        # Generate from base model
        base_completion = generate_completion(base_model, base_tokenizer, prompt)
        
        # Generate from fine-tuned model
        finetuned_completion = generate_completion(finetuned_model, finetuned_tokenizer, prompt)
        
        # Store results
        results.append({
            "prompt": prompt,
            "expected_completion": expected_completion,
            "base_completion": base_completion,
            "finetuned_completion": finetuned_completion
        })
    
    # Save all results to the output file
    logger.info(f"Saving generated answers to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create a separate file with just the fine-tuned model's prompts + completions (for influence analysis)
    finetuned_results = []
    for item in results:
        finetuned_results.append({
            "prompt": item["prompt"],
            "completion": item["finetuned_completion"]
        })
    
    logger.info(f"Saving fine-tuned model answers to {finetuned_output_file}")
    with open(finetuned_output_file, 'w') as f:
        json.dump(finetuned_results, f, indent=2)
    
    logger.info("Answer generation complete")
    
    return output_file, finetuned_output_file 