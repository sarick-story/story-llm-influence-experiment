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
    logger.info(f"--- Generating for prompt (first 50 chars): '{prompt[:50]}...' ---")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    logger.info(f"Input token length: {input_length}")
    
    # Generate completion
    raw_outputs = None
    generated_text_raw = "[ERROR DURING GENERATION]"
    generated_text_special = "[ERROR DURING GENERATION]"
    completion = "[ERROR DURING GENERATION]"
    
    try:
        with torch.no_grad():
            raw_outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,    # Use greedy decoding for deterministic output
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, # Explicitly set EOS token
                # Force the model to generate content even if it wants to output EOS early
                forced_decoder_ids=None,
                # Tell the model not to stop at EOS token during generation
                suppress_tokens=[tokenizer.eos_token_id]
            )
        
        output_length = raw_outputs.shape[1]
        logger.info(f"Raw output token length: {output_length}")
        logger.info(f"Raw output token IDs (last 10): {raw_outputs[0, -10:].tolist()}") # Log last 10 tokens

        # Decode WITHOUT skipping special tokens
        generated_text_special = tokenizer.decode(raw_outputs[0], skip_special_tokens=False)
        logger.info(f"Decoded text (with special tokens): '{generated_text_special}'")
        
        # Decode WITH skipping special tokens (standard way)
        generated_text_raw = tokenizer.decode(raw_outputs[0], skip_special_tokens=True)
        logger.info(f"Decoded text (no special tokens): '{generated_text_raw}'")
        
        # Extract just the completion (remove the original prompt)
        # Use raw decoded text for slicing, as special tokens might affect length
        prompt_decoded_raw = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        
        if generated_text_raw.startswith(prompt_decoded_raw):
             completion = generated_text_raw[len(prompt_decoded_raw):]
        else:
             # Fallback if prompt isn't exactly at the start (unlikely but possible)
             logger.warning("Generated text does not start with prompt text. Returning full generation minus prompt length guess.")
             # Attempt slicing based on input token length? More robust might be needed.
             # Let's try decoding only the generated part
             generated_ids = raw_outputs[0, input_length:]
             completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # If completion is still empty, try to force a minimal response
        if not completion.strip() and output_length > input_length:
            logger.warning("Empty completion detected despite successful generation. Using fallback method.")
            # Extract tokens after the prompt and before any EOS token
            generated_part = raw_outputs[0, input_length:]
            non_eos_tokens = [t for t in generated_part.tolist() if t != tokenizer.eos_token_id]
            logger.info(f"Non-EOS generated tokens: {non_eos_tokens}")
            
            if non_eos_tokens:
                completion = tokenizer.decode(non_eos_tokens, skip_special_tokens=True)
                logger.info(f"Fallback completion from non-EOS tokens: '{completion}'")
            else:
                # If still empty, return a placeholder
                completion = "The model did not generate a valid response."
                logger.warning("Model generated only EOS tokens or padding.")

        logger.info(f"Extracted completion: '{completion}'")

    except Exception as e:
         logger.error(f"Error during model generation or decoding: {e}", exc_info=True)
         # Fallback values defined above will be used
         if raw_outputs is not None:
             logger.error(f"Raw output tensor before error: {raw_outputs}")

    logger.info(f"--- Finished generating for prompt: '{prompt[:50]}...' ---")
    return completion

def main():
    args = parse_args()
    
    # Load prompts
    logger.info(f"Loading prompts from: {args.prompts_file}")
    try:
        with open(args.prompts_file, 'r') as f:
            prompts_data = json.load(f)
        logger.info(f"Successfully loaded {len(prompts_data)} prompts.")
    except FileNotFoundError:
        logger.error(f"Prompts file not found: {args.prompts_file}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from prompts file: {args.prompts_file}")
        return

    # Load base model
    logger.info("--- Loading Base Model ---")
    try:
        base_model, base_tokenizer = load_model_and_tokenizer(args.base_model_name)
        logger.info("Base model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load base model: {e}", exc_info=True)
        return # Cannot proceed without base model

    # Load fine-tuned model
    logger.info("--- Loading Fine-tuned Model ---")
    try:
        finetuned_model, finetuned_tokenizer = load_model_and_tokenizer(args.finetuned_model_path)
        # Quick check: Ensure tokenizers are compatible (vocab size)
        if base_tokenizer.vocab_size != finetuned_tokenizer.vocab_size:
             logger.warning("Base and fine-tuned tokenizer vocab sizes differ!")
        logger.info("Fine-tuned model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load fine-tuned model from {args.finetuned_model_path}: {e}", exc_info=True)
        # Decide if we should continue with only base model or stop
        logger.warning("Proceeding with generation for BASE MODEL ONLY.")
        finetuned_model, finetuned_tokenizer = None, None # Set to None to skip generation

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Generate answers for each prompt
    results = []
    for i, item in enumerate(prompts_data):
        prompt = item.get("prompt")
        # Ensure prompt is a string
        if not isinstance(prompt, str):
            logger.warning(f"Skipping item {i} due to non-string prompt: {prompt}")
            continue
            
        expected_completion = item.get("expected_completion", "") # Renamed from original_completion for clarity
        
        logger.info(f"--- Processing Prompt {i+1}/{len(prompts_data)} ---")
        
        # Generate answers using base model
        logger.info("Generating completion from BASE model...")
        base_completion = generate_answer(base_model, base_tokenizer, prompt, args.max_new_tokens)
        
        # Generate answers using fine-tuned model (if loaded)
        finetuned_completion = "[MODEL NOT LOADED]"
        if finetuned_model and finetuned_tokenizer:
             logger.info("Generating completion from FINE-TUNED model...")
             finetuned_completion = generate_answer(finetuned_model, finetuned_tokenizer, prompt, args.max_new_tokens)
        else:
             logger.warning("Skipping fine-tuned model generation as it failed to load.")
             
        # Add to results
        results.append({
            "prompt": prompt,
            "expected_completion": expected_completion,
            "base_completion": base_completion, # Renamed field
            "finetuned_completion": finetuned_completion # Renamed field
        })
    
    # Save the results
    try:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Generated answers saved to {args.output_file}")
    except IOError as e:
        logger.error(f"Failed to write output file {args.output_file}: {e}")

    # Also create a file with just the fine-tuned model completions 
    # Use the updated field name
    finetuned_prompts = []
    for item in results:
        finetuned_prompts.append({
            "prompt": item["prompt"],
            "completion": item["finetuned_completion"] 
        })
    
    # Construct fine-tuned output path relative to the main output file
    output_basename = os.path.basename(args.output_file)
    finetuned_output_filename = f"finetuned_{output_basename}"
    finetuned_output_path = os.path.join(output_dir, finetuned_output_filename)

    try:
        with open(finetuned_output_path, 'w') as f:
            json.dump(finetuned_prompts, f, indent=2)
        logger.info(f"Fine-tuned model prompts saved to {finetuned_output_path}")
    except IOError as e:
         logger.error(f"Failed to write fine-tuned output file {finetuned_output_path}: {e}")

if __name__ == "__main__":
    main() 