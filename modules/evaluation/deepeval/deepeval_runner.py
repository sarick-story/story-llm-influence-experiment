"""
DeepEval Evaluation Runner Module

This module runs DeepEval benchmarks on both the base and fine-tuned models.
"""

import os
import json
import logging
from pathlib import Path
from tqdm import tqdm
import torch
import pandas as pd
from typing import List, Dict, Any
import importlib
import re

# Assuming transformers is installed
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import base class (we'll dynamically import specific benchmarks)
from deepeval.models.base_model import DeepEvalBaseLLM

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path_or_name, config):
    """Loads a Hugging Face model and tokenizer."""
    logger.info(f"Loading model and tokenizer from: {model_path_or_name}")
    
    # Get max memory from config - default to 40GiB if not specified
    max_memory = config['evaluation']['deepeval'].get('max_memory', "40GiB")
    logger.info(f"Using max memory: {max_memory}")
    
    # Load model with bfloat16 precision and auto device mapping
    # Add better memory optimization parameters
    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_name,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        max_memory={0: max_memory},  # Use value from config
        offload_folder="offload",  # Enable disk offloading for parts of the model if needed
        low_cpu_mem_usage=True,
        use_cache=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    
    # Set padding side to 'left' for decoder-only models (important for proper generation)
    tokenizer.padding_side = 'left'
    
    # Add padding token if missing (common for some models like Llama)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # Ensure model config also reflects this if necessary
        model.config.pad_token_id = tokenizer.pad_token_id 
        
    logger.info("Model and tokenizer loaded.")
    return model, tokenizer

class TransformerLLMWrapper(DeepEvalBaseLLM):
    """Wrapper class for Transformer models to be used with DeepEval benchmarks."""
    
    def __init__(self, model, tokenizer, model_name):
        self.model = model
        self.tokenizer = tokenizer
        self._model_name = model_name
        
        # Set max length for the model
        self.max_length = getattr(model.config, "max_position_embeddings", 2048)
        logger.info(f"Using max_length: {self.max_length} for model {model_name}")
        
        # Ensure the tokenizer has proper padding settings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            model.config.pad_token_id = self.tokenizer.pad_token_id
    
    def load_model(self):
        """Required implementation for DeepEvalBaseLLM."""
        return self.model
    
    def extract_final_answer(self, text):
        """
        Extract the MCQ answer (A, B, C, or D) from the generated text.
        For MMLU, we just need the letter answer.
        """
        # Look for final answer pattern
        if not text:
            return ""
            
        # Try to find "Answer: X" pattern first
        answer_matches = re.findall(r"(?i)answer:\s*([A-D])", text)
        if answer_matches:
            return answer_matches[-1].upper()
            
        # Try to find standalone A, B, C, D at the end of the text
        last_lines = text.strip().split('\n')[-5:]  # Check last 5 lines
        for line in reversed(last_lines):
            standalone = re.search(r"(?i)^[^\w]*([A-D])[^\w]*$", line.strip())
            if standalone:
                return standalone.group(1).upper()
                
        # If all else fails, look for any A, B, C, D in the last part of the text
        last_part = text[-100:] if len(text) > 100 else text
        choices = re.findall(r"(?i)\b([A-D])\b", last_part)
        if choices:
            return choices[-1].upper()
            
        return ""
    
    def generate(self, prompt: str) -> str:
        """Generate text based on the prompt."""
        model = self.load_model()
        
        # Check if this is an MMLU prompt (contains multiple choice)
        is_mmlu = any(x in prompt for x in ["A.", "B.", "C.", "D."])
        
        # For MMLU, add a clear instruction to output the letter only
        if is_mmlu and not prompt.endswith("Answer:"):
            prompt = prompt + "\n\nOutput 'A', 'B', 'C', or 'D'. Full answer not needed."
        
        # Tokenize the prompt with proper settings
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - 100,  # Leave room for generation
            padding="max_length",
            add_special_tokens=True
        ).to(model.device)
        
        # Ensure attention mask is set correctly
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        
        # Generate the output with optimized settings
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,  # Shorter for MMLU answers
                do_sample=False,     # Use deterministic decoding for benchmarks
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,      # Enable KV-cache for faster generation
            )
        
        # Decode the output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # For MMLU, extract just the answer letter
        if is_mmlu:
            answer = self.extract_final_answer(generated_text)
            logger.debug(f"Extracted MMLU answer: {answer} from output")
            if answer:
                return answer
        
        return generated_text
    
    async def a_generate(self, prompt: str) -> str:
        """Async implementation required by DeepEvalBaseLLM."""
        return self.generate(prompt)
    
    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Generate text for a batch of prompts."""
        model = self.load_model()
        
        # Check if these are MMLU prompts
        is_mmlu_batch = any(any(x in prompt for x in ["A.", "B.", "C.", "D."]) for prompt in prompts)
        
        # For MMLU, add instructions to each prompt
        if is_mmlu_batch:
            processed_prompts = []
            for prompt in prompts:
                if not prompt.endswith("Answer:"):
                    prompt = prompt + "\n\nOutput 'A', 'B', 'C', or 'D'. Full answer not needed."
                processed_prompts.append(prompt)
            prompts = processed_prompts
        
        # Tokenize all prompts with optimized settings
        batch_inputs = self.tokenizer(
            prompts, 
            padding="max_length",
            truncation=True,
            max_length=self.max_length - 100,  # Leave room for generation
            return_tensors="pt",
            add_special_tokens=True
        ).to(model.device)
        
        # Ensure attention mask is set correctly
        if "attention_mask" not in batch_inputs:
            batch_inputs["attention_mask"] = torch.ones_like(batch_inputs["input_ids"])
        
        # Generate the outputs with optimized settings
        with torch.no_grad():
            batch_outputs = model.generate(
                batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                max_new_tokens=100,      # Shorter for MMLU answers
                do_sample=False,         # Use deterministic decoding for benchmarks
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,          # Enable KV-cache for faster generation
            )
        
        # Decode all outputs
        batch_texts = self.tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        
        # For MMLU, extract just the answer letters
        if is_mmlu_batch:
            batch_answers = [self.extract_final_answer(text) for text in batch_texts]
            # If we have valid answers, return them
            if any(batch_answers):
                return batch_answers
        
        return batch_texts
    
    def get_model_name(self):
        """Return the model name."""
        return self._model_name

def load_benchmark(benchmark_config):
    """
    Dynamically load a benchmark class based on configuration.
    
    Args:
        benchmark_config: A dictionary containing benchmark configuration:
            - name: The benchmark class name (e.g., "MMLU", "HellaSwag")
            - n_shots: Number of shots for few-shot learning
            - Other parameters specific to the benchmark
            
    Returns:
        An initialized benchmark instance
    """
    benchmark_name = benchmark_config.get("name")
    if not benchmark_name:
        raise ValueError("Benchmark configuration must include a 'name' field.")
    
    # Import the benchmark class from deepeval.benchmarks
    try:
        # Get the module that contains the benchmark class
        module = importlib.import_module("deepeval.benchmarks")
        
        # Get the benchmark class
        benchmark_class = getattr(module, benchmark_name)
        
        # Create a copy of the config to remove the name (not a constructor parameter)
        params = benchmark_config.copy()
        params.pop("name")
        
        # Initialize the benchmark with the parameters
        benchmark = benchmark_class(**params)
        logger.info(f"Loaded benchmark: {benchmark_name} with parameters: {params}")
        
        return benchmark
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load benchmark '{benchmark_name}': {e}")
        raise

def run_benchmark(benchmark, model_wrapper, name, output_dir, config):
    """Run a benchmark and save the results."""
    logger.info(f"Running {benchmark.__class__.__name__} benchmark for {name}...")
    
    try:
        # Get batch size from config - default to 32 if not specified
        batch_size = config['evaluation']['deepeval'].get('batch_size', 32)
        logger.info(f"Using batch size: {batch_size}")
        
        # Evaluate the model using the benchmark with larger batch size for speed
        # Using a much larger batch size for efficiency on A100 GPU
        results = benchmark.evaluate(model=model_wrapper, batch_size=batch_size)
        
        # Log the results
        logger.info(f"{name} - {benchmark.__class__.__name__} overall score: {benchmark.overall_score}")
        
        # Create the output directory if it doesn't exist
        benchmark_dir = output_dir / benchmark.__class__.__name__.lower()
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the overall score
        with open(benchmark_dir / f"{name}_overall_score.json", "w") as f:
            json.dump({"overall_score": benchmark.overall_score}, f, indent=4)
        
        # Save the task scores if available
        if hasattr(benchmark, "task_scores") and benchmark.task_scores is not None:
            if isinstance(benchmark.task_scores, pd.DataFrame):
                benchmark.task_scores.to_csv(benchmark_dir / f"{name}_task_scores.csv", index=False)
            else:
                with open(benchmark_dir / f"{name}_task_scores.json", "w") as f:
                    json.dump(benchmark.task_scores, f, indent=4)
        
        # Save the predictions if available
        if hasattr(benchmark, "predictions") and benchmark.predictions is not None:
            if isinstance(benchmark.predictions, pd.DataFrame):
                benchmark.predictions.to_csv(benchmark_dir / f"{name}_predictions.csv", index=False)
            else:
                with open(benchmark_dir / f"{name}_predictions.json", "w") as f:
                    json.dump(benchmark.predictions, f, indent=4)
        
        logger.info(f"Saved {benchmark.__class__.__name__} results for {name} to {benchmark_dir}")
        return True, benchmark.overall_score
    
    except Exception as e:
        logger.error(f"Error running {benchmark.__class__.__name__} for {name}: {e}", exc_info=True)
        return False, None

def run_deepeval_evaluation(config):
    """Run DeepEval benchmarks on base and fine-tuned models."""
    logger.info("Starting DeepEval benchmark evaluation...")

    # Enable memory efficient settings
    torch.backends.cudnn.benchmark = True
    
    # Configure output directory
    output_dir = Path(config['evaluation']['deepeval']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model paths from config
    base_model_name = config['models']['base']['name']
    base_model_id = config['models']['base']['id']
    finetuned_model_path = config['models']['finetuned']['path']
    finetuned_model_id = config['models']['finetuned']['id']
    
    # Check if we should evaluate both models or just one
    # Default to evaluating both models
    eval_base_model = config['evaluation']['deepeval'].get('eval_base_model', True)
    eval_finetuned_model = config['evaluation']['deepeval'].get('eval_finetuned_model', True)
    
    # Initialize model wrappers
    base_model_wrapper = None
    ft_model_wrapper = None
    
    # Load only the models we need to evaluate
    try:
        if eval_base_model:
            logger.info(f"Loading base model: {base_model_name}")
            base_model, base_tokenizer = load_model_and_tokenizer(base_model_name, config)
            base_model_wrapper = TransformerLLMWrapper(base_model, base_tokenizer, base_model_id)
        
        if eval_finetuned_model:
            logger.info(f"Loading fine-tuned model: {finetuned_model_path}")
            ft_model, ft_tokenizer = load_model_and_tokenizer(finetuned_model_path, config)
            ft_model_wrapper = TransformerLLMWrapper(ft_model, ft_tokenizer, finetuned_model_id)
    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)
        return
    
    # Initialize benchmarks from config
    benchmarks = []
    benchmarks_config = config['evaluation']['deepeval'].get('benchmarks', [])
    if not benchmarks_config:
        logger.warning("No benchmarks specified in config. Using defaults: MMLU(n_shots=5, tasks=['PROFESSIONAL_LAW'])")
        # Fall back to defaults if no benchmarks are specified
        from deepeval.benchmarks import MMLU
        from deepeval.benchmarks.mmlu.task import MMLUTask
        benchmarks = [
            {"name": "MMLU", "instance": MMLU(n_shots=5, tasks=[MMLUTask.PROFESSIONAL_LAW])}
        ]
    else:
        # Load benchmarks from config
        for bench_config in benchmarks_config:
            try:
                # For MMLU benchmarks, convert task strings to MMLUTask enums
                if bench_config["name"] == "MMLU" and "tasks" in bench_config:
                    from deepeval.benchmarks.mmlu.task import MMLUTask
                    # Convert string task names to MMLUTask enums
                    task_strings = bench_config["tasks"]
                    task_enums = []
                    for task_str in task_strings:
                        task_enum = getattr(MMLUTask, task_str)
                        task_enums.append(task_enum)
                    
                    # Replace string task list with enum task list
                    bench_config["tasks"] = task_enums
                
                benchmark = load_benchmark(bench_config)
                benchmarks.append({"name": bench_config["name"], "instance": benchmark})
            except Exception as e:
                logger.error(f"Failed to load benchmark from config: {e}")
    
    if not benchmarks:
        logger.error("No valid benchmarks to run. Aborting evaluation.")
        return
    
    # Run the benchmarks and collect results
    results = {
        "base_model": {"name": base_model_id, "benchmarks": {}},
        "finetuned_model": {"name": finetuned_model_id, "benchmarks": {}}
    }
    
    # Count how many benchmarks we'll actually run
    total_benchmarks = 0
    if eval_base_model:
        total_benchmarks += len(benchmarks)
    if eval_finetuned_model:
        total_benchmarks += len(benchmarks)
    
    # Track and report progress
    completed = 0
    
    # Run each benchmark on selected models
    for bench in benchmarks:
        benchmark_name = bench["name"]
        benchmark_instance = bench["instance"]
        
        # Run on base model if requested
        if eval_base_model and base_model_wrapper:
            base_success, base_score = run_benchmark(benchmark_instance, base_model_wrapper, "base_model", output_dir, config)
            completed += 1
            logger.info(f"Progress: {completed}/{total_benchmarks} ({(completed/total_benchmarks)*100:.1f}%)")
            
            if base_success:
                results["base_model"]["benchmarks"][benchmark_name.lower()] = base_score
            
            # Free up memory after running benchmark
            torch.cuda.empty_cache()
        
        # Run on fine-tuned model if requested
        if eval_finetuned_model and ft_model_wrapper:
            ft_success, ft_score = run_benchmark(benchmark_instance, ft_model_wrapper, "finetuned_model", output_dir, config)
            completed += 1
            logger.info(f"Progress: {completed}/{total_benchmarks} ({(completed/total_benchmarks)*100:.1f}%)")
            
            if ft_success:
                results["finetuned_model"]["benchmarks"][benchmark_name.lower()] = ft_score
            
            # Free up memory after running benchmark
            torch.cuda.empty_cache()
    
    # Save the summary results
    with open(output_dir / "benchmark_summary.json", "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"DeepEval benchmark evaluation completed. Results saved to {output_dir}")
    
    # Clean up to release memory
    if 'base_model' in locals():
        del base_model
    if 'ft_model' in locals():
        del ft_model
    torch.cuda.empty_cache()
    
    # Return a summary of the results
    return results 