"""
OLMES Evaluation Runner Module

This module runs OLMES standardized benchmarks on both the base and fine-tuned models.
"""

import os
import sys
import subprocess
import tempfile
import yaml
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def create_olmes_config(config):
    """Create a configuration file for OLMES evaluation."""
    # Extract model information
    base_model_name = config['models']['base']['name']
    finetuned_model_path = config['models']['finetuned']['path']
    base_model_id = config['models']['base']['id']
    finetuned_model_id = config['models']['finetuned']['id']
    
    # Get OLMES tasks from configuration
    olmes_tasks = config['evaluation']['olmes']['tasks']
    limit_examples = config['evaluation']['olmes'].get('limit_examples', 50)
    
    # Create OLMES configuration
    olmes_config = {
        'tasks': olmes_tasks,
        'models': [
            {
                'model_id': base_model_id,
                'name': 'Base TinyLlama',
                'source': base_model_name,
                'module': 'transformers'
            },
            {
                'model_id': finetuned_model_id,
                'name': 'Fine-tuned TinyLlama',
                'source': finetuned_model_path,
                'module': 'transformers'
            }
        ],
        'limit_examples': limit_examples
    }
    
    # Create temporary file for the configuration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(olmes_config, f, default_flow_style=False)
        config_path = f.name
    
    logger.info(f"Created OLMES configuration file: {config_path}")
    return config_path

def run_olmes_command(config_path, output_dir):
    """Run the OLMES evaluation command."""
    cmd = ['oe_eval', 'run', config_path, '--output-dir', output_dir]
    
    logger.info(f"Running OLMES command: {' '.join(cmd)}")
    
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("OLMES evaluation completed successfully")
        logger.debug(f"OLMES output: {result.stdout}")
        
        if result.stderr:
            logger.warning(f"OLMES warnings/errors: {result.stderr}")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"OLMES evaluation failed: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False

def find_latest_run_dir(output_dir):
    """Find the latest run directory in the OLMES output directory."""
    run_dirs = [d for d in Path(output_dir).iterdir() if d.is_dir() and d.name.startswith('run_')]
    
    if not run_dirs:
        logger.error(f"No run directories found in {output_dir}")
        return None
    
    # Sort by modification time, newest first
    latest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
    logger.info(f"Found latest run directory: {latest_run}")
    
    return latest_run

def check_olmes_installed():
    """Check if OLMES is installed."""
    try:
        subprocess.run(['oe_eval', '--help'], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def run_olmes_evaluation(config):
    """Run OLMES evaluation on the models."""
    output_dir = config['output']['olmes_results']
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting OLMES evaluation")
    
    # Check if OLMES is installed
    if not check_olmes_installed():
        # Try to install OLMES
        try:
            logger.info("OLMES not found, installing...")
            olmes_dir = config['evaluation']['olmes']['dir']
            
            if not os.path.exists(olmes_dir):
                logger.error(f"OLMES directory not found: {olmes_dir}")
                logger.error("Please clone the OLMES repository or update the configuration")
                raise FileNotFoundError(f"OLMES directory not found: {olmes_dir}")
            
            install_cmd = ['pip', 'install', '-e', olmes_dir]
            logger.info(f"Running: {' '.join(install_cmd)}")
            subprocess.run(install_cmd, check=True)
            logger.info("OLMES installed successfully")
        except Exception as e:
            logger.error(f"Failed to install OLMES: {e}")
            logger.error("Please install OLMES manually and try again")
            raise
    
    # Create OLMES configuration file
    config_path = create_olmes_config(config)
    
    try:
        # Run OLMES evaluation
        success = run_olmes_command(config_path, output_dir)
        
        if not success:
            logger.error("OLMES evaluation failed")
            return None
        
        # Find the latest run directory
        run_dir = find_latest_run_dir(output_dir)
        
        if not run_dir:
            logger.error("Could not find OLMES run directory")
            return None
        
        # Find the task_model_scores.json file
        scores_file = run_dir / 'task_model_scores.json'
        
        if not scores_file.exists():
            logger.error(f"Could not find task_model_scores.json in {run_dir}")
            return None
        
        logger.info(f"OLMES evaluation completed, results in {run_dir}")
        
        return run_dir
    finally:
        # Clean up the temporary configuration file
        try:
            os.unlink(config_path)
        except:
            pass 