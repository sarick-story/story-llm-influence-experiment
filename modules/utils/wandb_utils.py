"""
Weights & Biases (wandb) Utilities

This module provides utility functions for consistent wandb run naming and initialization
across different components of the LLM Influence Analysis Framework.
"""

import os
import wandb
from datetime import datetime

def generate_run_name(prefix, config=None):
    """Generate a unique run name for wandb.
    
    Args:
        prefix (str): Prefix for the run (e.g., 'train', 'factors', 'scores')
        config (dict, optional): Config dict to extract relevant info
    
    Returns:
        str: A unique run name
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Start with basic identifiers
    components = [prefix, timestamp]
    
    # Add relevant config information if available
    if config:
        # Add model identifier
        if 'models' in config:
            if 'id' in config['models'].get('finetuned', {}):
                components.append(config['models']['finetuned']['id'])
            elif 'id' in config['models'].get('base', {}):
                components.append(config['models']['base']['id'])
        
        # Add dataset size
        if 'dataset' in config and 'num_samples' in config['dataset']:
            components.append(f"n{config['dataset']['num_samples']}")
            
        # Add specific component configurations
        if prefix == 'factors':
            components.append(f"s{config['factors']['strategy']}")
        elif prefix == 'scores':
            if config['scores'].get('query_gradient_rank'):
                components.append(f"r{config['scores']['query_gradient_rank']}")
    
    # Join all components with underscores
    return "_".join(components)

def init_wandb(config, prefix):
    """Initialize wandb with consistent configuration.
    
    Args:
        config (dict): Main configuration dictionary
        prefix (str): Component prefix for the run name
    
    Returns:
        wandb.run or None: The wandb run object or None if wandb is not configured
    """
    if 'wandb' not in config:
        return None
    
    if wandb.run is not None:
        return wandb.run
    
    wandb_config = config['wandb'].copy()
    
    # Set a unique run name if not explicitly provided
    if not wandb_config.get('name'):
        wandb_config['name'] = generate_run_name(prefix, config)
    
    # Initialize wandb with this config
    run = wandb.init(
        entity=wandb_config.get('entity'),
        project=wandb_config.get('project'),
        name=wandb_config.get('name'),
        tags=wandb_config.get('tags'),
        config={
            'model': {
                'base': config['models']['base']['name'],
                'finetuned': config['models']['finetuned']['path'],
                'base_id': config['models']['base']['id'],
                'finetuned_id': config['models']['finetuned']['id']
            },
            'dataset': {
                'name': config['dataset']['name'],
                'num_samples': config['dataset']['num_samples'],
                'analysis_samples': config['dataset'].get('analysis_samples')
            },
            'component': prefix
        }
    )
    
    return run 