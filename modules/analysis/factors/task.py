from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from kronfluence.task import Task

BATCH_TYPE = Dict[str, torch.Tensor]


class LanguageModelingTask(Task):
    def __init__(self, tokenizer=None, modules=None, layer_mode=None, layer_config=None, num_layers=None):
        """
        Initialize the Language Modeling Task.
        
        Args:
            tokenizer: The tokenizer to use
            modules: List of modules to track, if provided directly
            layer_mode: Mode for layer selection ('all', 'specific', 'range')
            layer_config: Configuration for layer selection
            num_layers: Total number of layers in the model
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.modules = modules
        self.layer_mode = layer_mode
        self.layer_config = layer_config
        self.num_layers = num_layers
    
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))
        labels = batch["labels"][..., 1:].contiguous()
        if not sample:
            summed_loss = F.cross_entropy(logits, labels.view(-1), reduction="sum", ignore_index=-100)
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
                masks = labels.view(-1) == -100
                sampled_labels[masks] = -100
            summed_loss = F.cross_entropy(logits, sampled_labels, ignore_index=-100, reduction="sum")
        return summed_loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        shift_labels = batch["labels"][..., 1:].contiguous().view(-1)
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        return F.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="sum")

    def get_influence_tracked_modules(self) -> List[str]:
        """Return a list of module names to track for influence analysis."""
        if self.modules is not None:
            # If modules were directly provided, use them
            layers_to_track = []
            for module in self.modules:
                # For each MLP module, track all its sub-components
                if ".mlp" in module:
                    layer_num = module.split(".")[2]
                    layers_to_track.append(f"model.layers.{layer_num}.mlp.gate_proj")
                    layers_to_track.append(f"model.layers.{layer_num}.mlp.up_proj")
                    layers_to_track.append(f"model.layers.{layer_num}.mlp.down_proj")
                else:
                    # If not an MLP module, track as-is
                    layers_to_track.append(module)
            return layers_to_track
        
        # If no modules provided, use layer_mode and configuration
        total_modules = []
        
        # Determine which layers to track based on layer_mode
        layers_to_track = []
        
        if self.layer_mode == "all" and self.num_layers:
            # Track all layers
            layers_to_track = list(range(self.num_layers))
        elif self.layer_mode == "specific" and self.layer_config:
            # Track specific layers
            layers_to_track = self.layer_config.get('specific', [0])
        elif self.layer_mode == "range" and self.layer_config:
            # Track a range of layers
            range_config = self.layer_config.get('range', {})
            start = range_config.get('start', 0)
            end = range_config.get('end', self.num_layers - 1 if self.num_layers else 21)
            step = range_config.get('step', 1)
            layers_to_track = list(range(start, end + 1, step))
        else:
            # Default to just tracking the last layer or a specific default layer
            default_layer = 21  # Default to layer 21 for TinyLlama
            if self.num_layers and self.num_layers > 0:
                default_layer = self.num_layers - 1
            layers_to_track = [default_layer]
        
        # For each layer to track, add all the MLP components
        for i in layers_to_track:
            total_modules.append(f"model.layers.{i}.mlp.gate_proj")
            total_modules.append(f"model.layers.{i}.mlp.up_proj")
            total_modules.append(f"model.layers.{i}.mlp.down_proj")
        
        return total_modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]


class LanguageModelingWithMarginMeasurementTask(LanguageModelingTask):
    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        labels = batch["labels"][..., 1:].contiguous().view(-1)
        masks = labels != -100
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))

        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins[masks].sum() 