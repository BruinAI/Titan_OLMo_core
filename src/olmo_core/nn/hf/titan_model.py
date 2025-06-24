"""
Custom HuggingFace model for Titan OLMo with Neural Memory.
This preserves the custom architecture while making it compatible with HF ecosystem.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional, Union, Tuple, Dict, Any, List

from olmo_core.nn.transformer.model import Transformer
from olmo_core.nn.titans.neural_memory import NeuralMemory
from olmo_core.nn.transformer.config import TransformerConfig, TransformerBlockType, TransformerBlockConfig
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.memory_config import MemoryConfig
from olmo_core.config import DType


class TitanOLMoConfig(PretrainedConfig):
    """
    Configuration class for TitanOLMo model.
    
    This stores all parameters needed to reconstruct your Titan model from checkpoints,
    including the complete transformer config and memory settings.
    """
    model_type = "titan_olmo"
    
    def __init__(
        self,
        # Standard transformer config - based on actual checkpoint values
        vocab_size=100352,  # From your checkpoint: 100352 (padded from 100278)
        hidden_size=2048,   # d_model from your checkpoint
        intermediate_size=8192,  # feed_forward.hidden_size from your checkpoint
        num_hidden_layers=18,    # n_layers from your checkpoint  
        num_attention_heads=16,  # n_heads from your checkpoint
        max_position_embeddings=4096,  # Keep reasonable default
        rms_norm_eps=1e-6,      # eps from your checkpoint layer_norm config
        rope_theta=500000,      # theta from your checkpoint rope config
        pad_token_id=100277,    # From your checkpoint tokenizer
        bos_token_id=100257,    # From your checkpoint tokenizer (eos_token_id)
        eos_token_id=100257,    # From your checkpoint tokenizer
        # Titan-specific configuration
        memory_layers=None,     # List of layer indices that have memory [3,7,11,15]
        memory_config=None,     # MemoryConfig parameters
        use_sliding_window=True,
        window_size=512,        # window_size from your checkpoint
        # Store the complete transformer config for exact reconstruction
        complete_transformer_config=None,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        # Standard transformer parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        
        # Titan-specific parameters
        self.memory_layers = memory_layers or []
        self.memory_config = memory_config
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size
        
        # Store the complete transformer configuration for exact reconstruction
        self.complete_transformer_config = complete_transformer_config

    @classmethod
    def from_transformer_config(
        cls, 
        transformer_config: TransformerConfig, 
        memory_layers: Optional[List[int]] = None,
        **kwargs
    ) -> "TitanOLMoConfig":
        """
        Create TitanOLMoConfig from a complete TransformerConfig.
        
        This is the preferred way to create the config as it preserves
        all the exact settings from training.
        """
        # Extract basic parameters from transformer config
        config_dict = {
            "vocab_size": transformer_config.vocab_size,
            "hidden_size": transformer_config.d_model,
            "num_hidden_layers": transformer_config.n_layers,
            "memory_layers": memory_layers or [],
            "complete_transformer_config": transformer_config.as_dict(),
            **kwargs
        }
        
        # Extract attention config from the base block
        if hasattr(transformer_config.block, 'attention'):
            attention_config = transformer_config.block.attention
            config_dict.update({
                "num_attention_heads": attention_config.n_heads,
                "rope_theta": attention_config.rope.theta if attention_config.rope else 500000,
            })
        
        # Extract memory config from block overrides if present
        if transformer_config.block_overrides:
            for layer_idx, block_config in transformer_config.block_overrides.items():
                if block_config.memory_config:
                    # Convert MemoryConfig to dict format
                    memory_config_dict: Dict[str, Any] = {
                        "persistent_mem_len": block_config.memory_config.persistent_mem_len,
                        "window_size": block_config.memory_config.window_size,
                        "chunk_size": block_config.memory_config.chunk_size,
                        "n_layers": block_config.memory_config.n_layers,
                        "hidden_dim_multiple": block_config.memory_config.hidden_dim_multiple,
                    }
                        
                    config_dict["memory_config"] = memory_config_dict
                    break
        
        return cls(**config_dict)

class TitanOLMoForCausalLM(PreTrainedModel):
    """
    Custom HuggingFace model that wraps your Titan OLMo architecture.
    
    This allows you to use your custom model with HF ecosystem while preserving
    all the custom neural memory functionality.
    """
    config_class = TitanOLMoConfig
    
    def __init__(self, config: TitanOLMoConfig):
        super().__init__(config)
        
        # Store the original transformer config for building the model
        self.transformer_config = self._convert_hf_to_olmo_config(config)
        
        # Build the actual Titan model using your existing architecture
        self.model = self.transformer_config.build()
        
        # Initialize weights if needed
        self.post_init()
    
    def _convert_hf_to_olmo_config(self, hf_config: TitanOLMoConfig) -> TransformerConfig:
        """
        Convert HF config to OLMo Core TransformerConfig.
        
        This reconstructs the exact transformer configuration used during training,
        including all Titan-specific components.
        """
        if hf_config.complete_transformer_config is not None:
            # Use the stored complete config for exact reconstruction
            return TransformerConfig.from_dict(hf_config.complete_transformer_config)
        
        # Fallback: reconstruct from individual parameters
        # Build memory config if present
        memory_config = None
        if hf_config.memory_config is not None:
            if isinstance(hf_config.memory_config, dict):
                memory_config = MemoryConfig(**hf_config.memory_config)
            else:
                memory_config = hf_config.memory_config
        
        # Set up kwargs for transformer config
        kwargs = {}
        kwargs["block_name"] = TransformerBlockType.reordered_norm
        
        # Add memory block overrides for specific layers
        if hf_config.memory_layers and memory_config:
            block_overrides = {}
            for layer_idx in hf_config.memory_layers:
                # Create block configs exactly like the training script does
                # The system will automatically fill in None values during build
                try:
                    block_overrides[layer_idx] = TransformerBlockConfig(
                        name=TransformerBlockType.mag_reordered_norm,
                        attention=None,  # type: ignore # Will be filled by the config system
                        layer_norm=None,  # type: ignore # Will be filled by the config system
                        feed_forward=None,  # type: ignore # Will be filled by the config system
                        memory_config=memory_config
                    )
                except TypeError as e:
                    # Fallback: if the system doesn't accept None values, 
                    # warn and skip block overrides
                    print(f"Warning: Could not create block override for layer {layer_idx}: {e}")
                    print("This might be due to strict type checking. Using complete_transformer_config is recommended.")
                    block_overrides = {}
                    break
            
            if block_overrides:
                kwargs["block_overrides"] = block_overrides
        
        # Add sliding window attention if enabled
        if hf_config.use_sliding_window:
            kwargs["sliding_window"] = SlidingWindowAttentionConfig(
                pattern=[True],
                window_size=hf_config.window_size,
            )
            kwargs["use_flash"] = True
        
        # Create the transformer config using the olmo2_1B template
        # but with our specific parameters
        transformer_config = TransformerConfig.olmo2_1B(
            vocab_size=hf_config.vocab_size,
            dtype=DType.bfloat16,
            **kwargs
        )
        
        return transformer_config
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutput]:
        """
        Forward pass that delegates to your custom Titan model.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Your Titan model's forward pass
        if labels is not None:
            # Training mode - calculate loss
            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
                **kwargs
            )
            
            if hasattr(outputs, 'logits') and hasattr(outputs, 'loss'):
                # Already in the right format
                logits = outputs.logits
                loss = outputs.loss
            else:
                # Handle case where outputs is just logits
                logits = outputs
                loss = None
                if labels is not None:
                    # Calculate loss manually if needed
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            # Inference mode
            logits = self.model(input_ids=input_ids, **kwargs)
            loss = None
        
        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for generation."""
        # Handle any special preparation needed for your model
        model_inputs = {"input_ids": input_ids}
        model_inputs.update(kwargs)
        return model_inputs
    
    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.model.embeddings.wte if hasattr(self.model, 'embeddings') else None
    
    def set_input_embeddings(self, value):
        """Set input embeddings."""
        if hasattr(self.model, 'embeddings'):
            self.model.embeddings.wte = value
    
    def get_output_embeddings(self):
        """Get output embeddings."""
        return self.model.lm_head if hasattr(self.model, 'lm_head') else None
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings."""
        if hasattr(self.model, 'lm_head'):
            self.model.lm_head = new_embeddings


# OPTIONAL: Registration functions for HuggingFace AutoModel functionality
# These functions are LOCAL ONLY and do NOT upload anything online.
# They register the custom classes so that AutoConfig.from_pretrained() and 
# AutoModelForCausalLM.from_pretrained() can automatically find and use your custom classes.
# 
# If you prefer to explicitly specify the config_class and model_class when loading,
# you can comment out these lines:
#
# from transformers import AutoConfig, AutoModelForCausalLM
# AutoConfig.register("titan_olmo", TitanOLMoConfig)
# AutoModelForCausalLM.register(TitanOLMoConfig, TitanOLMoForCausalLM)
#
# Without registration, you would load the model like this:
# config = TitanOLMoConfig.from_pretrained(model_path)
# model = TitanOLMoForCausalLM.from_pretrained(model_path, config=config)
