"""
Custom HuggingFace model for Titan OLMo with Neural Memory.
This preserves the custom architecture while making it compatible with HF ecosystem.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional, Union, Tuple, Dict, Any, List
import enum

from olmo_core.nn.transformer.model import Transformer
from olmo_core.nn.transformer.config import TransformerConfig, TransformerBlockType, TransformerBlockConfig
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.memory_config import MemoryConfig
from olmo_core.config import DType


def ensure_json_serializable(obj, _seen=None, _depth=0):
    """
    Convert objects to JSON-serializable types, handling recursion safely.
    """
    if _depth > 100:
        return str(obj)
    if _seen is None:
        _seen = set()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    obj_id = id(obj)
    if obj_id in _seen:
        return str(obj)
    _seen.add(obj_id)
    if isinstance(obj, list):
        return [ensure_json_serializable(item, _seen, _depth + 1) for item in obj]
    if isinstance(obj, tuple):
        return tuple(ensure_json_serializable(item, _seen, _depth + 1) for item in obj)
    if isinstance(obj, dict):
        return {str(k): ensure_json_serializable(v, _seen, _depth + 1) for k, v in obj.items()}
    # Handle enums
    if isinstance(obj, enum.Enum):
        return obj.value if hasattr(obj, 'value') else str(obj)
    # Handle objects with as_dict method
    try:
        if hasattr(obj, 'as_dict') and callable(getattr(obj, 'as_dict')):
            return ensure_json_serializable(obj.as_dict(), _seen, _depth + 1)
    except Exception:
        pass
    # Handle objects with __dict__
    try:
        if hasattr(obj, '__dict__'):
            return {k: ensure_json_serializable(v, _seen, _depth + 1)
                    for k, v in obj.__dict__.items() if not k.startswith('_')}
    except Exception:
        pass
    return str(obj)


class TitanOLMoConfig(PretrainedConfig):
    """
    Configuration class for TitanOLMo model.
    
    This stores all parameters needed to reconstruct your Titan model from checkpoints,
    including the complete transformer config and memory settings.
    """
    model_type = "titan_olmo"
    
    def __init__(
        self,
        # Standard transformer config - updated to match your exact training setup
        vocab_size=100352,  # From tokenizer.padded_vocab_size() (dolma2)
        hidden_size=2048,   # d_model from olmo2_1B config
        intermediate_size=None,  # Will be calculated: d_model * 1.5 * hidden_size_multiple_of
        num_hidden_layers=18,    # n_layers from olmo2_1B config  
        num_attention_heads=16,  # n_heads from olmo2_1B config
        max_position_embeddings=4096,  # Keep reasonable default
        rms_norm_eps=1e-6,      # layer_norm_eps from olmo2_1B config
        rope_theta=500000,      # theta from olmo2_1B config (overridden from default 10_000)
        pad_token_id=100277,    # From dolma2 tokenizer
        bos_token_id=100257,    # From dolma2 tokenizer 
        eos_token_id=100257,    # From dolma2 tokenizer
        # Titan-specific configuration matching your training script
        memory_layers=None,     # List of layer indices that have memory [3,7,11,15]
        memory_config=None,     # MemoryConfig parameters as dict
        use_sliding_window=True,
        window_size=512,        # WINDOW_SIZE from your training config
        qk_norm=True,          # From olmo2_1B config
        hidden_size_multiplier=1.5,  # From olmo2_1B config
        use_flash=True,        # From your training config when sliding window enabled
        # Store the complete transformer config for exact reconstruction as dict
        complete_transformer_config=None,
        **kwargs
    ):
        # Only convert memory_config and complete_transformer_config
        if memory_config is not None:
            memory_config = ensure_json_serializable(memory_config)
        if complete_transformer_config is not None:
            complete_transformer_config = ensure_json_serializable(complete_transformer_config)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        # Standard transformer parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # Calculate intermediate_size if not provided (matches olmo2_1B calculation)
        if intermediate_size is None:
            # From llama_like: ensure_multiple_of(int(d_model * hidden_size_multiplier), hidden_size_multiple_of)
            # Default hidden_size_multiple_of is 256
            calculated_size = int(hidden_size * hidden_size_multiplier)
            self.intermediate_size = ((calculated_size + 255) // 256) * 256  # Round up to multiple of 256
        else:
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
        self.qk_norm = qk_norm
        self.hidden_size_multiplier = hidden_size_multiplier
        self.use_flash = use_flash
        
        # Store complete config as dict for JSON serialization
        self.complete_transformer_config = complete_transformer_config
        
        for k, v in list(self.__dict__.items()):
            if k.startswith("_"):
                continue
            try:
                setattr(self, k, ensure_json_serializable(v))
            except Exception:
                pass

    @classmethod
    def from_transformer_config(
        cls, 
        transformer_config: TransformerConfig, 
        memory_layers: Optional[List[int]] = None,
        max_position_embeddings: int = 4096,
    ) -> "TitanOLMoConfig":
        """
        Create TitanOLMoConfig from a complete TransformerConfig.
        
        This is the preferred way to create the config as it preserves
        all the exact settings from training.
        """
        # Extract memory config from the first memory layer if available
        memory_config_dict = None
        if (hasattr(transformer_config, 'block_overrides') and 
            transformer_config.block_overrides and 
            memory_layers):
            first_memory_layer = memory_layers[0]
            if first_memory_layer in transformer_config.block_overrides:
                memory_cfg = transformer_config.block_overrides[first_memory_layer].memory_config
                if memory_cfg:
                    # Use __dict__ if as_dict is not available
                    memory_config_dict = getattr(memory_cfg, 'as_dict', lambda: dict(memory_cfg.__dict__))()
        
        # Only store a simple config dict for complete_transformer_config
        simple_config_dict = {
            'vocab_size': transformer_config.vocab_size,
            'd_model': transformer_config.d_model,
            'n_layers': transformer_config.n_layers,
            'block_type': str(getattr(transformer_config, 'block_name', 'unknown')),
        }
        # Create the config using the extracted values, ensuring everything is JSON serializable
        config = cls(
            vocab_size=transformer_config.vocab_size,
            hidden_size=transformer_config.d_model,
            num_hidden_layers=transformer_config.n_layers,
            num_attention_heads=transformer_config.block.attention.n_heads,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=getattr(transformer_config.block.layer_norm, 'eps', 1e-6),
            rope_theta=getattr(transformer_config.block.attention.rope, 'theta', 500000),
            memory_layers=memory_layers or [],
            memory_config=memory_config_dict,
            use_sliding_window=True,  # Assume true if memory layers exist
            window_size=512,  # Default from your training config
            qk_norm=getattr(transformer_config.block.attention, 'qk_norm', True),
            hidden_size_multiplier=1.5,
            use_flash=True,
            complete_transformer_config=simple_config_dict,
        )
        
        return config


class TitanOLMoForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Custom HuggingFace model that wraps your Titan OLMo architecture.
    
    This allows you to use your custom model with HF ecosystem while preserving
    all the custom neural memory functionality.
    """
    config_class = TitanOLMoConfig
    
    def __init__(self, config: TitanOLMoConfig):
        super().__init__(config)
        
        # Convert HF config back to TransformerConfig and build model
        transformer_config = self._convert_hf_to_olmo_config(config)
        self.model = transformer_config.build()
        
        # Initialize weights if needed
        self.post_init()
    
    def _convert_hf_to_olmo_config(self, hf_config: 'TitanOLMoConfig') -> TransformerConfig:
        # Only use from_dict if config is a full dict, not a simple summary
        if isinstance(hf_config.complete_transformer_config, dict) and 'block_type' not in hf_config.complete_transformer_config:
            return TransformerConfig.from_dict(hf_config.complete_transformer_config)
        # Otherwise, reconstruct from HF config as before
        from olmo_core.memory_config import MemoryConfig
        from olmo_core.nn.transformer.config import TransformerBlockType, TransformerBlockConfig
        from olmo_core.nn.attention import SlidingWindowAttentionConfig
        memory_config = None
        if hf_config.memory_config and isinstance(hf_config.memory_config, dict):
            memory_config = MemoryConfig(**hf_config.memory_config)
        kwargs = {}
        kwargs["block_name"] = TransformerBlockType.reordered_norm
        if hf_config.use_sliding_window:
            kwargs["sliding_window"] = SlidingWindowAttentionConfig(
                pattern=[True],
                window_size=hf_config.window_size,
            )
            kwargs["use_flash"] = hf_config.use_flash
        base_config = TransformerConfig.olmo2_1B(
            vocab_size=hf_config.vocab_size,
            dtype=DType.bfloat16,
            block_name=TransformerBlockType.reordered_norm,
        )
        if hf_config.memory_layers and memory_config:
            block_overrides = {}
            for layer_idx in hf_config.memory_layers:
                block_overrides[layer_idx] = TransformerBlockConfig(
                    name=TransformerBlockType.mag_reordered_norm,
                    attention=base_config.block.attention,
                    layer_norm=base_config.block.layer_norm,
                    feed_forward=base_config.block.feed_forward,
                    memory_config=memory_config
                )
            kwargs["block_overrides"] = block_overrides
        return TransformerConfig.olmo2_1B(
            vocab_size=hf_config.vocab_size,
            dtype=DType.bfloat16,
            **kwargs
        )
    
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
            hidden_states=None,
            attentions=None,
        )
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Prepare inputs for generation. Signature matches GenerationMixin."""
        return super().prepare_inputs_for_generation(*args, **kwargs)
    
    def get_input_embeddings(self) -> nn.Module:
        """Get input embeddings."""
        if hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'wte') and isinstance(self.model.embeddings.wte, nn.Module):
            return self.model.embeddings.wte
        # Fallback: try to find embeddings in the model structure
        for name, module in self.model.named_modules():
            if 'embed' in name.lower() and isinstance(module, nn.Module):
                return module
        # Return a dummy module if nothing found (shouldn't happen in practice)
        return nn.Embedding(1, 1)  # More specific than Identity
    
    def set_input_embeddings(self, value):
        """Set input embeddings."""
        if hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'wte'):
            self.model.embeddings.wte = value
        else:
            # Fallback: try to find and set embeddings in the model structure
            for name, module in self.model.named_modules():
                if 'embed' in name.lower() and hasattr(module, 'weight'):
                    # Replace the entire module
                    parent_names = name.split('.')[:-1]
                    parent = self.model
                    for parent_name in parent_names:
                        parent = getattr(parent, parent_name)
                    setattr(parent, name.split('.')[-1], value)
                    break
    
    def get_output_embeddings(self) -> nn.Module:
        """Get output embeddings."""
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head
        # Fallback: try to find lm_head in the model structure
        for name, module in self.model.named_modules():
            if 'lm_head' in name.lower() or 'output' in name.lower():
                return module
        # Return a dummy module if nothing found (shouldn't happen in practice)
        return nn.Linear(1, 1)  # More specific than Identity
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings."""
        if hasattr(self.model, 'lm_head'):
            self.model.lm_head = new_embeddings


def export_padded_dolma2_tokenizer(save_dir: str, pad_multiple: int = 128):
    """
    Export a Dolma2 tokenizer with padded vocab size to match the model.
    
    This function loads the base Dolma2 tokenizer from HuggingFace and modifies
    it to have a padded vocab size (100352 by default) to match your Titan model.
    
    Args:
        save_dir (str): Directory to save the padded tokenizer files
        pad_multiple (int): Padding multiple (default: 128)
    
    Usage:
        # Export padded tokenizer to your model directory
        export_padded_dolma2_tokenizer("/path/to/your/model/directory")
        
        # Then load with correct vocab size
        tokenizer = AutoTokenizer.from_pretrained("/path/to/your/model/directory")
        print(tokenizer.vocab_size)  # Should print 100352
    """
    import os
    from transformers import AutoTokenizer
    from olmo_core.data.tokenizer import TokenizerConfig
    
    # Get the TokenizerConfig to calculate padded size
    tokenizer_config = TokenizerConfig.dolma2()
    base_vocab_size = tokenizer_config.vocab_size  # 100278
    padded_vocab_size = tokenizer_config.padded_vocab_size(pad_multiple)  # 100352
    
    print(f"Base Dolma2 vocab size: {base_vocab_size}")
    print(f"Padded vocab size: {padded_vocab_size}")
    
    # Load the base tokenizer from HuggingFace
    base_tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.identifier)
    
    # Add padding tokens to reach the padded vocab size
    padding_tokens_needed = padded_vocab_size - base_vocab_size
    print(f"Adding {padding_tokens_needed} padding tokens...")
    
    # Add special padding tokens
    new_tokens = []
    for i in range(padding_tokens_needed):
        new_tokens.append(f"<pad_{i}>")
    
    # Add the new tokens to the tokenizer
    num_added = base_tokenizer.add_tokens(new_tokens)
    print(f"Successfully added {num_added} tokens to tokenizer")
    
    # For some tokenizers, add_tokens doesn't increase vocab_size immediately
    # We need to manually resize the tokenizer if needed
    if base_tokenizer.vocab_size != padded_vocab_size:
        print(f"Tokenizer vocab_size is still {base_tokenizer.vocab_size}, manually resizing...")
        
        # Create a new tokenizer config with the padded vocab size
        import tempfile
        import json
        
        # Save current tokenizer to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            base_tokenizer.save_pretrained(temp_dir)
            
            # Read and modify tokenizer_config.json
            config_path = os.path.join(temp_dir, "tokenizer_config.json")
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update vocab size in config
            config_data["vocab_size"] = padded_vocab_size
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Also update tokenizer.json if it exists
            tokenizer_json_path = os.path.join(temp_dir, "tokenizer.json")
            if os.path.exists(tokenizer_json_path):
                with open(tokenizer_json_path, 'r') as f:
                    tokenizer_data = json.load(f)
                
                # Add the padding tokens to the tokenizer.json vocab
                if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
                    vocab = tokenizer_data["model"]["vocab"]
                    current_max_id = max(vocab.values()) if vocab else -1
                    
                    # Add padding tokens to vocab
                    for i, token in enumerate(new_tokens):
                        vocab[token] = current_max_id + 1 + i
                    
                    # Update vocab size in the model section
                    if "vocab_size" in tokenizer_data["model"]:
                        tokenizer_data["model"]["vocab_size"] = padded_vocab_size
                
                with open(tokenizer_json_path, 'w') as f:
                    json.dump(tokenizer_data, f, indent=2)
            
            # Reload the tokenizer from the modified files
            base_tokenizer = AutoTokenizer.from_pretrained(temp_dir)
    
    # Final verification
    if base_tokenizer.vocab_size == padded_vocab_size:
        print(f"✓ Tokenizer vocab size is now correctly set to {base_tokenizer.vocab_size}")
    else:
        print(f"⚠ Warning: Tokenizer vocab size is {base_tokenizer.vocab_size}, expected {padded_vocab_size}")
        print("This might still work, but there could be compatibility issues.")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the padded tokenizer
    base_tokenizer.save_pretrained(save_dir)
    
    print(f"Padded tokenizer saved to: {save_dir}")
    print(f"Vocab size: {base_tokenizer.vocab_size}")
    print("\nTo use this tokenizer:")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{save_dir}')")
    print("model = AutoModelForCausalLM.from_pretrained('your_model_path', trust_remote_code=True)")


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
