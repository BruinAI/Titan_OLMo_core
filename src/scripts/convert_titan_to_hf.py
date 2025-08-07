#!/usr/bin/env python3
"""
Convert Titan OLMo checkpoint to HuggingFace format while preserving custom architecture.

This script creates a HuggingFace-compatible model that wraps your Titan architecture,
allowing you to run benchmarks while keeping all custom functionality intact.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer

from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.config import DType
from olmo_core.nn.hf.titan_model import TitanOLMoForCausalLM, TitanOLMoConfig

log = logging.getLogger(__name__)


def convert_titan_checkpoint_to_hf(
    checkpoint_path: str,
    output_path: str,
    config_path: Optional[str] = None,
    tokenizer_name: str = "allenai/OLMo-2-0425-1B",  # Use OLMo2 tokenizer
    max_sequence_length: int = 4096,
):
    """
    Convert a Titan OLMo checkpoint to HuggingFace format.
    
    Args:
        checkpoint_path: Path to your Titan checkpoint directory
        output_path: Where to save the HuggingFace model
        config_path: Path to config.json (if available)
        tokenizer_name: HuggingFace tokenizer to use as base
        max_sequence_length: Maximum sequence length for the model
    """
    
    checkpoint_dir = Path(checkpoint_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load your original model configuration
    if config_path:
        with open(config_path, 'r') as f:
            experiment_config = json.load(f)
        transformer_config_dict = experiment_config["model"]
        tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer", {})
    else:
        # Fallback configuration - using your exact training settings
        log.warning("No config provided, using default Titan OLMo2-1B configuration matching training script")
        from olmo_core.memory_config import MemoryConfig
        from olmo_core.nn.transformer.config import TransformerBlockType, TransformerBlockConfig
        from olmo_core.nn.attention import SlidingWindowAttentionConfig
        
        # Use your exact memory configuration from training script
        memory_config = MemoryConfig(
            persistent_mem_len=4,  # PERSISTENT_MEM_LEN
            window_size=512,       # WINDOW_SIZE
            chunk_size=256,        # CHUNK_SIZE
            n_layers=2,           # N_LAYERS
            hidden_dim_multiple=2, # HIDDEN_DIM_MULTIPLE
        )
        
        # Use your exact memory layers from training script
        memory_layers = [3, 7, 11, 15]  # MEMORY_LAYERS
        
        # Set up kwargs exactly like your training script
        kwargs = {}
        kwargs["block_name"] = TransformerBlockType.reordered_norm
        
        # First create a base config to get the default attention config
        base_config = TransformerConfig.olmo2_1B(
            vocab_size=100352,  # Your tokenizer padded vocab size
            dtype=DType.bfloat16,
            block_name=TransformerBlockType.reordered_norm,
        )
        
        # Now create block overrides using the base attention config
        block_overrides = {}
        for layer_idx in memory_layers:
            block_overrides[layer_idx] = TransformerBlockConfig(
                name=TransformerBlockType.mag_reordered_norm,
                attention=base_config.block.attention,  # Use the base config's attention
                layer_norm=base_config.block.layer_norm,  # Use the base config's layer_norm
                feed_forward=base_config.block.feed_forward,  # Use the base config's feed_forward
                memory_config=memory_config
            )
        kwargs["block_overrides"] = block_overrides

        # Add sliding window attention (matching your training script)
        kwargs["sliding_window"] = SlidingWindowAttentionConfig(
            pattern=[True],
            window_size=512,  # WINDOW_SIZE
        )
        kwargs["use_flash"] = True
        
        # Create transformer config using your exact setup
        transformer_config = TransformerConfig.olmo2_1B(
            vocab_size=100352,  # Your tokenizer padded vocab size
            dtype=DType.bfloat16,
            **kwargs
        )
        transformer_config_dict = transformer_config.as_dict()
        
        tokenizer_config = TokenizerConfig.dolma2()
        tokenizer_config_dict = tokenizer_config.as_dict()
    
    # 2. Build and load your Titan model
    log.info(f"Loading Titan model from {checkpoint_dir}")
    transformer_config = TransformerConfig.from_dict(transformer_config_dict)
    model = transformer_config.build()
    
    # Load the checkpoint
    model_and_optim_dir = checkpoint_dir / "model_and_optim"
    if model_and_optim_dir.exists():
        load_model_and_optim_state(model_and_optim_dir, model)
    else:
        log.warning(f"Model checkpoint not found at {model_and_optim_dir}")
    
    # 3. Create HuggingFace config using the proper method
    log.info("Creating HuggingFace configuration")
    
    # Extract memory layers from block_overrides
    memory_layers = []
    if hasattr(transformer_config, 'block_overrides') and transformer_config.block_overrides:
        memory_layers = list(transformer_config.block_overrides.keys())
    
    # Use the proper factory method to create the config
    hf_config = TitanOLMoConfig.from_transformer_config(
        transformer_config=transformer_config,
        memory_layers=memory_layers,
        max_position_embeddings=max_sequence_length,
    )
    
    # 4. Create the HuggingFace model wrapper
    log.info("Creating HuggingFace model wrapper")
    hf_model = TitanOLMoForCausalLM(hf_config)
    
    # Replace the model with your loaded model
    hf_model.model = model

    # 5. Setup tokenizer with padding to match model vocab size
    log.info("Setting up tokenizer")
    try:
        # First, try to use Dolma2 tokenizer from our config
        tokenizer_config = TokenizerConfig.dolma2()
        base_tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.identifier)
        
        # Check if we need to pad the tokenizer
        if len(base_tokenizer) != transformer_config.vocab_size:
            log.warning(f"Tokenizer vocab size ({len(base_tokenizer)}) != model vocab size ({transformer_config.vocab_size})")
            log.info("Padding tokenizer to match model vocab size...")
            
            # Calculate padding needed
            padding_tokens_needed = transformer_config.vocab_size - len(base_tokenizer)
            
            # Add padding tokens
            new_tokens = [f"<pad_{i}>" for i in range(padding_tokens_needed)]
            num_added = base_tokenizer.add_tokens(new_tokens)
            log.info(f"Added {num_added} tokens to tokenizer")
            
            # Check if vocab size is now correct, if not try manual approach
            if len(base_tokenizer) != transformer_config.vocab_size:
                log.info(f"Tokenizer still shows {len(base_tokenizer)} tokens, trying manual resize...")
                
                # Save and reload with modified config
                import tempfile
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    base_tokenizer.save_pretrained(temp_dir)
                    
                    # Modify tokenizer config
                    import os
                    config_path = os.path.join(temp_dir, "tokenizer_config.json")
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config_data = json.load(f)
                        config_data["vocab_size"] = transformer_config.vocab_size
                        with open(config_path, 'w') as f:
                            json.dump(config_data, f, indent=2)
                    
                    # Reload tokenizer
                    base_tokenizer = AutoTokenizer.from_pretrained(temp_dir)
            
            # Verify the vocab size matches now
            if len(base_tokenizer) == transformer_config.vocab_size:
                log.info(f"Successfully padded tokenizer to {len(base_tokenizer)} tokens")
                tokenizer = base_tokenizer
            else:
                log.error(f"Failed to pad tokenizer correctly. Expected {transformer_config.vocab_size}, got {len(base_tokenizer)}")
                tokenizer = base_tokenizer
        else:
            log.info("Tokenizer vocab size matches model")
            tokenizer = base_tokenizer
            
    except Exception as e:
        log.error(f"Could not setup tokenizer: {e}")
        log.info("Falling back to basic tokenizer")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e2:
            log.error(f"Fallback tokenizer also failed: {e2}")
            tokenizer = None
    
    # 6. Save everything
    log.info(f"Saving HuggingFace model to {output_dir}")
    
    # Save the model and config
    hf_model.save_pretrained(str(output_dir))
    
    # Save tokenizer if available
    if tokenizer:
        tokenizer.save_pretrained(str(output_dir))
    
    # Save additional metadata
    metadata = {
        "model_type": "titan_olmo",
        "original_checkpoint": str(checkpoint_dir),
        "conversion_notes": [
            "This model preserves the original Titan Neural Memory architecture",
            "All custom forward pass logic and memory modules are intact",
            "The model should work with HuggingFace pipelines and evaluation tools"
        ],
        "transformer_config": transformer_config_dict,
    }
    
    with open(output_dir / "titan_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log.info("Conversion completed successfully!")
    log.info(f"Model saved to: {output_dir}")
    log.info(f"To load: model = AutoModelForCausalLM.from_pretrained('{output_dir}', trust_remote_code=True)")


def create_modeling_file(output_path: Path):
    """
    Create a modeling_titan_olmo.py file in the output directory.
    This is needed for the trust_remote_code=True option to work.
    """
    modeling_file = output_path / "modeling_titan_olmo.py"
    
    # Copy our custom model definition
    import olmo_core.nn.hf.titan_model as titan_model_module
    shutil.copy(titan_model_module.__file__, modeling_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to your Titan checkpoint directory"
    )
    parser.add_argument(
        "--output-path", 
        type=str,
        required=True,
        help="Where to save the HuggingFace model"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to config.json (optional)"
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="allenai/OLMo-2-0425-1B",
        help="HuggingFace tokenizer to use as base (default: OLMo2 tokenizer)"
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=4096,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    convert_titan_checkpoint_to_hf(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        config_path=args.config_path,
        tokenizer_name=args.tokenizer_name,
        max_sequence_length=args.max_sequence_length,
    )
    
    # Create the modeling file for trust_remote_code
    create_modeling_file(Path(args.output_path))


if __name__ == "__main__":
    main()
