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
from typing import Dict, Any

import torch
from transformers import AutoTokenizer

from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.nn.hf.titan_model import TitanOLMoForCausalLM, TitanOLMoConfig

log = logging.getLogger(__name__)


def convert_titan_checkpoint_to_hf(
    checkpoint_path: str,
    output_path: str,
    config_path: str = None,
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
    
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load your original model configuration
    if config_path:
        with open(config_path, 'r') as f:
            experiment_config = json.load(f)
        transformer_config_dict = experiment_config["model"]
        tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer", {})
    else:
        # Fallback configuration - you might need to adjust this
        log.warning("No config provided, using default Titan OLMo2-1B configuration")
        from olmo_core.memory_config import MemoryConfig
        
        memory_config = MemoryConfig(
            persistent_mem_len=4,
            window_size=64,
            chunk_size=64,
            n_layers=2,
            hidden_dim_multiple=1,
        )
        
        # You'll need to adapt this to match your actual model config
        transformer_config = TransformerConfig.olmo2_1B(vocab_size=50304)
        transformer_config_dict = transformer_config.as_config_dict()
        
        tokenizer_config = TokenizerConfig.dolma2()
        tokenizer_config_dict = tokenizer_config.as_config_dict()
    
    # 2. Build and load your Titan model
    log.info(f"Loading Titan model from {checkpoint_path}")
    transformer_config = TransformerConfig.from_dict(transformer_config_dict)
    model = transformer_config.build()
    
    # Load the checkpoint
    model_and_optim_dir = checkpoint_path / "model_and_optim"
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
    
    # 5. Setup tokenizer
    log.info("Setting up tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Adjust tokenizer vocab size if needed
        if len(tokenizer) != transformer_config.vocab_size:
            log.warning(f"Tokenizer vocab size ({len(tokenizer)}) != model vocab size ({transformer_config.vocab_size})")
    except Exception as e:
        log.error(f"Could not load tokenizer {tokenizer_name}: {e}")
        tokenizer = None
    
    # 6. Save everything
    log.info(f"Saving HuggingFace model to {output_path}")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the model and config
    hf_model.save_pretrained(str(output_dir))
    
    # Save tokenizer if available
    if tokenizer:
        tokenizer.save_pretrained(str(output_dir))
    
    # Save additional metadata
    metadata = {
        "model_type": "titan_olmo",
        "original_checkpoint": str(checkpoint_path),
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
