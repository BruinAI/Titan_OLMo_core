#!/usr/bin/env python3
"""
Evaluate OLMo2-1B baseline model (without memory) on dolmino_6b_sample dataset.
This script calculates baseline performance metrics including loss and perplexity
for comparison with memory-enabled models.
"""

import os
import sys
import logging
import torch
import wandb
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
import math

# Add olmo_core to path if needed
if "olmo_core" not in sys.path:
    sys.path.append("..")
if not os.getcwd().endswith("src/scripts"):
    os.chdir("src/scripts")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from olmo_core.config import DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyDatasetConfig, NumpyDatasetType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.utils import seed_all

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Configuration
SEQUENCE_LENGTH = 1024
BATCH_SIZE = 4
DATA_MANIFEST_PATH = "/ssd/karen/Titan_OLMo_core/src/scripts/train/anneal/dolmino6b_sample.txt"
BASE_DATA_PREFIX = "http://olmo-data.org"
MODEL_CHECKPOINT_PATH = "../../converted/olmo2_1b/model_and_optim"
EVAL_STEPS = 5000  # Number of batches to evaluate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DESIRED_GPU_ID = 0

# Sliding window configuration
USE_SLIDING_WINDOW = True
WINDOW_SIZE = 512  # Match your memory config window size

# WandB Configuration
WANDB_PROJECT = "Titan_OLMo"
WANDB_ENTITY = "k_moss"
WANDB_RUN_NAME = f"olmo2_1b_baseline_eval{'_sw' if USE_SLIDING_WINDOW else ''}"

def load_data_paths(manifest_path: str, base_prefix: str) -> List[str]:
    """Load data paths from manifest file."""
    actual_data_paths = []
    try:
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if line.startswith("http"):
                        actual_data_paths.append(line)
                    else:
                        actual_data_paths.append(f"{base_prefix}/{line}")
    except FileNotFoundError:
        log.error(f"Data manifest file not found: {manifest_path}")
        raise
    
    if not actual_data_paths:
        raise ValueError("No data paths found in manifest file")
    
    log.info(f"Loaded {len(actual_data_paths)} data paths from manifest")
    return actual_data_paths

def build_model() -> torch.nn.Module:
    """Build the baseline OLMo2-1B model without memory."""
    tokenizer_config = TokenizerConfig.dolma2()
    
    # Configure sliding window attention if enabled
    kwargs = {}
    if USE_SLIDING_WINDOW and sys.platform != "darwin":  # Disable on macOS like in load_model_test
        kwargs["sliding_window"] = SlidingWindowAttentionConfig(
            pattern=[True], 
            window_size=WINDOW_SIZE,
        )
        kwargs["use_flash"] = True
        log.info(f"Using sliding window attention with window size: {WINDOW_SIZE}")
    else:
        if sys.platform == "darwin":
            log.info("Sliding window attention disabled on macOS")
        else:
            log.info("Using full attention (no sliding window)")
    
    model_config = TransformerConfig.olmo2_1B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        dtype=DType.bfloat16,
        **kwargs
    )
    
    model = model_config.build()
    model = model.to(DEVICE)
    
    # Load pretrained weights
    load_model_and_optim_state(
        MODEL_CHECKPOINT_PATH,
        model,
        optim=None,
    )
    
    log.info(f"Built baseline model with {model.num_params:,d} parameters")
    return model

def build_dataloader():
    """Build the data loader for evaluation."""
    tokenizer_config = TokenizerConfig.dolma2()
    data_paths = load_data_paths(DATA_MANIFEST_PATH, BASE_DATA_PREFIX)
    
    dataset_config = NumpyDatasetConfig(
        paths=data_paths,
        name=NumpyDatasetType.fsl,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=tokenizer_config,
        work_dir=str(Path.home() / "dataset_cache"),
        generate_doc_lengths=False,
    )
    
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=BATCH_SIZE * SEQUENCE_LENGTH,
        seed=34521, 
        num_workers=2,
        prefetch_factor=2,
    )
    
    # Build dataset and dataloader
    dataset = dataset_config.build()
    dataloader = data_loader_config.build(dataset)
    
    log.info(f"Built dataloader with batch size {BATCH_SIZE}")
    return dataloader


def calculate_metrics(model: torch.nn.Module, dataloader):
    """Calculate evaluation metrics on the dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    # Reshuffle the dataloader to set the epoch
    dataloader.reshuffle()
    
    log.info(f"Starting evaluation for {EVAL_STEPS} steps...")
    
    with torch.no_grad():
        pbar = tqdm(dataloader, total=EVAL_STEPS, desc="Evaluating")
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= EVAL_STEPS:
                break
                
            # Move batch to device
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(DEVICE)
            
            # Forward pass
            with torch.amp.autocast('cuda', enabled=DEVICE.startswith('cuda')):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                # Calculate loss (shift labels for causal LM)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                # Flatten for loss calculation
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                # Calculate cross entropy loss
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)
            
            # Accumulate metrics
            batch_tokens = shift_labels.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            num_batches += 1
                # Calculate final metrics
            avg_loss = total_loss / total_tokens
            avg_perplexity = math.exp(avg_loss)
            
            metrics = {
                'eval_avg_loss': avg_loss,
                'eval_avg_perplexity': avg_perplexity,
                'eval_tokens': total_tokens,
                'eval_batches': num_batches,
                'eval_current_loss': loss.item(),
                'eval_current_perplexity': math.exp(loss.item())
            }
            
            wandb.log(metrics)
            
            # Update progress bar
            current_avg_loss = total_loss / total_tokens
            current_perplexity = math.exp(current_avg_loss)
            pbar.set_postfix({
                'loss': f'{current_avg_loss:.4f}',
                'ppl': f'{current_perplexity:.2f}'
            })

def main():
    """Main evaluation function."""
    # Set random seed for reproducibility
    seed_all(42)
    
    # GPU selection using CUDA_VISIBLE_DEVICES
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if DESIRED_GPU_ID < num_gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(DESIRED_GPU_ID)
            log.info(
                f"Setting CUDA_VISIBLE_DEVICES='{DESIRED_GPU_ID}'. "
                f"PyTorch will see GPU {DESIRED_GPU_ID} as cuda:0."
            )
        else:
            log.warning(
                f"Desired GPU ID {DESIRED_GPU_ID} is not available (found {num_gpus} GPUs). "
                f"Defaulting to GPU 0 if available. Otherwise, CPU will be used."
            )
            if num_gpus > 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        log.info("CUDA not available. Evaluation will use CPU.")
    
    # Initialize WandB
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        config={
            'model_type': 'olmo2_1b_baseline',
            'sequence_length': SEQUENCE_LENGTH,
            'batch_size': BATCH_SIZE,
            'eval_steps': EVAL_STEPS,
            'device': DEVICE,
            'data_manifest': DATA_MANIFEST_PATH,
            'checkpoint_path': MODEL_CHECKPOINT_PATH,
            'use_sliding_window': USE_SLIDING_WINDOW,
            'window_size': WINDOW_SIZE if USE_SLIDING_WINDOW else None,
        }
    )
    
    try:
        # Build model and dataloader
        log.info("Building model...")
        model = build_model()
        
        log.info("Building dataloader...")
        dataloader = build_dataloader()
        
        # Calculate metrics
        log.info("Starting evaluation...")
        
        calculate_metrics(model, dataloader)
        log.info("Evaluation completed successfully!")
        
    except Exception as e:
        log.error(f"Evaluation failed: {e}")
        wandb.log({"error": str(e)})
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()