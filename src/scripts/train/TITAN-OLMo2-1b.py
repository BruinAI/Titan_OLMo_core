"""
Titan Memory-enabled training script for OLMo-2-1B.
This script implements a two-phase training approach:
1. Train only the memory modules with backbone frozen
2. Unfreeze the backbone and continue training

Adapted for single GPU local training without Beaker
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, cast, Dict, Any
import copy  # Add this import at the top of the file with other imports

from tqdm import tqdm
from pathlib import Path

if "olmo_core" not in sys.path:
    sys.path.append("..")
if not os.getcwd().endswith("src/scripts"):  # for VS Code debugging
    os.chdir("src/scripts")
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to avoid warnings

# Environment variables for Torch Compile -> Torch Inductor -> Torch Dynamo -> Triton and/or LLVM
# MUST HAVE Triton (if using GPU) and LLVM installed
if sys.platform == "darwin":  # if macOS
    os.environ["PATH"] = "/opt/homebrew/opt/llvm/bin:" + os.environ["PATH"]
    os.environ["LDFLAGS"] = "-L/opt/homebrew/opt/llvm/lib"
    os.environ["CPPFLAGS"] = "-I/opt/homebrew/opt/llvm/include"

import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer
import bitsandbytes as bnb

from olmo_core.distributed.checkpoint import unshard_checkpoint
from olmo_core.nn.transformer.config import TransformerConfig, TransformerBlockType, TransformerBlockConfig, TransformerType
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.memory_config import MemoryConfig

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig, TransformerBlockType

from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.memory_config import MemoryConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride
from olmo_core.optim.adamw import BNBAdamWConfig
from olmo_core.train import (
    TrainerConfig,
    prepare_cli_environment,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all
import bitsandbytes as bnb

from olmo_core.train.callbacks import WandBCallback

from olmo_core.train.callbacks import Callback

log = logging.getLogger(__name__)

#######################
#### CONFIGURATION ####
#######################

# Phase configuration
# At the top of the file, change this line
PHASE = "full_model"  # Change from "memory_only" to "full_model"
CHECKPOINT = None  # Make sure this is None for a fresh start

# Set this if you want to start training from an existing checkpoint
CHECKPOINT: Optional[str] = None

# Data configuration
SEQUENCE_LENGTH = 1024  # Limited to 1024 tokens as specified
TOKENIZER_CONFIG = TokenizerConfig.dolma2()

# Use data from dolmino6b_sample.txt
DATA_PATHS: List[str] = [
    "/Titan_OLMo_core/src/scripts/train/anneal/dolmino6b_sample.txt"
]

# For single GPU, we can use a slightly larger batch size
BATCH_SIZE = 2  # Increased from 2 for single GPU efficiency
RANK_MICROBATCH_SIZE = BATCH_SIZE * SEQUENCE_LENGTH
GLOBAL_BATCH_SIZE = RANK_MICROBATCH_SIZE
INTRA_DOCUMENT_MASKING = False

# Memory configuration
MEMORY_LAYERS = [3, 7, 11, 15]  # Every 4th layer
USE_SLIDING_WINDOW = True
WINDOW_SIZE = 512
PERSISTENT_MEM_LEN = 4  # Number of persistent memory tokens
CHUNK_SIZE = 128  # Size of chunks for memory processing
N_LAYERS = 2  # Number of layers in memory component
HIDDEN_DIM_MULTIPLE = 1  # Multiple for memory hidden dimension

# WandB configuration
WANDB_PROJECT = "Titan_OLMo"  # Your WandB project name
WANDB_ENTITY = "k_moss"  # Your WandB username or organization
WANDB_RUN_NAME = None  # Set to None to use the run_name parameter

# Configure memory settings
memory_config = MemoryConfig(
    persistent_mem_len=PERSISTENT_MEM_LEN,
    window_size=WINDOW_SIZE,
    chunk_size=CHUNK_SIZE,
    n_layers=N_LAYERS,
    hidden_dim_multiple=HIDDEN_DIM_MULTIPLE,
    alpha=0.999,
    eta=0.60,
    theta=0.05
)

# Training configuration
MEMORY_ONLY_STEPS = 2000  # Number of steps for memory-only training
LEARNING_RATE = 5e-5
FULL_MODEL_LEARNING_RATE = 1e-5

# Local save path - create checkpoints directory if it doesn't exist
SAVE_DIR = os.path.expanduser("~/titan_checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

###########################
#### END CONFIGURATION ####
###########################

@dataclass
class TitanExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 420
    phase: str = "memory_only"
    memory_only_steps: int = MEMORY_ONLY_STEPS
    memory_layers: List[int] = field(default_factory=lambda: MEMORY_LAYERS)
    use_sliding_window: bool = USE_SLIDING_WINDOW
    window_size: int = WINDOW_SIZE

def build_config(run_name: str, overrides: List[str]) -> TitanExperimentConfig:
    # Setup dataset configuration
    dataset_config = NumpyDatasetConfig(
        paths=DATA_PATHS,
        name=NumpyDatasetType.fsl,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=TOKENIZER_CONFIG,
        work_dir=os.path.dirname(DATA_PATHS[0]),
        generate_doc_lengths=INTRA_DOCUMENT_MASKING,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=34521,
        num_workers=2,  # Reduced for single GPU
    )

    kwargs = {}
    
    # Apply only to specific layers
    kwargs["block_name"] = TransformerBlockType.reordered_norm
    block_overrides = {}
    
    # Create block configs for specific memory layers
    for layer_idx in MEMORY_LAYERS:
        block_overrides[layer_idx] = TransformerBlockConfig(
                name=TransformerBlockType.mag_reordered_norm,
                attention=None,  # Will be filled by the config system
                layer_norm=None,  # Will be filled by the config system
                feed_forward=None,  # Will be filled by the config system
                memory_config=memory_config
            )
    kwargs["block_overrides"] = block_overrides

    # Add sliding window attention if enabled
    if USE_SLIDING_WINDOW:
        kwargs["sliding_window"] = SlidingWindowAttentionConfig(
            pattern=[True],
            window_size=WINDOW_SIZE,
        )
        kwargs["use_flash"] = True

    # Configure the model
    model_config = TransformerConfig.olmo2_1B(
        vocab_size=TOKENIZER_CONFIG.padded_vocab_size(),
        dtype=DType.bfloat16,  # Add this line to ensure model uses bf16
        **kwargs
    )

    # Set learning rate based on phase
    current_lr = LEARNING_RATE if PHASE == "memory_only" else FULL_MODEL_LEARNING_RATE
    
    # Create a BNBAdamWConfig instance directly (instead of a function)
    optim_config = BNBAdamWConfig(
        lr=current_lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        optim_bits=8,
        is_paged=True
    )

    # Use the config instance in the train_module_config
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        max_sequence_length=dataset_config.effective_sequence_length,
        optim=optim_config,
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp, 
            param_dtype=DType.bfloat16, 
            reduce_dtype=DType.float32,
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=200),
    )


    # Configure the trainer for local use
    trainer_config = (
        TrainerConfig(
            save_folder=f"{SAVE_DIR}/{run_name}",
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=500,
                ephemeral_save_interval=100,
                save_async=True,
            ),
        )
        .with_callback(
                "wandb",
                WandBCallback(
                    name=WANDB_RUN_NAME or run_name,
                    entity=WANDB_ENTITY,
                    project=WANDB_PROJECT,
                    cancel_check_interval=10,
                    enabled=True,
                ),
            )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
    )
    
    # Define a callback class that always uses the global MEMORY_ONLY_STEPS
    class StepLimitCallback(Callback):
        def __init__(self):
            super().__init__()
            self.max_steps = MEMORY_ONLY_STEPS
            
        def on_train_batch_end(self, trainer, *args, **kwargs):
            if trainer.step >= self.max_steps:
                trainer.should_stop = True
    
    if PHASE == "memory_only":
        # Register the class directly (not a factory function or instance)
        trainer_config = trainer_config.with_callback(
            "step_limit", 
            StepLimitCallback  # Pass the class itself, not a function or instance
        )

    # Build and return config
    return TitanExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        phase=PHASE,
        memory_only_steps=MEMORY_ONLY_STEPS,
        memory_layers=MEMORY_LAYERS,
        use_sliding_window=USE_SLIDING_WINDOW,
        window_size=WINDOW_SIZE,
    ).merge(overrides)

def configure_training_parameters(model, only_train_memory=True):
    """Configure which parameters to train based on the training phase"""
    if only_train_memory:
        # First, freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # Identify and unfreeze memory modules
        memory_param_count = 0
        memory_module_names = []
        
        # Identify memory modules
        for name, module in model.named_modules():
            if hasattr(module, 'memory') and module.memory is not None:
                memory_prefix = f"{name}.memory."
                memory_module_names.append(memory_prefix)
        
        # Unfreeze only memory parameters
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in memory_module_names):
                param.requires_grad = True
                memory_param_count += param.numel()
        
        log.info(f"Training only memory parameters: {memory_param_count:,} parameters")
    else:
        # Train all parameters
        total_params = 0
        for param in model.parameters():
            param.requires_grad = True
            total_params += param.numel()
        log.info(f"Training all parameters: {total_params:,} parameters")
    
    # Return count of trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(config: TitanExperimentConfig):
    # Set RNG states on all devices
    seed_all(config.init_seed)
    
    # Enable higher precision matrix multiplication
    torch.set_float32_matmul_precision('high')
    
    # Build components
    model = config.model.build(init_device="meta")
    
    # Configure which parameters to train based on phase
    only_train_memory = (config.phase == "memory_only")
    trainable_params = configure_training_parameters(model, only_train_memory=only_train_memory)
    log.info(f"Training mode: {config.phase} ({trainable_params:,} trainable parameters)")
    
    # Build training components
    train_module = config.train_module.build(model)
    
    # Override the model_forward method to add dummy loss for unused parameters
    original_model_forward = train_module.model_forward
    
    def patched_model_forward(input_ids, labels=None, **kwargs):
        outputs = original_model_forward(input_ids, labels=labels, **kwargs)
        
        # If we're in memory-only training mode, add dummy loss for unused parameters
        if config.phase == "memory_only":
            logits, loss, ce_loss, z_loss = outputs
            
            # Add tiny loss contribution from all parameters to ensure they get gradients
            dummy_loss = 0
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    # Add a tiny loss contribution (will be zero-grad during backward)
                    dummy_loss = dummy_loss + 0.0 * param.sum()
            
            # Return modified outputs with dummy loss that doesn't affect actual training
            # The 0.0 coefficient ensures it doesn't change the actual optimization
            loss = loss + dummy_loss
            return logits, loss, ce_loss, z_loss
        else:
            return outputs
    
    # Apply the patch
    train_module.model_forward = patched_model_forward
    
    # Also override train_batch to use autocast
    original_train_batch = train_module.train_batch
    
    def train_batch_with_autocast(batch, *args, **kwargs):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            return original_train_batch(batch, *args, **kwargs)
    
    train_module.train_batch = train_batch_with_autocast
    
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    
    # Always create a trainer with checkpoint loading disabled for full_model
    if config.phase == "full_model":
        # Use deepcopy instead of .copy() method
        trainer_config = copy.deepcopy(config.trainer)
        trainer_config.load_from_checkpoint = False
        
        # Create trainer with modified config
        trainer = trainer_config.build(train_module, data_loader)
        
        # Disable automatic checkpoint loading
        trainer._loaded_checkpoint = True
        
        # Override methods to ensure no checkpoint loading
        original_maybe_load = trainer.maybe_load_checkpoint
        trainer.maybe_load_checkpoint = lambda *args, **kwargs: True
        
        # Save config to checkpoint dir
        config_dict = config.as_config_dict()
        cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict
    else:
        # Regular trainer creation for memory_only phase
        trainer = config.trainer.build(train_module, data_loader)
        
        # Save config to checkpoint dir
        config_dict = config.as_config_dict()
        cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict
        
        # Regular checkpoint loading for memory_only phase
        if CHECKPOINT is not None:
            if not trainer.maybe_load_checkpoint(trainer.save_folder):
                trainer.load_checkpoint(CHECKPOINT)
    
    # Train for the specified number of steps
    if config.phase == "memory_only":
        # Train memory-only for the specified steps
        log.info(f"Starting memory-only training phase for {config.memory_only_steps} steps")
        trainer.fit()
        
        # Save a checkpoint at the end of memory-only training
        memory_only_checkpoint = f"{trainer.save_folder}"
        
        # Use the trainer's save_checkpoint method instead
        log.info("Saving final memory-only checkpoint...")
        trainer.save_checkpoint()
        
        log.info(f"Memory-only training completed, checkpoint saved to {memory_only_checkpoint}")

        # Recommend next steps
        log.info("To continue training with the full model unfrozen, run this script with:")
        log.info(f"  --phase=full_model --checkpoint={memory_only_checkpoint}")
        log.info(f"Memory-only training completed, checkpoint saved to {memory_only_checkpoint}")
        
        # Recommend next steps
        log.info("To continue training with the full model unfrozen, run this script with:")
        log.info(f"  --phase=full_model --checkpoint={memory_only_checkpoint}")
    else:
        # Continue training with full model unfrozen
        log.info(f"Starting full-model training phase with fresh optimizer state")
        trainer.fit()
        

if __name__ == "__main__":
    usage = f"""
Usage
=====

› python {sys.argv[0]} train RUN_NAME [OVERRIDES...]

Examples
========

Train the memory-only phase:
› python {sys.argv[0]} train titan_run01 --phase=memory_only

Train the full-model phase with checkpoint:
› python {sys.argv[0]} train titan_run02 --phase=full_model --checkpoint=/path/to/memory_only_checkpoint
    """.strip()

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    cmd, run_name, *overrides = sys.argv[1:]

    if cmd != "train":
        print(usage)
        sys.exit(1)

    # For single GPU, use this environment variable to disable distributed setup
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"  # Add this line
    os.environ["MASTER_PORT"] = "12355" 
    os.environ["NUM_NODES"] = "1"       # Add this line
    os.environ["LOCAL_WORLD_SIZE"] = "1" # Add this line
    os.environ["PYTORCH_FIND_UNUSED_PARAMETERS"] = "True"
    
    # Prepare training environment for single GPU
    prepare_training_environment()

    config = build_config(run_name, overrides)
    log.info(config)

    try:
        train(config)
    finally:
        teardown_training_environment()