"""
Titan Memory-enabled training script for OLMo-2-1B.
This script implements a two-phase training approach:
1. Train only the memory modules with backbone frozen
2. Unfreeze the backbone and continue training

Adapted for single GPU local training without Beaker
"""
import numpy as np
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, cast, Dict, Any
import copy  # Add this import at the top of the file with other imports


import torch._dynamo as td
td.config.cache_size_limit = 16

DESIRED_GPU_ID = 1
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DESIRED_GPU_ID)
    
# Add test string for inference testing
TEST_STRING = """In a groundbreaking study on neural memory systems, researchers discovered that contextual associations 
between concepts can be mathematically represented as dense vector spaces. This finding has significant implications for 
how we understand both biological and artificial memory formation. The key insight from this work suggests that"""

# Inference parameters
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_P = 0.9
TOP_K = 40

from tqdm import tqdm
from pathlib import Path
import wandb
from olmo_core.nn.transformer.block import MAGReorderedNormTransformerBlock
from olmo_core.train.callbacks import Callback # Ensure Callback is imported
from olmo_core.train.trainer import Trainer # For type hinting
from olmo_core.train.train_module import TrainModule # For type hinting
import types # For MethodType

from olmo_core.ops import attach_auxiliary_loss


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

log = logging.getLogger(__name__)

#######################
#### CONFIGURATION ####
#######################

# GPU Configuration
# Set to 0 for the first GPU, 1 for the second, etc.
# PyTorch will see this GPU as 'cuda:0' after setting CUDA_VISIBLE_DEVICES.

# Phase configuration
# At the top of the file, change this line
PHASE = "full_model"  # Change from "memory_only" to "full_model"

# Set this if you want to start training from an existing checkpoint
CHECKPOINT: Optional[str] = None #"/ssd/karen/titan_checkpoints/restart_low_lr/step17400"   #"/ssd/karen/titan_checkpoints/shape_get/step800"

# Data configuration
# Path to your manifest file listing .npy data shards
DATA_MANIFEST_PATH: str = "/ssd/karen/Titan_OLMo_core/src/scripts/train/anneal/dolmino6b_sample.txt"

# Base prefix for the data shards listed in the manifest.
# If your .npy files are hosted, this would be the base HTTP/S3 URL.
# Example: BASE_DATA_PREFIX = "http://olmo-data.org"
# If they are local, this would be the root directory where the 'preprocessed/...' structure exists.
# Example: BASE_DATA_PREFIX = "/ssd/karen/olmo_data" # Ensure this path exists if local
# FIXME: PLEASE SET THIS TO THE CORRECT PREFIX FOR YOUR DATA:
BASE_DATA_PREFIX: str = "http://olmo-data.org" # Defaulting to HTTP, adjust if your data is local


# Memory configuration
MEMORY_LAYERS = [3,7,11,15]  # Every 4th layer
USE_SLIDING_WINDOW = True
WINDOW_SIZE = 512
PERSISTENT_MEM_LEN = 4  # Number of persistent memory tokens
CHUNK_SIZE = 256  # Size of chunks for memory processing
N_LAYERS = 2  # Number of layers in memory component
HIDDEN_DIM_MULTIPLE = 2  # Multiple for memory hidden dimension

# WandB configuration
WANDB_PROJECT = "Titan_OLMo"  # Your WandB project name
WANDB_ENTITY = "k_moss"  # Your WandB username or organization
WANDB_RUN_NAME = None  # Set to None to use the run_name parameter

CHUNK_NUMBER = 12
SEQUENCE_LENGTH = CHUNK_NUMBER*CHUNK_SIZE  # Limited to 1024 tokens as specified
INTRA_DOCUMENT_MASKING = False

# For single GPU, we can use a slightly larger batch size
BATCH_SIZE = 1  # Increased from 2 for single GPU efficiency
EFFECTIVE_BATCH_SIZE_COEFFICIENT = 1
RANK_MICROBATCH_SIZE = BATCH_SIZE * SEQUENCE_LENGTH
GLOBAL_BATCH_SIZE = EFFECTIVE_BATCH_SIZE_COEFFICIENT * BATCH_SIZE * SEQUENCE_LENGTH # Total batch size across all ranks

TOKENIZER_CONFIG = TokenizerConfig.dolma2()


# Configure memory settings
memory_config = MemoryConfig(
    persistent_mem_len=PERSISTENT_MEM_LEN,
    window_size=WINDOW_SIZE,
    chunk_size=CHUNK_SIZE,
    n_layers=N_LAYERS,
    hidden_dim_multiple=HIDDEN_DIM_MULTIPLE,
    # alpha=0.999,
    # eta=0.60,
    # theta=0.05
)

# Training configuration
MEMORY_ONLY_STEPS = 2000  # Number of steps for memory-only training
LEARNING_RATE = 5e-5
FULL_MODEL_LEARNING_RATE = 1e-4

# Local save path - create checkpoints directory if it doesn't exist
SAVE_DIR = os.path.expanduser("~/titan_checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

###########################
#### END CONFIGURATION ####
###########################

# Add these as class attributes to your trainer or module
class AuxiliaryLossTracker:
    def __init__(self, momentum=0.99, base_gates_weight=0.001, base_internal_weight=0.01, update_interval=10, enable_scaling=True):
        self.momentum = momentum
        self.running_avg_gates = None
        self.running_avg_internal = None
        self.running_avg_main = None
        self.step_count = 0
        self.base_gates_weight = base_gates_weight
        self.base_internal_weight = base_internal_weight
        self.update_interval = update_interval
        self.enable_scaling = enable_scaling  # New parameter to enable/disable scaling
        
        # Cache the current weights to avoid recalculating every step
        self.current_gates_weight = base_gates_weight
        self.current_internal_weight = base_internal_weight
        self.last_update_step = 0
    
    def update_and_get_weights(self, main_loss, gates_loss, internal_loss):
        
        # If scaling is disabled, just return the base weights without any computation
        if not self.enable_scaling:
            self.step_count += 1
            return self.base_gates_weight, self.base_internal_weight
        
        # Always update running averages (for monitoring) - only if scaling is enabled
        if self.running_avg_main is None:
            self.running_avg_main = main_loss.detach()
            self.running_avg_gates = gates_loss.detach()
            self.running_avg_internal = internal_loss.detach()
        else:
            self.running_avg_main = self.momentum * self.running_avg_main + (1 - self.momentum) * main_loss.detach()
            self.running_avg_gates = self.momentum * self.running_avg_gates + (1 - self.momentum) * gates_loss.detach()
            self.running_avg_internal = self.momentum * self.running_avg_internal + (1 - self.momentum) * internal_loss.detach()
        
        # Only recalculate weights every update_interval steps
        if (self.step_count % self.update_interval) == 0:
            self.current_gates_weight = self.base_gates_weight * self.running_avg_main / (self.running_avg_gates + 1e-8)
            self.current_internal_weight = self.base_internal_weight * self.running_avg_main / (self.running_avg_internal + 1e-8)
            
            self.last_update_step = self.step_count
        
        # Return the cached weights (updated only every update_interval steps)
        self.step_count += 1
        return self.current_gates_weight, self.current_internal_weight
    
    def get_current_weights(self):
        """Get current weights without updating anything"""
        if not self.enable_scaling:
            return self.base_gates_weight, self.base_internal_weight
        return self.current_gates_weight, self.current_internal_weight

class MemoryWeightTracker(Callback):
    def __init__(self, log_interval=10, track_block_idx=0, track_gradients=True):
        super().__init__()
        self.log_interval = log_interval
        self.track_block_idx = track_block_idx  # Only track this memory block index
        self.track_gradients = track_gradients
        self.target_memory_module = None
        self.target_module_name = None
        
    def pre_train(self):
        # Find and track only one specific memory module
        memory_modules_found = []
        
        for module_name, module in self.trainer.train_module.model.named_modules():
            if hasattr(module, 'memory') and module.memory is not None:
                memory_modules_found.append((module_name, module.memory))
        
        if len(memory_modules_found) > self.track_block_idx:
            self.target_module_name, self.target_memory_module = memory_modules_found[self.track_block_idx]
            print(f"Tracking only memory module: {self.target_module_name}")
            
            # Debug: Print structure of the tracked module
            print(f"Memory module type: {type(self.target_memory_module)}")
            param_count = 0
            for param_name, param in self.target_memory_module.named_parameters():
                param_count += 1
                print(f"  Parameter {param_count}: {param_name}, shape: {param.shape if param is not None else 'None'}")
        else:
            print(f"Warning: Could not find memory module at index {self.track_block_idx}")
    
    def post_train_batch(self):
        if self.trainer.global_step % self.log_interval == 0 and self.target_memory_module is not None:
            self._log_memory_metrics()
    
    def _log_memory_metrics(self):
        prefix = f"memory/{self.target_module_name.replace('.', '_')}"
        
        # Track only key metrics for main parameters
        self._log_main_parameters(prefix)
        
        # Track only one representative MLP template weight
        self._log_sample_mlp_weights(prefix)
        
    
    def _log_main_parameters(self, prefix):
        """Log only the main memory parameters (K, Q, V, alpha, eta, theta)"""
        main_params = ['K', 'Q', 'V', 'alpha', 'eta', 'theta']
        
        for param_name in main_params:
            if hasattr(self.target_memory_module, param_name):
                param_module = getattr(self.target_memory_module, param_name)
                if hasattr(param_module, 'weight') and param_module.weight is not None:
                    weight = param_module.weight
                    self.trainer.record_metric(f'{prefix}/{param_name}_weight_norm', weight.norm().item())
                    self.trainer.record_metric(f'{prefix}/{param_name}_weight_mean', weight.mean().item())
                    
                    # Track one sample element
                    self.trainer.record_metric(f'{prefix}/{param_name}_weight_sample', weight.flatten()[0].item())
                
                if hasattr(param_module, 'bias') and param_module.bias is not None:
                    bias = param_module.bias
                    self.trainer.record_metric(f'{prefix}/{param_name}_bias_norm', bias.norm().item())
                    self.trainer.record_metric(f'{prefix}/{param_name}_bias_mean', bias.mean().item())
    
    def _log_sample_mlp_weights(self, prefix):
        """Log only a sample of MLP template weights"""
        if not hasattr(self.target_memory_module, 'mlp_template_weights'):
            return
            
        mlp_weights = self.target_memory_module.mlp_template_weights
        
        if isinstance(mlp_weights, torch.nn.ParameterDict):
            # Just track the first weight parameter we find
            for key, param in mlp_weights.items():
                if param is not None and 'weight' in key:
                    self.trainer.record_metric(f'{prefix}/mlp_template_norm', param.norm().item())
                    self.trainer.record_metric(f'{prefix}/mlp_template_mean', param.mean().item())
                    self.trainer.record_metric(f'{prefix}/mlp_template_sample', param.flatten()[0].item())
                    break  # Only log the first weight we find

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
    # Initialize TOKENIZER_CONFIG
    tokenizer_config = TokenizerConfig.dolma2()

    # Parse the manifest file to get actual data paths
    actual_data_paths: List[str] = []
    log.info(f"Attempting to read data manifest from: {DATA_MANIFEST_PATH}")
    log.info(f"Using data base prefix: {BASE_DATA_PREFIX}")
    try:
        with open(DATA_MANIFEST_PATH, "r") as f:
            for line_num, line_content in enumerate(f):
                line_content = line_content.strip()
                if line_content and not line_content.startswith("#") and line_content.endswith(".npy"):
                    if BASE_DATA_PREFIX.endswith('/'):
                        full_path = f"{BASE_DATA_PREFIX}{line_content}"
                    else:
                        full_path = f"{BASE_DATA_PREFIX}/{line_content}"
                    actual_data_paths.append(full_path)
    except FileNotFoundError:
        log.error(f"Manifest file not found: {DATA_MANIFEST_PATH}")
        raise
    except Exception as e:
        log.error(f"Error reading manifest file {DATA_MANIFEST_PATH}: {e}")
        raise

    if not actual_data_paths:
        log.error(
            f"No .npy files found or parsed from manifest: {DATA_MANIFEST_PATH}. "
            f"Check manifest content and ensure BASE_DATA_PREFIX ('{BASE_DATA_PREFIX}') is correct."
        )
        raise ValueError(f"No .npy files parsed from manifest: {DATA_MANIFEST_PATH}")

    log.info(f"Successfully loaded {len(actual_data_paths)} data paths from manifest {DATA_MANIFEST_PATH}")
    if actual_data_paths:
        log.info(f"First few data paths: {actual_data_paths[:3]}")

    # Setup dataset configuration
    dataset_config = NumpyDatasetConfig(
        paths=actual_data_paths, # Use the parsed list of full .npy paths/URLs
        name=NumpyDatasetType.fsl,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=tokenizer_config, # Use the initialized tokenizer_config
        # work_dir should be a local path for caching.
        # Using a sub-directory in SAVE_DIR is a good practice.
        work_dir=str(Path(SAVE_DIR) / "dataset_cache"), # Ensure work_dir is a string
        generate_doc_lengths=INTRA_DOCUMENT_MASKING,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE, # Ensure GLOBAL_BATCH_SIZE is defined
        seed=34521, # Consider making this part of TitanExperimentConfig or a global
        num_workers=4,  # Reduced for single GPU
        # Add other NumpyDataLoaderConfig parameters as needed from your original setup
        prefetch_factor=4 if 2 > 0 else None, # Example, assuming NUM_WORKERS = 2
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
        weight_decay=0.01,
        betas=(0.9, 0.95),
        optim_bits=8,
        is_paged=True
    )

    # Use the config instance in the train_module_config
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SIZE, # Ensure RANK_MICROBATCH_SIZE is defined
        max_sequence_length=dataset_config.effective_sequence_length,
        optim=optim_config,
        compile_model=True, # Consider making this configurable
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp, 
            param_dtype=DType.bfloat16, 
            reduce_dtype=DType.float32,
        ),
        max_grad_norm=1.0, # Consider making this configurable
        scheduler=CosWithWarmup(warmup_steps=200), # Consider making warmup_steps configurable
    )

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
                    name=WANDB_RUN_NAME or run_name, # Ensure WANDB_RUN_NAME defined
                    entity=WANDB_ENTITY, # Ensure WANDB_ENTITY defined
                    project=WANDB_PROJECT, # Ensure WANDB_PROJECT defined
                    cancel_check_interval=10,
                    enabled=True, # Consider making this configurable
                ),
            )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        #.with_callback("memory_tracker", MemoryWeightTracker(log_interval=10))  # Add this line
    )
    
    class StepLimitCallback(Callback):
        def __init__(self):
            super().__init__()
            # Use the global MEMORY_ONLY_STEPS defined at the top of the script
            self.max_steps = MEMORY_ONLY_STEPS # Ensure MEMORY_ONLY_STEPS is defined
            
        def on_train_batch_end(self, trainer, *args, **kwargs):
            if trainer.step >= self.max_steps:
                trainer.should_stop = True
    
    if PHASE == "memory_only": # Ensure PHASE is defined
        trainer_config = trainer_config.with_callback(
            "step_limit", 
            StepLimitCallback
        )

    return TitanExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        phase=PHASE, # Ensure PHASE is defined
        memory_only_steps=MEMORY_ONLY_STEPS, # Ensure MEMORY_ONLY_STEPS is defined
        memory_layers=MEMORY_LAYERS, # Ensure MEMORY_LAYERS is defined
        use_sliding_window=USE_SLIDING_WINDOW, # Ensure USE_SLIDING_WINDOW is defined
        window_size=WINDOW_SIZE, # Ensure WINDOW_SIZE is defined
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


def log_accumulated_memory_metrics(train_module, trainer, aux_loss_tracker):
    """Log accumulated memory metrics averaged across gradient accumulation steps"""
    accumulator = train_module.memory_metrics_accumulator
    
    with torch.no_grad():
        # Log auxiliary loss metrics - only if scaling is enabled
        if aux_loss_tracker.enable_scaling:
            if accumulator['aux_gates_losses']:
                avg_gates_loss = torch.stack(accumulator['aux_gates_losses']).mean()
                trainer.record_metric('train/memory/aux_gates_squared_loss', avg_gates_loss)
                trainer.record_metric('train/memory/gates_weight', aux_loss_tracker.current_gates_weight)
                # Only log running averages if they exist (not None)
                if aux_loss_tracker.running_avg_gates is not None:
                    trainer.record_metric('train/memory/running_avg_gates', aux_loss_tracker.running_avg_gates)
            
            if accumulator['aux_internal_losses']:
                avg_internal_loss = torch.stack(accumulator['aux_internal_losses']).mean()
                trainer.record_metric('train/memory/aux_internal_loss', avg_internal_loss)
                trainer.record_metric('train/memory/internal_weight', aux_loss_tracker.current_internal_weight)
                # Only log running averages if they exist (not None)
                if aux_loss_tracker.running_avg_internal is not None:
                    trainer.record_metric('train/memory/running_avg_internal', aux_loss_tracker.running_avg_internal)
            
            # Log running averages - only if they exist
            if aux_loss_tracker.running_avg_main is not None:
                trainer.record_metric('train/memory/running_avg_main_loss', aux_loss_tracker.running_avg_main)
        else:
            # When scaling is disabled, just log the fixed weights
            if accumulator['aux_gates_losses']:
                avg_gates_loss = torch.stack(accumulator['aux_gates_losses']).mean()
                trainer.record_metric('train/memory/aux_gates_squared_loss', avg_gates_loss)
                trainer.record_metric('train/memory/gates_weight', aux_loss_tracker.base_gates_weight)
            
            if accumulator['aux_internal_losses']:
                avg_internal_loss = torch.stack(accumulator['aux_internal_losses']).mean()
                trainer.record_metric('train/memory/aux_internal_loss', avg_internal_loss)
                trainer.record_metric('train/memory/internal_weight', aux_loss_tracker.base_internal_weight)
        
        # Log chunk loss metrics (this doesn't depend on scaling)
        if accumulator['chunk_losses']:
            # Stack all chunk losses from all forward passes: [num_forwards, num_modules, num_chunks]
            all_chunk_losses = torch.stack(accumulator['chunk_losses'])  # Shape: [8, 4, 6]
            
            # Average across forward passes
            avg_chunk_losses = all_chunk_losses.mean(dim=0)  # Shape: [4, 6]
            
            # Calculate metrics
            avg_loss = avg_chunk_losses.mean()
            max_loss = avg_chunk_losses.max()
            loss_slope = (avg_chunk_losses[:, 0] - avg_chunk_losses[:, -1]).mean()
            last_loss = avg_chunk_losses[:, -1].mean()
            
            trainer.record_metric('train/memory/avg_chunk_loss', avg_loss)
            trainer.record_metric('train/memory/max_chunk_loss', max_loss)
            trainer.record_metric('train/memory/loss_slope', loss_slope)
            trainer.record_metric('train/memory/last_chunk_loss', last_loss)
        
        # Log gate statistics (this doesn't depend on scaling)
        if accumulator['gate_stats']:
            # Stack all gate stats from all forward passes: [num_forwards, num_modules, num_stats]
            all_gate_stats = torch.stack(accumulator['gate_stats'])  # Shape: [8, 4, 7]
            
            # Average across forward passes
            avg_gate_stats = all_gate_stats.mean(dim=0)  # Shape: [4, 7]
            
            # Calculate metrics (averaging across modules)
            min_gates_stat = avg_gate_stats[:, 0].mean()
            max_gate_stat = avg_gate_stats[:, 1].mean()
            mean_gate_stat = avg_gate_stats[:, 2].mean()
            std_gate_stat = avg_gate_stats[:, 3].mean()
            med_gate_stat = avg_gate_stats[:, 4].mean()
            small_gate_stat = avg_gate_stats[:, 5].mean()
            high_gate_stat = avg_gate_stats[:, 6].mean()
            
            trainer.record_metric('train/memory/mean_gate_stat', mean_gate_stat)
            trainer.record_metric('train/memory/max_gate_stat', max_gate_stat)
            trainer.record_metric('train/memory/min_gate_stat', min_gates_stat)
            trainer.record_metric('train/memory/std_gate_stat', std_gate_stat)
            trainer.record_metric('train/memory/small_gate_stat', small_gate_stat)
            trainer.record_metric('train/memory/med_gate_stat', med_gate_stat)
            trainer.record_metric('train/memory/high_gate_stat', high_gate_stat)
        
        # Log memory layer count (this doesn't need accumulation)
        memory_layer_count = 0
        for module_name, module_instance in train_module.model.named_modules():
            if isinstance(module_instance, MAGReorderedNormTransformerBlock):
                memory_layer_count += 1
        trainer.record_metric('train/memory/memory_layers_count', memory_layer_count)

def train(config: TitanExperimentConfig):
    # Set RNG states on all devices
    seed_all(config.init_seed)
    
    # Enable higher precision matrix multiplication
    torch.set_float32_matmul_precision('high')
    
    # Explicitly check which device we're using
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    
    # Diagnostic output to confirm which GPU is being used
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        device_props = torch.cuda.get_device_properties(current_device)
        log.info(f"Using CUDA device {current_device}: {device_name}")
        log.info(f"Device capability: {device_props.major}.{device_props.minor}")
        log.info(f"Total memory: {device_props.total_memory / 1e9:.2f} GB")
    
    # Build components - use "cuda:0" here as this will map to your selected GPU
    model = config.model.build(init_device=device)
    model = model.to(device)
    
    # Configure which parameters to train based on phase
    only_train_memory = (config.phase == "memory_only")
    trainable_params = configure_training_parameters(model, only_train_memory=only_train_memory)
    log.info(f"Training mode: {config.phase} ({trainable_params:,} trainable parameters)")
    
    # Build training components
    train_module = config.train_module.build(model)    
    
    # Keep a reference to the original method from the instance
    original_tm_model_forward = train_module.model_forward
    
    aux_loss_tracker = AuxiliaryLossTracker(
        momentum=0.999, 
        base_gates_weight=0.001, 
        base_internal_weight=0.02,
        update_interval=100,  # Update weights every 50 steps
        enable_scaling=True
    )

    # Define the new method that will replace train_module.model_forward
    # 'slf' will be the train_module instance when this is called as a method
    def new_patched_model_forward(slf, input_ids, labels=None, verbose_memory=False, **kwargs):
        # slf here is the train_module instance.
        # 'model' (the nn.Module) is accessible via slf.model
        # 'config' (TitanExperimentConfig) is accessible from the outer scope (closure)
        
        # The DEBUG print uses 'config' from the closure, which is fine.
        print(f"DEBUG: input_ids min: {input_ids.min()}, max: {input_ids.max()}, vocab_size: {config.model.vocab_size}")
        # Add near the top of your training loop
        torch.cuda.empty_cache()
        
        # Call the original model_forward method correctly
        # torch.compiler.cudagraph_mark_step_begin()
        outputs = original_tm_model_forward(input_ids, labels=labels, **kwargs)
        logits, loss, ce_loss, z_loss = outputs
        
        # Collect auxiliary losses from memory-enabled layers
        total_gates_squared_loss = 0.0
        total_internal_loss = 0.0
        memory_layer_count = 0
        
        # Iterate over the modules of the raw model (slf.model)
        for module_name, module_instance in slf.model.named_modules():
            if isinstance(module_instance, MAGReorderedNormTransformerBlock):
                memory_layer_count += 1
                
                if hasattr(module_instance, 'gates') and module_instance.gates is not None:
                    gates_squared = torch.sum(module_instance.gates ** 2)
                    total_gates_squared_loss += gates_squared
                
                if hasattr(module_instance, 'internal_loss') and module_instance.internal_loss is not None:
                    total_internal_loss += module_instance.internal_loss
        
        
        # Initialize accumulation storage if it doesn't exist
        if not hasattr(slf, 'memory_metrics_accumulator'):
            slf.memory_metrics_accumulator = {
                'chunk_losses': [],
                'gate_stats': [],
                'aux_gates_losses': [],
                'aux_internal_losses': [],
                'forward_count': 0
            }
        
        # Get adaptive weights based on running averages
        if memory_layer_count > 0:
            gates_weight, internal_weight = aux_loss_tracker.update_and_get_weights(
                loss, total_gates_squared_loss, total_internal_loss
            )
            loss = attach_auxiliary_loss(loss, gates_weight * total_gates_squared_loss)
            # Accumulate instead of logging immediately
            slf.memory_metrics_accumulator['aux_gates_losses'].append(total_gates_squared_loss.detach())
        
            loss = attach_auxiliary_loss(loss, internal_weight * total_internal_loss)
            # Accumulate instead of logging immediately
            slf.memory_metrics_accumulator['aux_internal_losses'].append(total_internal_loss.detach())
                
                
        # Manual gradient clipping after loss computation
        # Note: This will only work if gradients have been computed
        # if hasattr(slf, 'model'):
        #     # Check if gradients exist (they should after backward pass)
        #     has_grads = any(p.grad is not None for p in slf.model.parameters() if p.requires_grad)
        #     if has_grads:
        #         # Get the max_grad_norm from the train_module config
        #         max_grad_norm = getattr(slf, 'max_grad_norm', 5.0)
                
        #         # Manually call gradient clipping
        #         grad_norm = slf._clip_grad_norm(max_grad_norm)
                
        #         # Log the gradient norm for debugging
        #         print(f"DEBUG: Manual gradient clipping applied. Grad norm: {grad_norm:.4f}")
        
        
        # Collect memory chunk losses
        all_chunk_losses_for_step = []
        gate_stats_this_step = []
        
        for module_name, module_instance in slf.model.named_modules():
            if isinstance(module_instance, MAGReorderedNormTransformerBlock):
                if hasattr(module_instance, 'chunk_losses_this_forward') and module_instance.chunk_losses_this_forward:
                    all_chunk_losses_for_step.append(torch.stack(module_instance.chunk_losses_this_forward))
                    module_instance.chunk_losses_this_forward = []
                if hasattr(module_instance, 'gates_stats') and module_instance.gates_stats:
                    gate_stats_this_step.append(torch.stack(module_instance.gates_stats))
                    module_instance.gates_stats = []
        
        # Accumulate chunk losses and gate stats
        if all_chunk_losses_for_step:
            stacked_losses = torch.stack(all_chunk_losses_for_step)
            slf.memory_metrics_accumulator['chunk_losses'].append(stacked_losses.detach())
        
        if gate_stats_this_step:
            stacked_gate_stats = torch.stack(gate_stats_this_step)
            slf.memory_metrics_accumulator['gate_stats'].append(stacked_gate_stats.detach())
        
        # Increment forward count
        slf.memory_metrics_accumulator['forward_count'] += 1
        
        # Check if we've completed a full gradient accumulation cycle
        # This should match your EFFECTIVE_BATCH_SIZE_COEFFICIENT (8 in your case)
        if slf.memory_metrics_accumulator['forward_count'] >= EFFECTIVE_BATCH_SIZE_COEFFICIENT:
            # Time to log the accumulated metrics
            log_accumulated_memory_metrics(slf, trainer, aux_loss_tracker)
            
            # Reset accumulator for next step
            slf.memory_metrics_accumulator = {
                'chunk_losses': [],
                'gate_stats': [],
                'aux_gates_losses': [],
                'aux_internal_losses': [],
                'forward_count': 0
            }
        
        # ... rest of existing code ...
        
        # In the patched model_forward function, add this line after calculating gradients:
        if verbose_memory:
            for module_name, module in slf.model.named_modules():
                if hasattr(module, 'memory') and module.memory is not None:
                    module.memory.check_parameter_gradients()
                    print(module.memory.mlp_template_weights)

        # Dummy loss logic (using slf.model for parameters)
        # 'config' is from the closure.
        if config.phase == "memory_only":
            dummy_loss = 0.0
            for name, param in slf.model.named_parameters():
                if not param.requires_grad:
                    dummy_loss = dummy_loss + param.sum() * 0.0
            loss = loss + dummy_loss
            return logits, loss, ce_loss, z_loss
        else:  # "full_model" phase
            dummy_param_loss = 0.0
            for param in slf.model.parameters():
                if param.requires_grad:
                    dummy_param_loss += param.sum() * 0.0
            loss = loss + dummy_param_loss
            return logits, loss, ce_loss, z_loss
    
    # Bind the new function as a method to the train_module instance
    train_module.model_forward = types.MethodType(new_patched_model_forward, train_module)
    
    # Also override train_batch to use autocast
    original_train_batch = train_module.train_batch
    
    def train_batch_with_autocast(batch, *args, **kwargs):
        #torch.compiler.cudagraph_mark_step_begin()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            result = original_train_batch(batch, *args, **kwargs)
        #torch.cuda.synchronize()
        return result
    
    train_module.train_batch = train_batch_with_autocast
    
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    
    # Always create a trainer with checkpoint loading disabled for full_model
    if config.phase == "full_model":
        trainer_config = copy.deepcopy(config.trainer)
        trainer = trainer_config.build(train_module, data_loader)
        
        if CHECKPOINT is not None:
            trainer.load_checkpoint(CHECKPOINT)
        
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
        
# Custom generate function
def simple_generate(model, input_ids, max_new_tokens=20, eos_token_id=None, pad_token_id=None):
    """
    Simple autoregressive generation using argmax token selection.
    
    Args:
        model: The transformer model
        input_ids: Initial token ids to start generation from
        max_new_tokens: Maximum number of new tokens to generate
        eos_token_id: End of sequence token ID (optional)
        pad_token_id: Padding token ID (optional)
    
    Returns:
        torch.Tensor: The generated token IDs including the input
    """
    # Make sure model is in evaluation mode
    model.eval()
    
    # Keep track of original input shape and device
    device = input_ids.device
    batch_size = input_ids.shape[0]
    
    # Current sequence of tokens (will be extended during generation)
    curr_ids = input_ids
    
    # Reset memory MLPs for clean generation
    for module in model.modules():
        if hasattr(module, 'memory') and module.memory is not None:
            module.memory.reset_mlps()
    
    # Generate max_new_tokens, one token at a time
    with torch.no_grad():
        for _ in tqdm(range(max_new_tokens), desc="Generating"):
            # Forward pass through the model to get next token logits
            outputs = model(curr_ids)
            
            # Get logits for the next token (last position in sequence)
            next_token_logits = outputs[:, -1, :]
            
            # Simple argmax selection (no sampling)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append the new token to our current sequence
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # Stop if EOS token is generated in any sequence
            if eos_token_id is not None and (next_token == eos_token_id).any():
                # Find which sequences generated EOS
                eos_indices = (next_token == eos_token_id).nonzero()
                
                # Only stop if all sequences generated EOS
                if len(eos_indices) == batch_size:
                    break
    
    return curr_ids

def test_inference(config):
    """Run inference using a saved model checkpoint to continue the TEST_STRING."""
    # Set RNG states on all devices
    seed_all(config.init_seed)
    
    # Enable higher precision matrix multiplication
    torch.set_float32_matmul_precision('high')
    
    # Explicitly check which device we're using
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    
    # Diagnostic output to confirm which GPU is being used
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        device_props = torch.cuda.get_device_properties(current_device)
        log.info(f"Using CUDA device {current_device}: {device_name}")
        log.info(f"Device capability: {device_props.major}.{device_props.minor}")
        log.info(f"Total memory: {device_props.total_memory / 1e9:.2f} GB")
    
    # Build components - use "cuda:0" here as this will map to your selected GPU
    model = config.model.build(init_device=device)
    model = model.to(device)
    
    # Configure which parameters to train based on phase
    only_train_memory = (config.phase == "memory_only")
    trainable_params = configure_training_parameters(model, only_train_memory=only_train_memory)
    log.info(f"Training mode: {config.phase} ({trainable_params:,} trainable parameters)")
    
    # Build training components
    train_module = config.train_module.build(model)   
    
    
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)

    trainer_config = copy.deepcopy(config.trainer)
    trainer = trainer_config.build(train_module, data_loader)
    
    if CHECKPOINT is not None:
        trainer.load_checkpoint(CHECKPOINT)
    
    # Save config to checkpoint dir
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict
    
    
    
    # 8. Set model to eval mode after loading
    model = train_module.model
    model.eval()
    
    # 9. Add generate function to the model
    model.generate = types.MethodType(simple_generate, model)
    
    # 10. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    
    # 11. Tokenize input text
    log.info(f"Input text: {TEST_STRING[:100]}...")
    input_ids = tokenizer.encode(TEST_STRING, return_tensors="pt").to(model.device)
    
    # 12. Generate output
    log.info(f"Generating text with {MAX_NEW_TOKENS} tokens...")
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Reset memory MLPs before generation
            for module in model.modules():
                if hasattr(module, 'memory') and module.memory is not None:
                    module.memory.reset_mlps()
                    
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
    
    # 13. Decode output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # 14. Print results
    log.info("=" * 80)
    log.info("GENERATED TEXT:")
    log.info("=" * 80)
    log.info(generated_text)
    log.info("=" * 80)
    
    # 15. Highlight the newly generated portion
    original_length = len(TEST_STRING)
    continuation = generated_text[original_length:]
    
    log.info("CONTINUATION ONLY:")
    log.info("=" * 80)
    log.info(continuation)
    log.info("=" * 80)
    
    return generated_text

if __name__ == "__main__":
    usage = f"""
Usage
=====

› python {sys.argv[0]} train RUN_NAME [OVERRIDES...]
› python {sys.argv[0]} test CHECKPOINT_PATH

Examples
========

Train the memory-only phase:
› python {sys.argv[0]} train titan_run01 --phase=memory_only

Train the full-model phase with checkpoint:
› python {sys.argv[0]} train titan_run02 --phase=full_model --checkpoint=/path/to/memory_only_checkpoint

Test a model by generating text:
› python {sys.argv[0]} test /path/to/checkpoint/step1000
    """.strip()

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    cmd = sys.argv[1]
    
    # Set GPU selection using CUDA_VISIBLE_DEVICES before any CUDA calls
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
        log.info("CUDA not available. Using CPU.")
        
    # For single GPU, use these environment variables to set up distributed environment
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355" 
    os.environ["NUM_NODES"] = "1"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    os.environ["PYTORCH_FIND_UNUSED_PARAMETERS"] = "True"
        

    if cmd == "train":
        run_name, *overrides = sys.argv[2:]
        # Prepare training environment for single GPU
        prepare_training_environment()

        config = build_config(run_name, overrides)
        log.info(config)

        try:
            train(config)
        finally:
            teardown_training_environment()
            
    elif cmd == "test":
        run_name, *overrides = sys.argv[2:]
        # Prepare training environment for single GPU
        prepare_training_environment()
        
        # Test inference doesn't need distributed setup
        config = build_config(run_name, overrides)
        try:
            test_inference(config)
        finally:
            teardown_training_environment()
        
    else:
        print(usage)
        sys.exit(1)