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

DESIRED_GPU_ID = 1
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(DESIRED_GPU_ID)
    
# Add test string for inference testing
TEST_STRING = """In a groundbreaking study on neural memory systems, researchers discovered that contextual associations 
between concepts can be mathematically represented as dense vector spaces. This finding has significant implications for 
how we understand both biological and artificial memory formation. The key insight from this work suggests that memory is 
not simply a passive storage mechanism, but rather an active, dynamic process that continuously integrates new information 
with prior knowledge.

The research team conducted a series of experiments using both computational models and neuroimaging techniques. They found 
that when individuals were presented with new information, their brains rapidly mapped these inputs onto existing memory 
structures, allowing for efficient retrieval and flexible reasoning. This mapping process was observed to be highly 
context-dependent, with the strength of associations varying according to relevance and recency.

Moreover, the study demonstrated that artificial neural networks equipped with similar memory architectures were able to 
outperform traditional models on a range of tasks, including language understanding, problem-solving, and pattern 
recognition. These networks exhibited remarkable generalization abilities, adapting to novel situations with minimal 
additional training.

The implications of these findings extend beyond neuroscience and artificial intelligence. They offer new perspectives on 
education, suggesting that teaching methods which emphasize contextual learning and the integration of new material with 
existing knowledge may be particularly effective. Furthermore, the research opens up exciting possibilities for the 
development of next-generation AI systems capable of human-like reasoning and lifelong learning.

As the field continues to evolve, future studies will likely explore the mechanisms underlying memory consolidation, the 
role of attention in shaping memory representations, and the potential for enhancing memory through targeted interventions. 
Ultimately, this line of research promises to deepen our understanding of intelligence itself, bridging the gap between 
biological and artificial systems in unprecedented ways. The key insight from this work suggests that"""

# Inference parameters
MAX_NEW_TOKENS = 100

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

from datasets import load_dataset
import json

all_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

def run_goemotion_benchmark(
        model, tokenizer, test_data, round, max_new_tokens=100, 
        output_file="goemotion_olmo2_1b_results.json", test_number=500
    ):
    all_labels = ["admiration",
                    "amusement",
                    "anger",
                    "annoyance",
                    "approval",
                    "caring",
                    "confusion",
                    "curiosity",
                    "desire",
                    "disappointment",
                    "disapproval",
                    "disgust",
                    "embarrassment",
                    "excitement",
                    "fear",
                    "gratitude",
                    "grief",
                    "joy",
                    "love",
                    "nervousness",
                    "optimism",
                    "pride",
                    "realization",
                    "relief",
                    "remorse",
                    "sadness",
                    "surprise",
                    "neutral"]

    def select_data(given_dataset, number_of_turns):
        selected_data_list = []
        label_to_data_dict = {}
        for data in given_dataset:
            if len(data['labels']) == 1:
                cur_label = data['labels'][0]
                if cur_label in label_to_data_dict:
                    label_to_data_dict[cur_label].append(data)
                else:
                    label_to_data_dict[cur_label] = [data]
        data_label_list = list(label_to_data_dict.keys())
        selected_label_to_count = {key:0 for key in data_label_list}
        for turn in range(number_of_turns):
            for i, key in enumerate(data_label_list):
                if len(label_to_data_dict[key]) > selected_label_to_count[key]:
                    selected_data_list.append(label_to_data_dict[key][selected_label_to_count[key]])
                    selected_label_to_count[key] += 1
                else:
                    for other in range(i+1, len(data_label_list)):
                        other_key = data_label_list[other]
                        if len(label_to_data_dict[other_key]) > selected_label_to_count[other_key]:
                            selected_data_list.append(label_to_data_dict[other_key][selected_label_to_count[other_key]])
                            selected_label_to_count[other_key] += 1
                            break

        print("selected_data_list: ", selected_data_list)
        print("selected data list length: ", len(selected_data_list))
        return selected_data_list

    def format_discovery_prompt(data_dict_list, round=0, with_instruction=False, context_token_number="2k"):
        token_shot_map_dict = {"2k": 73, "5k": 190, "10k": 380, "15k": 560, "20k": 740, "25k": 920,
                            "32k": 1180}
        prompt = 'Given a comment, please predict the emotion category of this comment. The predict answer must come from the demonstration examples with the exact format.'
        if with_instruction:
            prompt = prompt + 'You can only make prediction from the following categories: '
            for i, word in enumerate(all_labels):
                if i != len(all_labels) - 1:
                    prompt = prompt + word + ', '
                else:
                    prompt = prompt + word + '.\n'
        prompt = prompt + ' The examples are as follows: \n'
        if round != 0:
            index = len(data_dict_list)
            print(f"======={round} round running========")
            print("number of instances: ", index)
        else:
            index = token_shot_map_dict[context_token_number]
        data_list = data_dict_list[:index]
        for data in data_list:
            prompt = prompt + "comment: " + data['text'] + "\nemotion category: " + all_labels[data['labels'][0]] + '\n'
        return prompt

    dataset = load_dataset("go_emotions")
    train_data = dataset['train']
    test_data = dataset['test']
    demo_data = select_data(given_dataset=train_data, number_of_turns=round)
    eva_data = select_data(given_dataset=test_data, number_of_turns=25)
    total = 0
    correct = 0
    
    demo_prompt = format_discovery_prompt(demo_data,
                                        round=round,
                                        with_instruction=False,
                                        context_token_number="2k")

    if not os.path.exists(output_file.split('/')[0]):
        os.makedirs(output_file.split('/')[0])
    with open(output_file, mode='w', encoding='utf-8') as f:
        feeds = []
        f.write(json.dumps(feeds, indent=2))

    print(f"==========Evluation for Titans OLMo Round {round}==============")
    for example in eva_data[:test_number]:
        cur_prompt = demo_prompt + "comment: " + example['text'] + "\nemotion category: "
        inputs = tokenizer(cur_prompt, return_tensors='pt')
        print(inputs['input_ids'].shape)
        # For custom models, handle generation with proper device placement
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)
        
        with torch.no_grad():
            try:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=20,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    
            except RuntimeError as e:
                if "dtype" in str(e).lower():
                    print(f"Dtype issue detected, falling back to mixed precision: {e}")
                    # Fallback without autocast if there are dtype issues
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=20,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                else:
                    raise e
        
        # Decode only the new tokens
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

        print("org response: ", response)
        temp_prompt = "emotion category:"
        if example['text'] in response:
            response = list(response.split(example['text']))[-1].strip().split(temp_prompt)
            if len(response) > 1:
                response = response[1].split("comment: ")[0]
            else:
                response = response[0]
        else:
            response = response.split("comment: ")[0]
        response = response.strip().split("\n")[0]

        response = response.lower().strip()
        print("pred: ", response)
        label = all_labels[example['labels'][0]]
        label = label.lower()
        print("label: ", label)
        if response == label:
            correct += 1
        total += 1
        print("accuracy: ", correct/total)
        print("correct: ", correct)
        print("all: ", total)

        output_dict = {}
        output_dict['text'] = example['text']
        output_dict['label'] = label
        output_dict['pred'] = response
        feeds.append(output_dict)
        with open(output_file, mode='w', encoding='utf-8') as feedsjson:
            feedsjson.write(json.dumps(feeds, indent=2))


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
CHECKPOINT: Optional[str] = "/ssd/karen/titan_checkpoints/restart_low_lr/step17400"   #"/ssd/karen/titan_checkpoints/shape_get/step800"

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

SEQUENCE_LENGTH = 1024  # Limited to 1024 tokens as specified
INTRA_DOCUMENT_MASKING = False

# For single GPU, we can use a slightly larger batch size
BATCH_SIZE = 3  # Increased from 2 for single GPU efficiency
EFFECTIVE_BATCH_SIZE_COEFFICIENT = 16
RANK_MICROBATCH_SIZE = BATCH_SIZE * SEQUENCE_LENGTH
GLOBAL_BATCH_SIZE = EFFECTIVE_BATCH_SIZE_COEFFICIENT * BATCH_SIZE * SEQUENCE_LENGTH # Total batch size across all ranks

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
FULL_MODEL_LEARNING_RATE = 2e-4

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
        for _ in range(max_new_tokens):
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
            
    if cmd == "test":
        run_name, *overrides = sys.argv[2:]
        # Prepare training environment for single GPU
        prepare_training_environment()
        
        # Test inference doesn't need distributed setup
        config = build_config(run_name, overrides)
        try:
            test_inference(config)
        finally:
            teardown_training_environment()
        
    elif cmd == "goemotion":
        run_name, *overrides = sys.argv[2:]
        prepare_training_environment()
        config = build_config(run_name, overrides)
        try:
            # Model and tokenizer setup (same as test_inference)
            seed_all(config.init_seed)
            torch.set_float32_matmul_precision('high')
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = config.model.build(init_device=device)
            model = model.to(device)
            configure_training_parameters(model, only_train_memory=False)
            train_module = config.train_module.build(model)
            dataset = config.dataset.build()
            data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
            trainer_config = copy.deepcopy(config.trainer)
            trainer = trainer_config.build(train_module, data_loader)
            trainer.load_checkpoint(CHECKPOINT)
            model = train_module.model
            model.eval()
            model.generate = types.MethodType(simple_generate, model)
            tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
            goemotions = load_dataset("go_emotions")
            test_data = goemotions['test']
            for i in range(1, 6):
                run_goemotion_benchmark(model, tokenizer, test_data, round=i, max_new_tokens=MAX_NEW_TOKENS)
        finally:
            teardown_training_environment()
    else:
        print(usage)
        sys.exit(1)