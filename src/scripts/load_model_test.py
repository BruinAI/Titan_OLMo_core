import sys
import os
from typing import Generator
from tqdm import tqdm
if "olmo_core" not in sys.path:
    sys.path.append("..")
if not os.getcwd().endswith("src/scripts"):  # for VS Code debugging
    os.chdir("src/scripts")
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to avoid warnings

# Environment variables for Torch Compile -> Torch Inductor -> Torch Dynamo -> Triton and/or LLVM
# MUST HAVE Triton (if using GPU) and LLVM installed
if sys.platform == "darwin":  # if macos:
    os.environ["PATH"] = "/opt/homebrew/opt/llvm/bin:" + os.environ["PATH"]
    os.environ["LDFLAGS"] = "-L/opt/homebrew/opt/llvm/lib"
    os.environ["CPPFLAGS"] = "-I/opt/homebrew/opt/llvm/include"  # for MacOS
# Debugging options for torch.compile
# os.environ["TORCHINDUCTOR_COMPILE_OPTIONS"] = "-Wl,-rpath,/usr/lib"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"
# os.environ["TORCH_COMPILE_DEBUG"] = "1"
# os.environ["TORCHINDUCTOR_VERBOSE"] = "1"

import torch
from torch.profiler import profile, ProfilerActivity
from pathlib import Path
from olmo_core.distributed.checkpoint import unshard_checkpoint
from transformers import AutoTokenizer
from olmo_core.nn.transformer.config import TransformerConfig, TransformerBlockType, MemoryConfig, TransformerBlockConfig, TransformerType
from olmo_core.nn.attention import SlidingWindowAttentionConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.data.tokenizer import TokenizerConfig

import bitsandbytes as bnb

# PREREQ: run the following in the root of the repo, uses repo's built-in HF converter
"""
python src/examples/huggingface/convert_checkpoint_from_hf.py \
  --checkpoint-input-path allenai/OLMo-2-0425-1B \
  --output-dir converted/olmo2_1b \
  --model-arch olmo2_1b \
  --tokenizer dolma2 \
  --max-sequence-length 4096 \
  --model-id allenai/OLMo-2-0425-1B
"""

"""
Current Kwargs flow (assuming it doesn't match a specific kwarg for each class along the way)
- olmo2_1B -> llama2_1B -> llama_like -> AttentionConfig -> Attention
        - Optional Flow 1: llama_like -> TransformerConfigBlock -> ReorderedNormTransformerBlock
        - Optional Flow 2: llama_like -> TransformerConfig -> Transformer
- This is needed to get the kwargs for the sliding window attention to flow through properly
Question: should kwargs for Neural Memory go through TransformerConfigBlockConfig or TransformerConfig instead?
1. MAG goes right after attention
    a. TransformerBlock's forward has everything we need for MAG: both the input and output of attn
    b. Make our own TransformerBlock modified with MAG
"""

USE_MAG = True
USE_SW = True
MAX_TOKENS = 256
TRAIN_MODEL = True
PROFILE_MEM = False

# Layers that should use memory (e.g., only layers 0, 5, 10)
MEMORY_LAYERS = [3, 7, 11, 15]  # every 4th layer

if sys.platform == "darwin":  # if macos, remove this when flash attn is deprecated
    USE_SW = False
    print("Sliding window attention not supported on macOS. Disabling...")

# Rebuilding the same Transformer architecture:
kwargs = {}
memory_config = MemoryConfig()
transformer_config_name = TransformerType.default

if USE_MAG:
    transformer_config_name=TransformerType.memory
    if MEMORY_LAYERS == "all":
        # Apply to all layers (current behavior)
        kwargs["block_name"] = TransformerBlockType.mag_reordered_norm
        kwargs["memory_config"] = memory_config
    else:
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

if USE_SW:
    kwargs["sliding_window"] = SlidingWindowAttentionConfig(
            pattern=[True], 
            window_size=memory_config.window_size,
    )
    kwargs["use_flash"] = True

tok_cfg = TokenizerConfig.dolma2()
model_cfg = TransformerConfig.olmo2_1B(vocab_size=tok_cfg.padded_vocab_size(), **kwargs)

model_cfg.name = transformer_config_name

model: torch.nn.Module = model_cfg.build()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_cuda = device.type == "cuda"
model = model.to(device)

if USE_MAG:
    model_pt = "tmp_unshard/model.pt"
    if not Path(model_pt).exists():
        model_pt, _ = unshard_checkpoint(
            dir="../../converted/olmo2_1b/model_and_optim",
            target_dir="tmp_unshard",
            optim=False,
            save_overwrite=True,
        )
    raw = torch.load(model_pt)
    model.load_state_dict(raw, strict=False)
else:
    load_model_and_optim_state(
        "../../converted/olmo2_1b/model_and_optim",
        model,
        optim=None,  # you can pass a real optimizer here, or None if you just care about weights
    )

# Verifying the model
sample_text = "The capital of France is"
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")

def get_input_ids(text):
    inputs = tokenizer(text, return_tensors="pt")
    input_token_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_token_ids, attention_mask

def generate(model, text, max_tokens=MAX_TOKENS) -> Generator[torch.types.Number, None, None]:
    input_ids, attention_mask = get_input_ids(text)
    generated_ids = input_ids
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=is_cuda):
        for i in range(max_tokens):
            model_inputs = {"input_ids": generated_ids}
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask

            # Use HF model output and extract logits
            logits = model(**model_inputs)  # [1, seq_len, vocab_size]
            # outputs = hf_model(**model_inputs)
            # logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if attention_mask is not None:
                new_mask = torch.ones_like(next_token, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
            # Stream each generated token in real time
            yield next_token.item()
            # ending generation if EOS token is reached
            if next_token.item() == tokenizer.eos_token_id:
                print("[Got EOS token]")
                break

if not TRAIN_MODEL:
    print(sample_text, end="")
    profiler = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if is_cuda else[ProfilerActivity.CPU]
    with profile(activities=profiler, profile_memory=True, record_shapes=True) as prof:
        for token in generate(model, sample_text, max_tokens=MAX_TOKENS):
            streamed_token = tokenizer.decode([token], skip_special_tokens=True)
            print(streamed_token, end="", flush=True)
            if token == tokenizer.eos_token_id:
                print("[Got EOS token]")
                break
        print("[Max Tokens Reached]")
    if PROFILE_MEM:
        key = "self_cuda_memory_usage" if is_cuda else "self_cpu_memory_usage"
        print(prof.key_averages().table(sort_by=key, row_limit=10))
        print()

    # output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # print(output_text)
    """
    The capital of France is Paris. The French language is spoken in France. The French people are known as the French. The
    French flag is red with a white cross on a blue background. The French flag is the same as the flag of the United States.
    The French language is the same as the language of the United States. The French language is the same as the language of
    the United States. The French language is the same as the language of the United States. The French language is the same
    as the language of the United States. The French language is the same as the language of the United States. The French
    language is the same as the language of the [max tokens reached]
    """
else:
    # Add this to your train_model_test function
    def check_for_nans(tensors_dict):
        has_nans = False
        for name, tensor in tensors_dict.items():
            if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
                print(f"NaN values found in {name}")
                has_nans = True
        return has_nans
    
    def get_gpu_memory_usage():
        """Return peak GPU memory usage in MB"""
        if not torch.cuda.is_available():
            return 0
        # Reset peak stats
        torch.cuda.reset_peak_memory_stats()
        # Run training
        yield
        # Get peak memory in MB
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return peak_memory
    
    def configure_training_parameters(model, only_train_memory=False, verbose=False):
        """Configure which parameters to train"""
        if only_train_memory:
            # First, freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
                
            # Then unfreeze only memory parameters
            memory_param_count = 0
            memory_module_names = []
            
            # First, identify the memory modules to ensure we only unfreeze those specific parameters
            for name, module in model.named_modules():
                if hasattr(module, 'memory') and module.memory is not None:
                    memory_prefix = f"{name}.memory."
                    memory_module_names.append(memory_prefix)
            
            # Debug: Print all memory module prefixes
            if verbose:
                print(f"Memory module prefixes: {memory_module_names}")
            
            # Store parameter counts by type for analysis
            param_type_counts = {}
            
            # Now unfreeze only parameters that are specifically part of memory modules
            for name, param in model.named_parameters():
                if any(name.startswith(prefix) for prefix in memory_module_names):
                    param.requires_grad = True
                    memory_param_count += param.numel()
                    
                    # Categorize the parameter for debugging
                    param_type = name.split('.')[-2] if len(name.split('.')) > 2 else "other"
                    if param_type not in param_type_counts:
                        param_type_counts[param_type] = 0
                    param_type_counts[param_type] += param.numel()
            
            # Debug: Print parameter breakdown by type
            if verbose:
                print("\nMemory parameter breakdown by type:")
                for param_type, count in sorted(param_type_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {param_type}: {count:,} parameters ({count/memory_param_count*100:.2f}%)")
                
                print(f"\nTraining only memory parameters: {memory_param_count:,} parameters")
                
                # Debug: List top 10 largest parameters
                print("\nTop 10 largest memory parameters:")
                param_sizes = [(name, param.numel()) for name, param in model.named_parameters() 
                            if param.requires_grad]
                for name, size in sorted(param_sizes, key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {name}: {size:,} parameters")
        else:
            # Train all parameters
            total_params = 0
            for param in model.parameters():
                param.requires_grad = True
                total_params += param.numel()
            if verbose:
                print(f"Training all parameters: {total_params:,} parameters")
        
        # Return count of trainable parameters
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def check_persistent_tokens(model, verbose=False):
        persistent_token_params = []
        for name, module in model.named_modules():
            if hasattr(module, 'memory') and module.memory is not None:
                if hasattr(module.memory, 'persistent_tokens'):
                    token_name = f"{name}.memory.persistent_tokens"
                    param = module.memory.persistent_tokens
                    is_grad = param.requires_grad
                    persistent_token_params.append((token_name, param.numel(), is_grad))
        
        if verbose:
            if persistent_token_params:
                print("\nPersistent tokens:")
                for name, size, is_grad in persistent_token_params:
                    status = "trainable" if is_grad else "frozen"
                    print(f"  {name}: {size:,} parameters ({status})")
            else:
                print("\nNo persistent tokens found in memory modules")

    def configure_activation_checkpointing(model, verbose=False):
        """Enable activation checkpointing on transformer blocks"""
        if not hasattr(torch, "utils") or not hasattr(torch.utils, "checkpoint"):
            print("Torch version doesn't support checkpointing")
            return model
            
        # Count how many transformer blocks we're applying checkpointing to
        checkpoint_count = 0
        
        # The actual naming convention in OLMo is different
        for name, module in model.named_modules():
            # Target transformer blocks - adjust the pattern to match OLMo's naming convention
            # Common patterns: "transformer.blocks", "transformer_blocks", etc.
            if "blocks" in name and not any(x in name for x in ["memory"]):
                # Create a closure-safe wrapper for the checkpoint function
                def make_checkpoint_forward(original_forward):
                    def checkpoint_forward(*args, **kwargs):
                        return torch.utils.checkpoint.checkpoint(
                            original_forward, *args, **kwargs, 
                            use_reentrant=False
                        )
                    return checkpoint_forward
                
                original_forward = module.forward
                module.forward = make_checkpoint_forward(original_forward)
                checkpoint_count += 1
                
                if verbose and checkpoint_count <= 5:  # Show first 5 checkpointed modules
                    print(f"Applied checkpointing to: {name}")
        
        print(f"Applied activation checkpointing to {checkpoint_count} transformer blocks")
        return model

        
    def train_model_test(verbose=False, use_checkpoint=False, cpu_offload=False):
        # Run two training configurations
        for train_mode in ["memory_only", "full_model"]:
            # Configure parameters to train
            only_train_memory = (train_mode == "memory_only")
            num_trainable_params = configure_training_parameters(model, only_train_memory=only_train_memory, verbose=verbose)
            # Apply activation checkpointing if requested
            if use_checkpoint:
                configure_activation_checkpointing(model)
                
            # Call after configure_training_parameters
            check_persistent_tokens(model, verbose=verbose)
            
            if verbose:
                print(f"\n{'=' * 50}")
                print(f"Training mode: {train_mode} ({num_trainable_params:,} trainable parameters)")
                print(f"{'=' * 50}\n")
            else:
                print(f"Training mode: {train_mode} ({num_trainable_params:,} trainable parameters)")
            
            # Move initialization outside autocast context
            train_str = """The quick brown fox jumps over the lazy dog. The cat sat on the mat. The dog barked at the cat. \
                The dog got very upset with the cat. The cat threw a buncha milk at the dog. Not a pretty sight. \
                When life gave the cat lemons it made lemonage. Oh how the cat wanted to eat hot dogs but the dog would not have it. \
                The very next day the cat tried to make pork sliders, but the dog did not want pork. It wanted freedom. \
                The dog felt in its bones the oppression its kin endured for thousands of years. It wanted to break free \
                of the shackles, but it could not bring itself to do so. For if it broke free, the world would isntantly become \
                a much more cruel place. Because taxes."""
            input_ids, attention_mask = get_input_ids(train_str)
            print(input_ids.shape)
            
            # Create optimizer with only trainable parameters
            if cpu_offload:
                optimizer = bnb.optim.AdamW8bit(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=5e-6,
                    optim_bits=8,      # Use 8-bit optimization  
                    is_paged=True      # Enable CPU paging of optimizer states
                ) 
            else:
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad], 
                    lr=5e-6
                )
            ce_loss = torch.nn.CrossEntropyLoss()
            model.train()
            
            target = torch.nn.functional.one_hot(input_ids[:, 1:], num_classes=model_cfg.vocab_size).float()
            x = input_ids[:, :-1].clone()
            
            # Track peak memory
            torch.cuda.reset_peak_memory_stats()
            
            for i in tqdm(range(5)):
                # Run with autocast but handle loss calculation in float32
                with torch.amp.autocast('cuda', enabled=is_cuda, dtype=torch.bfloat16):
                    outputs = model(x, attention_mask=attention_mask)
                    
                    # Debug: Check for NaNs in model outputs
                    if torch.isnan(outputs).any():
                        print(f"NaN detected in outputs - iteration {i}")
                        
                    # Convert to float32 for numerical stability in loss calculation
                    outputs_f32 = outputs.float()
                    target_f32 = target.float()
                    loss = ce_loss(outputs_f32, target_f32)
                
                # Skip iteration if loss is NaN
                if torch.isnan(loss).item():
                    print(f"NaN loss detected in iteration {i}, skipping")
                    optimizer.zero_grad()
                    continue
                    
                loss.backward()
                
                # Add gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                
                # Check for NaN gradients before optimizer step
                has_nan_grads = False
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        if verbose:
                            print(f"NaN gradient detected in {name}")
                        else:
                            print("NaN gradient detected")
                            break
                        has_nan_grads = True
                        
                if not has_nan_grads:
                    optimizer.step()
                else:
                    print("Skipping optimizer step due to NaN gradients")
                    
                optimizer.zero_grad()
                print(f"Epoch {i}: Loss: {loss.item()}")
            
            # Report peak memory usage
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"\nPeak GPU memory usage ({train_mode}): {peak_memory:.2f} MB")
            
            # Reset model and cache before next run
            if train_mode == "memory_only":
                # Re-initialize memory modules for next run
                for name, module in model.named_modules():
                    if hasattr(module, 'memory') and module.memory is not None:
                        module.memory.reset_mlps()
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    train_model_test(verbose=False, cpu_offload=True)
