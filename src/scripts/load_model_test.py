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
MAX_TOKENS = 128
PROFILE_MEM = False
TRAIN_MODEL = False

# Layers that should use memory (e.g., only layers 0, 5, 10)
MEMORY_LAYERS = [0, 1, 2, 3, 4] # Maximum number of memory layers I can have without crashing on 20gb 5/19

if sys.platform == "darwin":  # if macos:
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

    def train_model_test():
        train_str = "The quick brown fox jumps over the lazy dog. The cat sat on the mat. The dog barked at the cat."  # > 
        input_ids, attention_mask = get_input_ids(train_str)

        # since model hasn't been called yet, mlps haven't been initialized -> not in model.parameters()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        ce_loss = torch.nn.CrossEntropyLoss()
        model.train()

        target = torch.nn.functional.one_hot(input_ids[:, 1:], num_classes=model_cfg.vocab_size).float()
        x = input_ids[:, :-1].clone()
        for i in tqdm(range(3)):
            outputs = model(x, attention_mask=attention_mask)[:, NUM_PERSISTENT:, :]
            loss = ce_loss(outputs, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {i}: Loss: {loss.item()}")

    train_model_test()
