import sys
if "olmo_core" not in sys.path:
    sys.path.append("..")

import torch
from pathlib import Path
from olmo_core.distributed.checkpoint import unshard_checkpoint, prune_state_dict
from transformers import AutoTokenizer
from olmo_core.nn.transformer.config import TransformerConfig, TransformerBlockType, MemoryConfig
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

# Rebuilding the same Transformer architecture:
tok_cfg = TokenizerConfig.dolma2()
sliding_window_config = SlidingWindowAttentionConfig(pattern=[True], window_size=32)
USE_MAG = True
kwargs = {}
if USE_MAG:
    kwargs["block_name"] = TransformerBlockType.mag_reordered_norm
    kwargs["memory_config"] = MemoryConfig()
model_cfg = TransformerConfig.olmo2_1B(
    vocab_size=tok_cfg.padded_vocab_size(),
    # sliding_window=sliding_window_config, use_flash=True,
    **kwargs
)
model: torch.nn.Module = model_cfg.build()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
 # model = hf_model  # to use HF model directly (for sanity check)
inputs = tokenizer(sample_text, return_tensors="pt")

# Move inputs to the chosen device
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs.get("attention_mask", None)
if attention_mask is not None:
    attention_mask = attention_mask.to(device)

# Manual autoregressive decoding loop
model.eval()
max_new_tokens = 128
generated_ids = input_ids

with torch.no_grad() and torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
    for _ in range(max_new_tokens):
        model_inputs = {"input_ids": generated_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        # Use HF model output and extract logits
        logits = model(**model_inputs)  # [1, seq_len, vocab_size]
        # outputs = hf_model(**model_inputs)
        # logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        if attention_mask is not None:
            new_mask = torch.ones_like(next_token, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(output_text)
"""
The capital of France is Paris. The French language is spoken in France. The French people are known as the French. The
French flag is red with a white cross on a blue background. The French flag is the same as the flag of the United States.
The French language is the same as the language of the United States. The French language is the same as the language of
the United States. The French language is the same as the language of the United States. The French language is the same
as the language of the United States. The French language is the same as the language of the United States. The French
language is the same as the language of the [max tokens reached]
"""
