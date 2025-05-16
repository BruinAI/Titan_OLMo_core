import torch
from transformers import AutoTokenizer
from olmo_core.nn.transformer.config import TransformerConfig
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

# Rebuilding the same Transformer architecture:
tok_cfg = TokenizerConfig.dolma2()
t_cfg = TransformerConfig.olmo2_1B(tok_cfg.padded_vocab_size())
model: torch.nn.Module = t_cfg.build().eval()

# this will pull in all of the sharded weights for you
load_model_and_optim_state(
    "../../converted/olmo2_1b/model_and_optim",
    model,
    optim=None,       # you can pass a real optimizer here, or None if you just care about weights
)


# Verifying the model
sample_text = "The capital of France is"
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
# model = hf_model  # to use HF model directly (for sanity check)
inputs = tokenizer(sample_text, return_tensors="pt")

# Manual autoregressive decoding loop
model.eval()
input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask", None)
max_new_tokens = 50
generated_ids = input_ids

with torch.no_grad():
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
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        if attention_mask is not None:
            new_mask = torch.ones_like(next_token, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(output_text)
