from olmo_core.distributed.checkpoint import load_state_dict
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.data import TokenizerConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer_config = TokenizerConfig.dolma2()
OLMo_1B = TransformerConfig.olmo2_1B_v2(vocab_size=tokenizer_config.padded_vocab_size())
model = OLMo_1B.build()
base_state = model.state_dict()

# Download the Hugging Face model into a Transformers model
# alternate path: https://olmo-checkpoints.org/ai2-llm/peteish1/step1907359-unsharded/
hf_model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B")
hf_state = hf_model.state_dict()

def format_hf_keys(hf_state):
    formatted_state = {}
    for hf_key, value in hf_state.items():
        key = hf_key
        # remove 'model.' prefix
        if key.startswith("model."):
            key = key[len("model."):]

        # embed tokens
        if key == "embed_tokens.weight":
            key = "embeddings.weight"
        # positional embeddings, if present
        elif key == "embed_positions.weight":
            key = "embeddings.position_embeddings.weight"

        # transformer blocks
        elif key.startswith("layers."):
            # replace layer prefix
            parts = key.split(".")
            layer_idx = parts[1]
            suffix = ".".join(parts[2:])
            base_prefix = f"blocks.{layer_idx}."

            # self-attention projections and norms
            if suffix == "self_attn.q_proj.weight":
                key = base_prefix + "attention.w_q.weight"
            elif suffix == "self_attn.k_proj.weight":
                key = base_prefix + "attention.w_k.weight"
            elif suffix == "self_attn.v_proj.weight":
                key = base_prefix + "attention.w_v.weight"
            elif suffix == "self_attn.o_proj.weight":
                key = base_prefix + "attention.w_out.weight"
            elif suffix == "self_attn.q_norm.weight":
                key = base_prefix + "attention.q_norm.weight"
            elif suffix == "self_attn.k_norm.weight":
                key = base_prefix + "attention.k_norm.weight"

            # post-attention layer norm
            elif suffix == "post_attention_layernorm.weight":
                key = base_prefix + "attention_norm.weight"
            # MLP projections
            elif suffix == "mlp.gate_proj.weight":
                key = base_prefix + "feed_forward.w1.weight"
            elif suffix == "mlp.up_proj.weight":
                key = base_prefix + "feed_forward.w2.weight"
            elif suffix == "mlp.down_proj.weight":
                key = base_prefix + "feed_forward.w3.weight"
            # post-feedforward layer norm
            elif suffix == "post_feedforward_layernorm.weight":
                key = base_prefix + "feed_forward_norm.weight"
            else:
                # for any other layers.* keys, skip mapping
                continue

        # final layer norm and head
        elif key == "norm.weight":
            key = "lm_head.norm.weight"
        elif key == "norm.bias":
            key = "lm_head.norm.bias"
        elif key == "lm_head.weight":
            key = "lm_head.w_out.weight"
        elif key == "lm_head.bias" or key == "final_logits_bias":
            key = "lm_head.w_out.bias"

        else:
            # skip any other keys
            continue

        formatted_state[key] = value
    return formatted_state

def format_hf_values(base_state, load_state):
    for key, value in load_state.items():
        if key in base_state:
            # Ensure the shapes match
            base_shape = base_state[key].shape
            if base_shape != value.shape:
                assert base_shape == value.T.shape, f"Shape mismatch for {key}: {base_state[key].shape} vs {value.shape}"
                value = value.T
            load_state[key] = value.to(base_state[key].dtype)
    return load_state

format_state = format_hf_keys(hf_state)
format_state = format_hf_values(base_state, format_state)

missing, unexpected = model.load_state_dict(format_state, strict=True)
print(f"Missing {len(missing)} keys: {missing}")
print(f"Unexpected {len(unexpected)} keys: {unexpected}")

# Verifying the model
sample_text = "The capital of France is"
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
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
        logits = model(**model_inputs)  # [1, seq_len, vocab_size]
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        if attention_mask is not None:
            new_mask = torch.ones_like(next_token, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)

output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(output_text)
