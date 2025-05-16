from olmo_core.distributed.checkpoint import load_state_dict
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.data import TokenizerConfig
from transformers import AutoModel, AutoModelForCausalLM
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
    # Convert the Hugging Face state dict to match the OLMo model's state dict
    formatted_state = {}
    for key, value in hf_state.items():
        if key.startswith("model."):
            key = key.replace("model.", "")
        if key.startswith("embed_tokens."):
            key = key.replace("embed_tokens.", "embeddings.")
        if key.startswith("layers."):
            key = key.replace("layers.", "blocks.")
        if ".self_attn." in key:
            key = key.replace(".self_attn.", ".attention.")
        if ".q_proj." in key:
            key = key.replace(".q_proj.", ".w_q.")
        if ".k_proj." in key:
            key = key.replace(".k_proj.", ".w_k.")
        if ".v_proj." in key:
            key = key.replace(".v_proj.", ".w_v.")
        # Map attention output projection
        if ".attention.o_proj." in key:
            key = key.replace(".attention.o_proj.", ".attention.w_out.")
        # Map post-attention layer norm
        if ".post_attention_layernorm." in key:
            key = key.replace(".post_attention_layernorm.", ".attention_norm.")
        # Map post-feedforward layer norm
        if ".post_feedforward_layernorm." in key:
            key = key.replace(".post_feedforward_layernorm.", ".feed_forward_norm.")
        # Map MLP projections to feed_forward
        if ".mlp.gate_proj." in key:
            key = key.replace(".mlp.gate_proj.", ".feed_forward.w1.")
        if ".mlp.up_proj." in key:
            key = key.replace(".mlp.up_proj.", ".feed_forward.w2.")
        if ".mlp.down_proj." in key:
            key = key.replace(".mlp.down_proj.", ".feed_forward.w3.")
        # Map final layer norms and head weights
        if key == "norm.weight":
            key = "lm_head.norm.weight"
        if key.startswith("lm_head.weight"):
            key = key.replace("lm_head.weight", "lm_head.w_out.weight")
        formatted_state[key] = value
    return formatted_state

def format_hf_values(base_state, load_state):
    for key, value in load_state.items():
        if key in base_state:
            # Ensure the shapes match
            if base_state[key].shape != value.shape:
                value = value.view(base_state[key].shape)
            load_state[key] = value.to(base_state[key].dtype)
    return load_state

format_state = format_hf_keys(hf_state)
format_state = format_hf_values(base_state, format_state)

missing, unexpected = model.load_state_dict(format_state, strict=False)
print(f"Missing {len(missing)} keys: {missing}")
print(f"Unexpected {len(unexpected)} keys: {unexpected}")

# Verifying the model
sample_text = "Hello, how are you?"
inputs = tokenizer_config.tokenize(sample_text)
