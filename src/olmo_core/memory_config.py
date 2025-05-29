from dataclasses import dataclass

@dataclass
class MemoryConfig:
    """Configuration class for memory components in the model."""
    persistent_mem_len = 4
    window_size = 16
    chunk_size = 16
    n_layers = 2
    hidden_dim_multiple = 2
    alpha = 0.999
    eta = 0.60
    theta = 0.05