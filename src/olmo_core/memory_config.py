from dataclasses import dataclass
from typing import Optional

@dataclass
class MemoryConfig:
    """Configuration class for memory components in the model."""
    persistent_mem_len: int = 4
    window_size: int = 16
    chunk_size: int = 16
    n_layers: int = 2
    hidden_dim_multiple: int = 2
    alpha: float = 0.999
    eta: float = 0.60
    theta: float = 0.05