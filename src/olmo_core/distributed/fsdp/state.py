from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

import torch
from torch.utils.hooks import RemovableHandle

from olmo_core.utils import get_default_device

from .stream import Stream

if TYPE_CHECKING:
    from .fsdp import FSDP


@dataclass
class FSDPState:
    device: torch.device = field(default_factory=get_default_device)
    """
    The device the FSDP node is running on.
    """

    pre_backward_hook_handles: List[RemovableHandle] = field(default_factory=list)
    """
    Backward hooks registered to the output tensors from the wrapped module's forward method.
    """

    post_backward_hook_handles: Dict[str, RemovableHandle] = field(default_factory=dict)
    """
    Post-backward hooks registered to the next autograd function in the graph for each parameter.
    The keys are parameter FQNs.
    """

    sharded_grad_cache: Dict[str, torch.Tensor] = field(default_factory=dict)
    """
    For caching sharded gradients during gradient accumulation.
    Maps param FQNs to the corresponding local sharded gradient.
    """

    lazy_init_complete: bool = False
    """
    Marked true when final initialization runs lazily during the first forward pass.
    """

    params_prefetched: bool = False
    """
    Indicates that the unsharded params have already been prefetched.
    """

    forward_execution_order: List[FSDP] = field(default_factory=list)
    """
    The forward-pass execution order of all FSDP instances as determined by the first forward pass.
    This is used on subsequent steps to determine the prefetch order.
    """

    forward_execution_order_finalized: bool = False
    """
    Marked true when the forward pass execution order has been finalized after the first forward pass.
    """

    forward_prefetch_queue: deque[FSDP] = field(default_factory=lambda: deque([]))
    """
    Queue of FSDP modules to prefetch for unsharding during forward pass.
    """

    backward_execution_order: List[FSDP] = field(default_factory=list)
    """
    The backward-pass execution order of all FSDP instances as determined by the first backward pass.
    This is used on subsequent steps to determine the prefetch order.
    """

    backward_execution_order_finalized: bool = False
    """
    Marked true when the backward pass execution order has been finalized after the first backward pass.
    """

    backward_prefetch_queue: deque[FSDP] = field(default_factory=lambda: deque([]))
    """
    Queue of FSDP modules to prefetch for unsharding during backward pass.
    """

    compute_stream: Stream = field(default_factory=Stream.default)
    """
    Default stream for computation.
    """

    unshard_stream: Stream = field(default_factory=Stream.default)
    """
    Stream for unsharding parameters.
    """

    reduce_stream: Stream = field(default_factory=Stream.default)
    """
    Stream for reducing gradients after the backward pass.
    """

    @property
    def current_stream(self) -> Stream:
        return Stream.current(self.device)
