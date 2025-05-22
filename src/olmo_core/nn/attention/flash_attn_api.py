from typing import Optional, Tuple
import numpy as np

import torch
import torch.distributed as dist

from .ring import RingAttentionLoadBalancerType

try:
    import flash_attn  # type: ignore
except ImportError:
    flash_attn = None

try:
    import ring_flash_attn  # type: ignore
except ImportError:
    ring_flash_attn = None
import torch, paddle
import torch.utils.dlpack as tdl
import paddle.utils.dlpack as pdl
from paddle.nn.functional.flash_attention import flashmask_attention
    
class PaddleFlashAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v,
                window_size=128, num_global_tokens=0,
                dropout_p=0.0, softmax_scale=None, causal=True):
        # >>> 1  bridge tensors to Paddle
        q_pd = paddle.from_dlpack(tdl.to_dlpack(q)); q_pd.stop_gradient=False
        k_pd = paddle.from_dlpack(tdl.to_dlpack(k)); k_pd.stop_gradient=False
        v_pd = paddle.from_dlpack(tdl.to_dlpack(v)); v_pd.stop_gradient=False

        # >>> 2  build sparse indices
        B, S, H, _ = q.shape
        starts = build_global_sliding_indices(S, window_size, num_global_tokens)
        starts = starts.unsqueeze(0).unsqueeze(0).expand(B, H, S)
        starts_pd = paddle.from_dlpack(tdl.to_dlpack(starts))

        # >>> 3  forward
        out_pd = flashmask_attention(
            q_pd, k_pd, v_pd,
            startend_row_indices=starts_pd,
            dropout=dropout_p,
            causal=causal,
            softmax_scale=softmax_scale,
            return_softmax=False,
        )

        # save Paddle tensors for backward
        ctx.save_for_backward(q_pd, k_pd, v_pd, starts_pd)
        ctx.other_args = (dropout_p, softmax_scale, causal)
        return torch.from_dlpack(pdl.to_dlpack(out_pd))

    @staticmethod
    def backward(ctx, grad_out):
        q_pd, k_pd, v_pd, starts_pd = ctx.saved_tensors
        grad_pd = paddle.from_dlpack(tdl.to_dlpack(grad_out.contiguous()))

        # run Paddle backward
        dq, dk, dv = paddle.grad(
            outputs=[q_pd, k_pd, v_pd],
            inputs=[q_pd, k_pd, v_pd],
            grad_outputs=[grad_pd]*3,
            retain_graph=False,
            create_graph=False,
        )
        # return grads to PyTorch
        return (torch.from_dlpack(pdl.to_dlpack(dq)),
                torch.from_dlpack(pdl.to_dlpack(dk)),
                torch.from_dlpack(pdl.to_dlpack(dv)),
                None, None, None, None, None)  # non-tensor args



def _flatten_batch_dim(x: torch.Tensor) -> torch.Tensor:
    B, T, *other = x.shape
    return x.view(B * T, *other)


def build_global_sliding_indices(seq_len: int, window_size: int, num_global_tokens: int) -> torch.Tensor:
    """Build indices for global + sliding window attention.
    
    Args:
        seq_len (int): Sequence length
        window_size (int): Size of sliding window
        num_global_tokens (int): Number of global tokens
        
    Returns:
        torch.Tensor: Start indices tensor of shape [seq_len]
    """
    indices = torch.arange(seq_len, device="cpu")
    # First num_global_tokens are always attended to (global tokens)
    starts = torch.clamp(indices - window_size // 2, min=num_global_tokens)
    # But we always want the global tokens (0 to num_global_tokens-1) to be attended to
    starts = torch.minimum(starts, torch.tensor(num_global_tokens))
    return starts

def dispatch_paddle_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 128,
    num_global_tokens: int = 0,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    **kwargs
) -> torch.Tensor:
    """Dispatch to PaddlePaddle's flash attention with global sliding window support.
    
    Args:
        q, k, v: Query, key, value tensors
        window_size: Size of sliding window
        num_global_tokens: Number of global tokens that receive attention from all positions
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for softmax
        causal: Whether to use causal attention
        
    Returns:
        torch.Tensor: Output of flash attention
    """
    try:
        import paddle
        from paddle.nn.functional.flash_attention import flash_attention
    except ImportError:
        raise ImportError(
            "PaddlePaddle is not installed. Please install it with `pip install paddlepaddle-gpu`"
        )
       
    
    # forward through Paddle kernel
    out = PaddleFlashAttn.apply(q, k, v, window_size=window_size, num_global_tokens=num_global_tokens,
                            dropout_p=dropout_p, causal=causal)
    
    return out


def dispatch_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if flash_attn is None:
        raise RuntimeError("flash-attn is required!")

    if cu_seqlens is not None:
        if cu_seqlens_q is None:
            cu_seqlens_q = cu_seqlens
        if cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens
    if max_seqlen is not None:
        if max_seqlen_q is None:
            max_seqlen_q = max_seqlen
        if max_seqlen_k is None:
            max_seqlen_k = max_seqlen

    varlen = all(x is not None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k))

    if varlen:
        return flash_attn.flash_attn_varlen_func(
            _flatten_batch_dim(q),
            _flatten_batch_dim(k),
            _flatten_batch_dim(v),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
    else:
        return flash_attn.flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )


def dispatch_flash_attn_qkvpacked(
    qkv: torch.Tensor,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int],
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if flash_attn is None:
        raise RuntimeError("flash-attn is required!")

    if cu_seqlens is not None and max_seqlen is not None:
        return flash_attn.flash_attn_varlen_qkvpacked_func(
            _flatten_batch_dim(qkv),
            cu_seqlens,
            max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )
    else:
        return flash_attn.flash_attn_qkvpacked_func(
            qkv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
        )


@torch._dynamo.disable()
def dispatch_ring_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    strategy: "RingAttentionLoadBalancerType",
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    heads_k_stride: Optional[int] = None,
    local_k_slice: Optional[slice] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if ring_flash_attn is None:
        raise RuntimeError("flash-attn and ring-flash-attn are required!")

    if strategy == RingAttentionLoadBalancerType.zig_zag:
        if any(x is not None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)):
            raise RuntimeError(
                f"{strategy} load balancing strategy requires unified QK doc lengths"
            )

        if local_k_slice is not None:
            raise RuntimeError(f"'local_k_slice' is invalid for {strategy} load balancing strategy")

        if cu_seqlens is not None and max_seqlen is not None:
            out = ring_flash_attn.zigzag_ring_flash_attn_varlen_func(
                _flatten_batch_dim(q),
                _flatten_batch_dim(k),
                _flatten_batch_dim(v),
                cu_seqlens,
                max_seqlen,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                group=group,
                window_size=window_size,
            )
        else:
            out = ring_flash_attn.zigzag_ring_flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                group=group,
                window_size=window_size,
            )
    elif strategy == RingAttentionLoadBalancerType.llama3:
        if any(x is not None for x in (cu_seqlens, max_seqlen)):
            raise RuntimeError(
                f"{strategy} load balancing strategy requires seperate QK doc lengths"
            )

        if (
            cu_seqlens_q is None
            or cu_seqlens_k is None
            or max_seqlen_q is None
            or max_seqlen_k is None
            or heads_k_stride is None
            or local_k_slice is None
        ):
            raise RuntimeError(
                f"{strategy} load balancing strategy is only implemented for 'varlen' variant.\n"
                "The following arguments are required: 'cu_seqlens_(q|k)', 'max_seqlen_(q|k)', "
                "'heads_k_stride', and 'local_k_slice'."
            )

        out = ring_flash_attn.llama3_flash_attn_varlen_func(
            _flatten_batch_dim(q),
            _flatten_batch_dim(k),
            _flatten_batch_dim(v),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            heads_k_stride,
            local_k_slice,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            group=group,
            window_size=window_size,
        )
    else:
        raise NotImplementedError(strategy)

    return out  # type: ignore


@torch._dynamo.disable()
def dispatch_ring_flash_attn_qkvpacked(
    qkv: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    strategy: RingAttentionLoadBalancerType,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int],
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    if ring_flash_attn is None:
        raise RuntimeError("flash-attn and ring-flash-attn are required!")

    if strategy == RingAttentionLoadBalancerType.zig_zag:
        if cu_seqlens is not None and max_seqlen is not None:
            out = ring_flash_attn.zigzag_ring_flash_attn_varlen_qkvpacked_func(
                _flatten_batch_dim(qkv),
                cu_seqlens,
                max_seqlen,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                group=group,
                window_size=window_size,
            )
        else:
            out = ring_flash_attn.zigzag_ring_flash_attn_qkvpacked_func(
                qkv,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                group=group,
                window_size=window_size,
            )
    else:
        raise NotImplementedError(strategy)

    return out  # type: ignore
