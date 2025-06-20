# AOT ID: ['5_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_karen/74/c74sf73stm4xhba7sw5hzhd3uvfmyeulsa4iltgjdw2w5xkluw6v.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.native_layer_norm_backward, aten._to_copy, aten.native_layer_norm]
# Source node to ATen node mapping:
#   input_5 => convert_element_type_17, mul_4, sub_1
# Graph fragment:
#   %mul_7 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %primals_8), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, 2048), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_7, [1], True), kwargs = {})
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%addmm, torch.float32), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_17, %getitem_3), kwargs = {})
#   %mul_4 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %mul_4), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_9, [1], True), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %sum_3), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_8, %sum_2), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_3, %mul_10), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 2048), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %sub_4), kwargs = {})
#   %convert_element_type_18 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11, torch.bfloat16), kwargs = {})
triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_0 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp11 = tmp9 * tmp10
        tmp12 = tmp2 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr2 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = 0.00048828125
        tmp17 = tmp10 * tmp16
        tmp20 = tmp18 * tmp19
        tmp21 = 2048.0
        tmp22 = tmp20 * tmp21
        tmp23 = tmp22 - tmp4
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp25 - tmp8
        tmp27 = tmp26 * tmp10
        tmp28 = tmp27 * tmp14
        tmp29 = tmp23 - tmp28
        tmp30 = tmp17 * tmp29
        tmp31 = tmp30.to(tl.float32)
        tl.store(out_ptr2 + (r1 + 2048*x0), tmp31, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_karen/ab/cabew2dyvuayrawo7yrjq2qg42oaq6hpixpc75tnsx2qktkd3rbz.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._to_copy, aten.fill, aten.native_layer_norm, aten.silu, aten.sub, aten.mul, aten.add, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   input_2 => add_1, convert_element_type_10, mul_1, mul_2, sub
#   input_3 => sigmoid_1
# Graph fragment:
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 2048], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_10 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_1, torch.float32), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_10, %getitem_1), kwargs = {})
#   %mul_1 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %primals_4), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %primals_5), kwargs = {})
#   %sigmoid_1 : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default, %sigmoid_1), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %sub_5), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_13, 1), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_1, %add_5), kwargs = {})
#   %mul_15 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, %mul_14), kwargs = {})
#   %mul_17 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %primals_4), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, 2048), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_17, [1], True), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %mul_1), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_19, [1], True), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %sum_8), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_18, %sum_7), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_7, %mul_20), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 2048), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %sub_8), kwargs = {})
#   %convert_element_type_27 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_21, torch.bfloat16), kwargs = {})
triton_red_fused__to_copy_add_fill_mul_native_layer_norm_native_layer_norm_backward_silu_sub_1 = async_compile.triton('triton_red_fused__to_copy_add_fill_mul_native_layer_norm_native_layer_norm_backward_silu_sub_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'out_ptr0': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_fill_mul_native_layer_norm_native_layer_norm_backward_silu_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_fill_mul_native_layer_norm_native_layer_norm_backward_silu_sub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr5 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 - tmp2
        tmp5 = tmp3 * tmp4
        tmp7 = tmp5 * tmp6
        tmp9 = tmp7 + tmp8
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.sigmoid(tmp9)
        tmp13 = 1.0
        tmp14 = tmp13 - tmp12
        tmp15 = tmp9 * tmp14
        tmp16 = tmp15 + tmp13
        tmp17 = tmp12 * tmp16
        tmp18 = tmp11 * tmp17
        tmp19 = tmp18 * tmp6
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
        tmp23 = tmp19 * tmp5
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
        tl.store(out_ptr0 + (r1 + 2048*x0), tmp9, rmask & xmask)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_ptr5 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(out_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp39 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp44 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = 0.00048828125
        tmp28 = tmp4 * tmp27
        tmp30 = tmp29.to(tl.float32)
        tmp32 = tl.sigmoid(tmp31)
        tmp33 = 1.0
        tmp34 = tmp33 - tmp32
        tmp35 = tmp31 * tmp34
        tmp36 = tmp35 + tmp33
        tmp37 = tmp32 * tmp36
        tmp38 = tmp30 * tmp37
        tmp40 = tmp38 * tmp39
        tmp41 = 2048.0
        tmp42 = tmp40 * tmp41
        tmp43 = tmp42 - tmp21
        tmp45 = tmp44.to(tl.float32)
        tmp46 = tmp45 - tmp2
        tmp47 = tmp46 * tmp4
        tmp48 = tmp47 * tmp25
        tmp49 = tmp43 - tmp48
        tmp50 = tmp28 * tmp49
        tmp51 = tmp50.to(tl.float32)
        tl.store(out_ptr3 + (r1 + 2048*x0), tmp51, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_karen/rx/crxhyqwr3n36savsdses2xalkq3e4z4lzhifywni7sijjnzjophc.py
# Topologically Sorted Source Nodes: [input_5, queries, queries_1], Original ATen: [aten._to_copy, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.add, aten.silu, aten.div, aten.neg, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   input_5 => convert_element_type_17, mul_4, sub_1
#   queries => convert_element_type_3, convert_element_type_4, mul, sigmoid
#   queries_1 => div
# Graph fragment:
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%addmm, torch.float32), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_17, %getitem_3), kwargs = {})
#   %mul_4 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %mul_4), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_12, [0]), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%select_1, [0]), kwargs = {})
#   %convert_element_type_32 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_5, torch.float32), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 32, 2048], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %convert_element_type_32, 0, 0), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_1, %select_scatter_default), kwargs = {})
#   %convert_element_type_3 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_3,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, %sigmoid), kwargs = {})
#   %convert_element_type_4 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_4, %expand), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div, %expand), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_6,), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %div_4), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_23, [1], True), kwargs = {dtype: torch.float32})
triton_per_fused__to_copy_add_div_mul_native_layer_norm_native_layer_norm_backward_neg_select_backward_silu_sum_2 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_native_layer_norm_native_layer_norm_backward_neg_select_backward_silu_sum_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_native_layer_norm_native_layer_norm_backward_neg_select_backward_silu_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 3, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_native_layer_norm_native_layer_norm_backward_neg_select_backward_silu_sum_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2048*r1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 2048*r1), xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x0 + 2048*r1), xmask, other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr5 + (x0 + 2048*r1), xmask, other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = tmp16 == tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = 0.0
    tmp21 = tl.where(tmp17, tmp19, tmp20)
    tmp22 = tmp0 + tmp21
    tmp23 = -tmp22
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tl.sigmoid(tmp25)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28.to(tl.float32)
    tmp31 = 1e-08
    tmp32 = triton_helpers.maximum(tmp30, tmp31)
    tmp33 = tmp29 / tmp32
    tmp34 = tmp33 / tmp32
    tmp35 = tmp23 * tmp34
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
    tmp38 = tl.where(xmask, tmp36, 0)
    tmp39 = tl.sum(tmp38, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, xmask)
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tl.store(out_ptr2 + (x0), tmp39, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_karen/3j/c3j6errrc4ryh7hxs4efrqygjc6kxy5gxcqjyfahrkdqc64kittl.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_18, [0], True), kwargs = {dtype: torch.float32})
triton_per_fused_sum_3 = async_compile.triton('triton_per_fused_sum_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_sum_3(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2048*r1), xmask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_karen/xs/cxshtk4mvkvq7ctt7jrztxflxr65rvniyti3vm24y6gmwgj7fsar.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_25 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_3, torch.float32), kwargs = {})
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_karen/45/c45aoywkevqgzojgsfxaccbuagdldlcy6vaxpfw35cdyhuxs4tha.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._to_copy, aten.fill, aten.native_layer_norm, aten.silu, aten.sub, aten.mul, aten.add, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   input_2 => convert_element_type_10, mul_1, sub
#   input_3 => sigmoid_1
# Graph fragment:
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 2048], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_10 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_1, torch.float32), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_10, %getitem_1), kwargs = {})
#   %mul_1 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %sigmoid_1 : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default, %sigmoid_1), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %sub_5), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_13, 1), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_1, %add_5), kwargs = {})
#   %mul_15 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, %mul_14), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %mul_1), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_22, [0]), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_15, [0]), kwargs = {})
triton_per_fused__to_copy_add_fill_mul_native_layer_norm_native_layer_norm_backward_silu_sub_5 = async_compile.triton('triton_per_fused__to_copy_add_fill_mul_native_layer_norm_native_layer_norm_backward_silu_sub_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_fill_mul_native_layer_norm_native_layer_norm_backward_silu_sub_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_fill_mul_native_layer_norm_native_layer_norm_backward_silu_sub_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2048*r1), xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 + 2048*r1), xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0 + 2048*r1), xmask, other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = 1.0
    tmp5 = tmp4 - tmp3
    tmp6 = tmp2 * tmp5
    tmp7 = tmp6 + tmp4
    tmp8 = tmp3 * tmp7
    tmp9 = tmp1 * tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 - tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp20, xmask)
    tl.store(out_ptr1 + (x0), tmp24, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_karen/p2/cp2fzlfexxolpbr6duf6cj6tbjouwitd5aitsakycdnzf7lyvyqt.py
# Topologically Sorted Source Nodes: [queries], Original ATen: [aten._to_copy, aten.select_backward, aten.add, aten.silu, aten.div, aten.scalar_tensor, aten.ge, aten.where, aten.eq, aten.masked_fill, aten.mul, aten.sigmoid, aten.fill, aten.sub]
# Source node to ATen node mapping:
#   queries => convert_element_type_3, convert_element_type_4, mul, sigmoid
# Graph fragment:
#   %convert_element_type_32 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_5, torch.float32), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 32, 2048], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %convert_element_type_32, 0, 0), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_1, %select_scatter_default), kwargs = {})
#   %convert_element_type_3 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_3,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, %sigmoid), kwargs = {})
#   %convert_element_type_4 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_6, %expand), kwargs = {})
#   %convert_element_type_34 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_5, torch.bfloat16), kwargs = {})
#   %full_default_2 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%pow_2, 1e-08), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %sum_11, %full_default_2), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_4, %pow_2), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_2, 0), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default_2, %div_6), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %where_1), kwargs = {})
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_24, torch.bfloat16), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_34, %convert_element_type_35), kwargs = {})
#   %sigmoid_3 : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_1,), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 32, 2048], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_4, %sigmoid_3), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %sub_9), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_25, 1), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_3, %add_8), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, %mul_26), kwargs = {})
triton_poi_fused__to_copy_add_div_eq_fill_ge_masked_fill_mul_scalar_tensor_select_backward_sigmoid_silu_sub_where_6 = async_compile.triton('triton_poi_fused__to_copy_add_div_eq_fill_ge_masked_fill_mul_scalar_tensor_select_backward_sigmoid_silu_sub_where_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_eq_fill_ge_masked_fill_mul_scalar_tensor_select_backward_sigmoid_silu_sub_where_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_eq_fill_ge_masked_fill_mul_scalar_tensor_select_backward_sigmoid_silu_sub_where_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp1 == tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.0
    tmp6 = tl.where(tmp2, tmp4, tmp5)
    tmp7 = tmp0 + tmp6
    tmp9 = 1e-08
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = tmp7 / tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp8 >= tmp9
    tmp15 = tl.where(tmp13, tmp14, tmp5)
    tmp16 = tmp8 == tmp5
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22 / tmp8
    tmp24 = tl.where(tmp16, tmp5, tmp23)
    tmp25 = tmp15 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp12 + tmp26
    tmp28 = tl.sigmoid(tmp17)
    tmp29 = 1.0
    tmp30 = tmp29 - tmp28
    tmp31 = tmp17 * tmp30
    tmp32 = tmp31 + tmp29
    tmp33 = tmp28 * tmp32
    tmp34 = tmp27 * tmp33
    tl.store(in_out_ptr0 + (x2), tmp34, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_4, primals_5, primals_8, view, mm, pow_2, convert_element_type_7, mm_1, getitem_1, rsqrt, convert_element_type_13, addmm, getitem_3, rsqrt_1, permute_3, permute_10, permute_15, tangents_1 = args
    args.clear()
    assert_size_stride(primals_4, (2048, ), (1, ))
    assert_size_stride(primals_5, (2048, ), (1, ))
    assert_size_stride(primals_8, (2048, ), (1, ))
    assert_size_stride(view, (32, 2048), (2048, 1))
    assert_size_stride(mm, (32, 2048), (2048, 1))
    assert_size_stride(pow_2, (1, 1, 2048), (2048, 2048, 1))
    assert_size_stride(convert_element_type_7, (32, 2048), (2048, 1))
    assert_size_stride(mm_1, (32, 2048), (2048, 1))
    assert_size_stride(getitem_1, (32, 1), (1, 1))
    assert_size_stride(rsqrt, (32, 1), (1, 1))
    assert_size_stride(convert_element_type_13, (32, 2048), (2048, 1))
    assert_size_stride(addmm, (32, 2048), (2048, 1))
    assert_size_stride(getitem_3, (32, 1), (1, 1))
    assert_size_stride(rsqrt_1, (32, 1), (1, 1))
    assert_size_stride(permute_3, (2048, 2048), (2048, 1))
    assert_size_stride(permute_10, (2048, 2048), (2048, 1))
    assert_size_stride(permute_15, (2048, 2048), (2048, 1))
    assert_size_stride(tangents_1, (1, 32, 2048), (65536, 2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf4 = empty_strided_cuda((32, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.native_layer_norm_backward, aten._to_copy, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_native_layer_norm_backward_0.run(tangents_1, primals_8, addmm, getitem_3, rsqrt_1, buf4, 32, 2048, grid=grid(32), stream=stream0)
        del primals_8
        buf5 = empty_strided_cuda((32, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, permute_3, out=buf5)
        del permute_3
        buf9 = empty_strided_cuda((32, 2048), (2048, 1), torch.float32)
        buf14 = empty_strided_cuda((32, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._to_copy, aten.fill, aten.native_layer_norm, aten.silu, aten.sub, aten.mul, aten.add, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_fill_mul_native_layer_norm_native_layer_norm_backward_silu_sub_1.run(mm_1, getitem_1, rsqrt, primals_4, primals_5, buf5, buf9, buf14, 32, 2048, grid=grid(32), stream=stream0)
        del primals_4
        del primals_5
        buf16 = empty_strided_cuda((32, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf14, permute_10, out=buf16)
        del permute_10
        buf2 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf3 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf18 = empty_strided_cuda((1, 1, 2048), (2048, 2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, queries, queries_1], Original ATen: [aten._to_copy, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.add, aten.silu, aten.div, aten.neg, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_div_mul_native_layer_norm_native_layer_norm_backward_neg_select_backward_silu_sum_2.run(tangents_1, addmm, getitem_3, rsqrt_1, buf16, mm, pow_2, buf2, buf3, buf18, 2048, 32, grid=grid(2048), stream=stream0)
        del addmm
        del getitem_3
        del rsqrt_1
        buf6 = empty_strided_cuda((2048, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (2048, 32), (1, 2048), 0), convert_element_type_13, out=buf6)
        del convert_element_type_13
        buf7 = empty_strided_cuda((1, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_3.run(buf4, buf7, 2048, 32, grid=grid(2048), stream=stream0)
        del buf4
        buf8 = empty_strided_cuda((2048, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(buf6, buf8, 4194304, grid=grid(4194304), stream=stream0)
        buf12 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        buf13 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._to_copy, aten.fill, aten.native_layer_norm, aten.silu, aten.sub, aten.mul, aten.add, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_fill_mul_native_layer_norm_native_layer_norm_backward_silu_sub_5.run(buf5, buf9, mm_1, getitem_1, rsqrt, buf12, buf13, 2048, 32, grid=grid(2048), stream=stream0)
        del buf5
        del buf9
        del getitem_1
        del mm_1
        del rsqrt
        buf15 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (2048, 32), (1, 2048), 0), convert_element_type_7, out=buf15)
        del convert_element_type_7
        buf17 = empty_strided_cuda((2048, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(buf15, buf17, 4194304, grid=grid(4194304), stream=stream0)
        buf19 = reinterpret_tensor(buf16, (1, 32, 2048), (65536, 2048, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [queries], Original ATen: [aten._to_copy, aten.select_backward, aten.add, aten.silu, aten.div, aten.scalar_tensor, aten.ge, aten.where, aten.eq, aten.masked_fill, aten.mul, aten.sigmoid, aten.fill, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_div_eq_fill_ge_masked_fill_mul_scalar_tensor_select_backward_sigmoid_silu_sub_where_6.run(buf19, tangents_1, pow_2, buf18, mm, 65536, grid=grid(65536), stream=stream0)
        del buf18
        del mm
        del pow_2
        del tangents_1
        buf20 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (2048, 32), (1, 2048), 0), view, out=buf20)
        del view
        buf21 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (32, 2048), (2048, 1), 0), permute_15, out=buf21)
        del buf19
        del permute_15
        buf22 = empty_strided_cuda((2048, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(buf20, buf22, 4194304, grid=grid(4194304), stream=stream0)
        del buf20
    return (buf22, reinterpret_tensor(buf21, (1, 32, 2048), (65536, 2048, 1), 0), buf17, buf12, buf13, buf8, reinterpret_tensor(buf7, (2048, ), (1, ), 0), buf2, buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((32, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    mm = rand_strided((32, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    pow_2 = rand_strided((1, 1, 2048), (2048, 2048, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_7 = rand_strided((32, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_1 = rand_strided((32, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((32, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((32, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_13 = rand_strided((32, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    addmm = rand_strided((32, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_3 = rand_strided((32, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((32, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    permute_3 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_10 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_15 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((1, 32, 2048), (65536, 2048, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_4, primals_5, primals_8, view, mm, pow_2, convert_element_type_7, mm_1, getitem_1, rsqrt, convert_element_type_13, addmm, getitem_3, rsqrt_1, permute_3, permute_10, permute_15, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
