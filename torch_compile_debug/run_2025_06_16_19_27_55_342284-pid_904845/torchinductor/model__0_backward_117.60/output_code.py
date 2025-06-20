# AOT ID: ['0_backward']
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


# kernel path: /tmp/torchinductor_karen/ur/curmrftecmu7tffzejy7oyohyliszwtsxmqasqzaxth3gi4nxnkq.py
# Topologically Sorted Source Nodes: [type_as_7, x_11], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.div, aten.pow, aten.add]
# Source node to ATen node mapping:
#   type_as_7 => convert_element_type_30
#   x_11 => convert_element_type_29
# Graph fragment:
#   %convert_element_type_32 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tangents_1, torch.float32), kwargs = {})
#   %convert_element_type_30 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_14, torch.float32), kwargs = {})
#   %mul_14 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_32, %convert_element_type_30), kwargs = {})
#   %convert_element_type_29 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_19, torch.float32), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %convert_element_type_29), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %rsqrt_3), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_16, [2], True), kwargs = {dtype: torch.float32})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 2048), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_29, 1.0), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_6, 2.0), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %mul_20), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %mul_21), kwargs = {})
#   %convert_element_type_34 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.bfloat16), kwargs = {})
triton_red_fused__to_copy_add_div_mul_pow_sum_0 = async_compile.triton('triton_red_fused__to_copy_add_div_mul_pow_sum_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_pow_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 1, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_pow_sum_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp4 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp12 * tmp14
        tmp17 = tmp15 * tmp16
        tmp18 = -0.5
        tmp19 = tmp9 * tmp18
        tmp20 = tmp16 * tmp16
        tmp21 = tmp20 * tmp16
        tmp22 = tmp19 * tmp21
        tmp23 = 0.00048828125
        tmp24 = tmp22 * tmp23
        tmp26 = tmp25.to(tl.float32)
        tmp27 = 2.0
        tmp28 = tmp26 * tmp27
        tmp29 = tmp24 * tmp28
        tmp30 = tmp17 + tmp29
        tmp31 = tmp30.to(tl.float32)
        tl.store(out_ptr1 + (r1 + 2048*x0), tmp31, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_karen/qf/cqf7de5x7zhrbwwysnchcziyagqy6cky4jbjigatt3sidtavrjcd.py
# Topologically Sorted Source Nodes: [silu], Original ATen: [aten.silu, aten.mul, aten.sigmoid, aten.fill, aten.sub, aten.add]
# Source node to ATen node mapping:
#   silu => convert_element_type_23, convert_element_type_24, mul_10, sigmoid
# Graph fragment:
#   %convert_element_type_23 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_15, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_23,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_23, %sigmoid), kwargs = {})
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.bfloat16), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, %convert_element_type_24), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, %view_17), kwargs = {})
#   %sigmoid_1 : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_15,), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 128, 8192], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default, %sigmoid_1), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, %sub), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_24, 1), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_1, %add_10), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %mul_25), kwargs = {})
triton_poi_fused_add_fill_mul_sigmoid_silu_sub_1 = async_compile.triton('triton_poi_fused_add_fill_mul_sigmoid_silu_sub_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_silu_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_fill_mul_sigmoid_silu_sub_1(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tmp8 = tmp0 * tmp7
    tmp9 = tl.sigmoid(tmp1)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp1 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp8 * tmp14
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(in_out_ptr0 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_karen/vz/cvz4jxqjaep2geerfqfqqmtxfwyz7nejbpjtc4age7cmq5ms7xqj.py
# Topologically Sorted Source Nodes: [x_11, x_12, x_8, x_9], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.add]
# Source node to ATen node mapping:
#   x_11 => convert_element_type_29
#   x_12 => mul_12
#   x_8 => convert_element_type_18
#   x_9 => mul_8
# Graph fragment:
#   %convert_element_type_32 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tangents_1, torch.float32), kwargs = {})
#   %convert_element_type_29 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_19, torch.float32), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_29, %rsqrt_3), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_32, %mul_12), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_15, [0, 1], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_33 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_20, torch.bfloat16), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_1, %view_24), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %view_26), kwargs = {})
#   %convert_element_type_47 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_11, torch.float32), kwargs = {})
#   %convert_element_type_18 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_13, torch.float32), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_18, %rsqrt_2), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_47, %mul_8), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_28, [0, 1], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_48 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_27, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mul_sum_2 = async_compile.triton('triton_per_fused__to_copy_add_mul_sum_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_mul_sum_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2048*r1), xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 + 2048*r1), xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0 + 2048*r1), xmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr4 + (x0 + 2048*r1), xmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (x0 + 2048*r1), xmask, other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr6 + (r1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp12 = tmp0 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp15 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp10.to(tl.float32)
    tmp26 = tmp24.to(tl.float32)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp26, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_karen/6y/c6yscyv4k6iask3tdlpyastgi3viqmmh2hv3tgitfwpz7junlv4z.py
# Topologically Sorted Source Nodes: [type_as_6, x_8], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum, aten.div, aten.pow]
# Source node to ATen node mapping:
#   type_as_6 => convert_element_type_19
#   x_8 => convert_element_type_18
# Graph fragment:
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_1, %view_24), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %view_26), kwargs = {})
#   %convert_element_type_47 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_11, torch.float32), kwargs = {})
#   %convert_element_type_19 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_10, torch.float32), kwargs = {})
#   %mul_27 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_47, %convert_element_type_19), kwargs = {})
#   %convert_element_type_18 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_13, torch.float32), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, %convert_element_type_18), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, %rsqrt_2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_29, [2], True), kwargs = {dtype: torch.float32})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_1, 2048), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_18, 1.0), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_8, 2.0), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %mul_33), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %mul_34), kwargs = {})
#   %convert_element_type_49 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.bfloat16), kwargs = {})
triton_red_fused__to_copy_add_div_mul_pow_sum_3 = async_compile.triton('triton_red_fused__to_copy_add_div_mul_pow_sum_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_pow_sum_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 1, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_pow_sum_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_out_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp5 * tmp7
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp8 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp24 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_out_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tmp15 + tmp16
        tmp19 = tmp17 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 * tmp22
        tmp25 = tmp23 * tmp24
        tmp26 = -0.5
        tmp27 = tmp13 * tmp26
        tmp28 = tmp24 * tmp24
        tmp29 = tmp28 * tmp24
        tmp30 = tmp27 * tmp29
        tmp31 = 0.00048828125
        tmp32 = tmp30 * tmp31
        tmp34 = tmp33.to(tl.float32)
        tmp35 = 2.0
        tmp36 = tmp34 * tmp35
        tmp37 = tmp32 * tmp36
        tmp38 = tmp25 + tmp37
        tmp39 = tmp38.to(tl.float32)
        tl.store(in_out_ptr0 + (r1 + 2048*x0), tmp39, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_karen/bb/cbbysllekxbfuyienfnxmaxt37w3f4jcbwynqbd3rzzlu3prxe7l.py
# Topologically Sorted Source Nodes: [type_as_1, x_3, type_as, x], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.div, aten.pow, aten.add]
# Source node to ATen node mapping:
#   type_as => convert_element_type_7
#   type_as_1 => convert_element_type_10
#   x => convert_element_type_6
#   x_3 => convert_element_type_9
# Graph fragment:
#   %convert_element_type_58 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_36, torch.float32), kwargs = {})
#   %convert_element_type_10 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_6, torch.float32), kwargs = {})
#   %mul_39 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_58, %convert_element_type_10), kwargs = {})
#   %convert_element_type_9 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.float32), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_39, %convert_element_type_9), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_39, %rsqrt_1), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_41, [2], True), kwargs = {dtype: torch.float32})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_2, 2048), kwargs = {})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_9, 1.0), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_10, 2.0), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %mul_45), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_42, %mul_46), kwargs = {})
#   %convert_element_type_60 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_15, torch.bfloat16), kwargs = {})
#   %convert_element_type_61 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_37, torch.float32), kwargs = {})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_5, torch.float32), kwargs = {})
#   %mul_47 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_61, %convert_element_type_7), kwargs = {})
#   %convert_element_type_6 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_47, %convert_element_type_6), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_47, %rsqrt), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_49, [2], True), kwargs = {dtype: torch.float32})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_3, 2048), kwargs = {})
#   %pow_12 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_6, 1.0), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_12, 2.0), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %mul_53), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %mul_54), kwargs = {})
#   %convert_element_type_63 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_16, torch.bfloat16), kwargs = {})
triton_red_fused__to_copy_add_div_mul_pow_sum_4 = async_compile.triton('triton_red_fused__to_copy_add_div_mul_pow_sum_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*bf16', 'in_ptr8': '*bf16', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*bf16', 'out_ptr5': '*bf16', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_pow_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 22, 'num_reduction': 2, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_pow_sum_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr3, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp54 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr2 + (128*x0 + ((r1 % 128))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp41 = tl.load(in_ptr3 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp47 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp50 = tl.load(in_ptr5 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp0 = (r1 % 128)
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 64, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (64 + 128*(r1 // 128) + 2048*x0 + ((r1 % 128))), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.load(in_ptr1 + (64 + 128*x0 + ((r1 % 128))), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
        tmp10 = tl.where(tmp4, tmp8, tmp9)
        tmp11 = tmp0 >= tmp3
        tmp12 = tl.full([1, 1], 128, tl.int64)
        tmp13 = tmp0 < tmp12
        tmp14 = tl.load(in_ptr0 + (128*(r1 // 128) + 2048*x0 + ((-64) + ((r1 % 128)))), rmask & tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (128*x0 + ((-64) + ((r1 % 128)))), rmask & tmp11 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp15 * tmp16
        tmp18 = -tmp17
        tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
        tmp20 = tl.where(tmp11, tmp18, tmp19)
        tmp21 = tl.where(tmp4, tmp10, tmp20)
        tmp23 = tmp22.to(tl.float32)
        tmp25 = tmp23 * tmp24
        tmp26 = tmp21 + tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tl.load(in_ptr3 + (64 + 128*(r1 // 128) + 2048*x0 + ((r1 % 128))), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 * tmp7
        tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
        tmp33 = tl.where(tmp4, tmp31, tmp32)
        tmp34 = tl.load(in_ptr3 + (128*(r1 // 128) + 2048*x0 + ((-64) + ((r1 % 128)))), rmask & tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp35 * tmp16
        tmp37 = -tmp36
        tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
        tmp39 = tl.where(tmp11, tmp37, tmp38)
        tmp40 = tl.where(tmp4, tmp33, tmp39)
        tmp42 = tmp41.to(tl.float32)
        tmp43 = tmp42 * tmp24
        tmp44 = tmp40 + tmp43
        tmp45 = tmp44.to(tl.float32)
        tmp46 = tmp45.to(tl.float32)
        tmp48 = tmp47.to(tl.float32)
        tmp49 = tmp28 * tmp48
        tmp51 = tmp50.to(tl.float32)
        tmp52 = tmp49 * tmp51
        tmp53 = tl.broadcast_to(tmp52, [XBLOCK, RBLOCK])
        tmp55 = _tmp54 + tmp53
        _tmp54 = tl.where(rmask & xmask, tmp55, _tmp54)
        tl.store(out_ptr0 + (r1 + 2048*x0), tmp28, rmask & xmask)
        tl.store(out_ptr1 + (r1 + 2048*x0), tmp46, rmask & xmask)
    tmp54 = tl.sum(_tmp54, 1)[:, None]
    tmp60 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp56 = tl.load(out_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp57 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp69 = tl.load(in_ptr5 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp58 = tmp57.to(tl.float32)
        tmp59 = tmp56 * tmp58
        tmp61 = tmp59 * tmp60
        tmp62 = -0.5
        tmp63 = tmp54 * tmp62
        tmp64 = tmp60 * tmp60
        tmp65 = tmp64 * tmp60
        tmp66 = tmp63 * tmp65
        tmp67 = 0.00048828125
        tmp68 = tmp66 * tmp67
        tmp70 = tmp69.to(tl.float32)
        tmp71 = 2.0
        tmp72 = tmp70 * tmp71
        tmp73 = tmp68 * tmp72
        tmp74 = tmp61 + tmp73
        tmp75 = tmp74.to(tl.float32)
        tl.store(out_ptr3 + (r1 + 2048*x0), tmp75, rmask & xmask)
    _tmp84 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp76 = tl.load(out_ptr1 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp77 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp80 = tl.load(in_ptr8 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp78 = tmp77.to(tl.float32)
        tmp79 = tmp76 * tmp78
        tmp81 = tmp80.to(tl.float32)
        tmp82 = tmp79 * tmp81
        tmp83 = tl.broadcast_to(tmp82, [XBLOCK, RBLOCK])
        tmp85 = _tmp84 + tmp83
        _tmp84 = tl.where(rmask & xmask, tmp85, _tmp84)
    tmp84 = tl.sum(_tmp84, 1)[:, None]
    tmp90 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp86 = tl.load(out_ptr1 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp87 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp99 = tl.load(in_ptr8 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp88 = tmp87.to(tl.float32)
        tmp89 = tmp86 * tmp88
        tmp91 = tmp89 * tmp90
        tmp92 = -0.5
        tmp93 = tmp84 * tmp92
        tmp94 = tmp90 * tmp90
        tmp95 = tmp94 * tmp90
        tmp96 = tmp93 * tmp95
        tmp97 = 0.00048828125
        tmp98 = tmp96 * tmp97
        tmp100 = tmp99.to(tl.float32)
        tmp101 = 2.0
        tmp102 = tmp100 * tmp101
        tmp103 = tmp98 * tmp102
        tmp104 = tmp91 + tmp103
        tmp105 = tmp104.to(tl.float32)
        tl.store(out_ptr5 + (r1 + 2048*x0), tmp105, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_karen/ta/ctalhwooa4glurfqgxpocdlnnbihuvp3k52bu7otbl5bj724a4pc.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   x_3 => convert_element_type_9
#   x_4 => mul_2
# Graph fragment:
#   %convert_element_type_9 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.float32), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_9, %rsqrt_1), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_58, %mul_2), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_40, [0, 1], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_59 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_38, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_mul_sum_5 = async_compile.triton('triton_per_fused__to_copy_mul_sum_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_mul_sum_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_mul_sum_5(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    RBLOCK: tl.constexpr = 128
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
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_karen/vr/cvr3qbtept4owbtprfgqddoaryxbpaqb7hufhdf6owujcl54tax7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_1, %view_24), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %view_26), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_45), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %view_47), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %view_49), kwargs = {})
triton_poi_fused_add_6 = async_compile.triton('triton_poi_fused_add_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tl.store(in_out_ptr0 + (x0), tmp10, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_5, primals_6, primals_7, primals_8, primals_10, primals_14, view, mm, mm_1, rsqrt, rsqrt_1, view_8, convert_element_type_14, convert_element_type_15, getitem_4, getitem_5, getitem_7, mm_3, rsqrt_2, view_14, mm_4, mm_5, view_18, mm_6, rsqrt_3, permute_9, permute_13, permute_18, permute_22, permute_30, permute_34, permute_38, tangents_1 = args
    args.clear()
    assert_size_stride(primals_5, (2048, ), (1, ))
    assert_size_stride(primals_6, (2048, ), (1, ))
    assert_size_stride(primals_7, (128, 128), (128, 1))
    assert_size_stride(primals_8, (128, 128), (128, 1))
    assert_size_stride(primals_10, (2048, ), (1, ))
    assert_size_stride(primals_14, (2048, ), (1, ))
    assert_size_stride(view, (128, 2048), (2048, 1))
    assert_size_stride(mm, (128, 2048), (2048, 1))
    assert_size_stride(mm_1, (128, 2048), (2048, 1))
    assert_size_stride(rsqrt, (1, 128, 1), (128, 1, 1))
    assert_size_stride(rsqrt_1, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_8, (1, 128, 16, 128), (262144, 2048, 128, 1))
    assert_size_stride(convert_element_type_14, (1, 128, 16, 128), (262144, 2048, 128, 1))
    assert_size_stride(convert_element_type_15, (1, 128, 16, 128), (262144, 2048, 128, 1))
    assert_size_stride(getitem_4, (1, 128, 16, 128), (262144, 2048, 128, 1))
    assert_size_stride(getitem_5, (1, 16, 128), (2048, 128, 1))
    assert_size_stride(getitem_7, (2, ), (1, ))
    assert_size_stride(mm_3, (128, 2048), (2048, 1))
    assert_size_stride(rsqrt_2, (1, 128, 1), (128, 1, 1))
    assert_size_stride(view_14, (128, 2048), (2048, 1))
    assert_size_stride(mm_4, (128, 8192), (8192, 1))
    assert_size_stride(mm_5, (128, 8192), (8192, 1))
    assert_size_stride(view_18, (128, 8192), (8192, 1))
    assert_size_stride(mm_6, (128, 2048), (2048, 1))
    assert_size_stride(rsqrt_3, (1, 128, 1), (128, 1, 1))
    assert_size_stride(permute_9, (2048, 8192), (8192, 1))
    assert_size_stride(permute_13, (8192, 2048), (2048, 1))
    assert_size_stride(permute_18, (8192, 2048), (2048, 1))
    assert_size_stride(permute_22, (2048, 2048), (2048, 1))
    assert_size_stride(permute_30, (2048, 2048), (2048, 1))
    assert_size_stride(permute_34, (2048, 2048), (2048, 1))
    assert_size_stride(permute_38, (2048, 2048), (2048, 1))
    assert_size_stride(tangents_1, (1, 128, 2048), (262144, 2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((1, 128, 2048), (262144, 2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [type_as_7, x_11], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.div, aten.pow, aten.add]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_0.run(tangents_1, primals_14, mm_6, rsqrt_3, buf3, 128, 2048, grid=grid(128), stream=stream0)
        del primals_14
        buf5 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (128, 2048), (2048, 1), 0), permute_9, out=buf5)
        del permute_9
        buf6 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
        buf9 = reinterpret_tensor(mm_5, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_5  # reuse
        # Topologically Sorted Source Nodes: [silu], Original ATen: [aten.silu, aten.mul, aten.sigmoid, aten.fill, aten.sub, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_1.run(buf9, buf5, mm_4, buf6, 1048576, grid=grid(1048576), stream=stream0)
        del buf5
        del mm_4
        buf11 = empty_strided_cuda((128, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (128, 8192), (8192, 1), 0), permute_18, out=buf11)
        del permute_18
        buf8 = empty_strided_cuda((128, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf6, (128, 8192), (8192, 1), 0), permute_13, out=buf8)
        del permute_13
        buf1 = empty_strided_cuda((2048, ), (1, ), torch.bfloat16)
        buf13 = empty_strided_cuda((2048, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_11, x_12, x_8, x_9], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mul_sum_2.run(tangents_1, mm_6, rsqrt_3, buf8, buf11, mm_3, rsqrt_2, buf1, buf13, 2048, 128, grid=grid(2048), stream=stream0)
        del mm_6
        del rsqrt_3
        buf4 = empty_strided_cuda((2048, 8192), (8192, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (2048, 128), (1, 2048), 0), view_18, out=buf4)
        del view_18
        buf7 = empty_strided_cuda((8192, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf6, (8192, 128), (1, 8192), 0), view_14, out=buf7)
        del buf6
        buf10 = empty_strided_cuda((8192, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (8192, 128), (1, 8192), 0), view_14, out=buf10)
        del buf9
        del view_14
        buf15 = reinterpret_tensor(mm_3, (1, 128, 2048), (262144, 2048, 1), 0); del mm_3  # reuse
        # Topologically Sorted Source Nodes: [type_as_6, x_8], Original ATen: [aten.add, aten._to_copy, aten.mul, aten.sum, aten.div, aten.pow]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_3.run(buf15, tangents_1, buf8, buf11, primals_10, rsqrt_2, 128, 2048, grid=grid(128), stream=stream0)
        del primals_10
        del rsqrt_2
        buf16 = empty_strided_cuda((2048, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (2048, 128), (1, 2048), 0), reinterpret_tensor(getitem_4, (128, 2048), (2048, 1), 0), out=buf16)
        buf17 = reinterpret_tensor(buf3, (128, 2048), (2048, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (128, 2048), (2048, 1), 0), permute_22, out=buf17)
        del permute_22
        buf18 = reinterpret_tensor(buf15, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf15  # reuse
        buf19 = empty_strided_cuda((1, 128, 16, 128), (262144, 2048, 128, 1), torch.bfloat16)
        buf20 = empty_strided_cuda((1, 128, 16, 128), (262144, 2048, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf21 = torch.ops.flash_attn._flash_attn_backward.default(reinterpret_tensor(buf17, (1, 128, 16, 128), (262144, 2048, 128, 1), 0), convert_element_type_14, convert_element_type_15, view_8, getitem_4, getitem_5, buf18, buf19, buf20, 0.0, 0.08838834764831845, True, 64, 0, 0.0, None, False, getitem_7)
        del convert_element_type_14
        del convert_element_type_15
        del getitem_4
        del getitem_5
        del getitem_7
        del view_8
        buf25 = buf21
        del buf21
        del buf25
        buf26 = empty_strided_cuda((1, 128, 2048), (262144, 2048, 1), torch.float32)
        buf30 = empty_strided_cuda((1, 128, 2048), (262144, 2048, 1), torch.float32)
        buf36 = reinterpret_tensor(buf17, (1, 128, 2048), (262144, 2048, 1), 0); del buf17  # reuse
        buf39 = empty_strided_cuda((1, 128, 2048), (262144, 2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [type_as_1, x_3, type_as, x], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.div, aten.pow, aten.add]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mul_pow_sum_4.run(buf19, primals_7, primals_8, buf18, primals_6, mm_1, rsqrt_1, primals_5, mm, rsqrt, buf26, buf30, buf36, buf39, 128, 2048, grid=grid(128), stream=stream0)
        del buf18
        del primals_5
        del primals_6
        del primals_7
        del primals_8
        buf28 = empty_strided_cuda((2048, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_mul_sum_5.run(buf26, mm_1, rsqrt_1, buf28, 2048, 128, grid=grid(2048), stream=stream0)
        del buf26
        del mm_1
        del rsqrt_1
        buf32 = empty_strided_cuda((2048, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_mul_sum_5.run(buf30, mm, rsqrt, buf32, 2048, 128, grid=grid(2048), stream=stream0)
        del buf30
        del mm
        del rsqrt
        buf34 = empty_strided_cuda((2048, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (2048, 128), (1, 2048), 0), view, out=buf34)
        buf35 = reinterpret_tensor(buf19, (128, 2048), (2048, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (128, 2048), (2048, 1), 0), permute_30, out=buf35)
        del permute_30
        buf37 = empty_strided_cuda((2048, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (2048, 128), (1, 2048), 0), view, out=buf37)
        buf38 = reinterpret_tensor(buf20, (128, 2048), (2048, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (128, 2048), (2048, 1), 0), permute_34, out=buf38)
        del permute_34
        buf40 = empty_strided_cuda((2048, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (2048, 128), (1, 2048), 0), view, out=buf40)
        del view
        buf41 = reinterpret_tensor(buf36, (128, 2048), (2048, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (128, 2048), (2048, 1), 0), permute_38, out=buf41)
        del buf39
        del permute_38
        buf42 = reinterpret_tensor(buf8, (1, 128, 2048), (262144, 2048, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_6.run(buf42, tangents_1, buf11, buf35, buf38, buf41, 262144, grid=grid(262144), stream=stream0)
        del buf11
        del buf35
        del buf38
        del buf41
        del tangents_1
    return (buf42, buf40, buf37, buf34, buf32, buf28, None, None, buf16, buf13, buf10, buf7, buf4, buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_5 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_14 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    view = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    mm = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_1 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_8 = rand_strided((1, 128, 16, 128), (262144, 2048, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_14 = rand_strided((1, 128, 16, 128), (262144, 2048, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_15 = rand_strided((1, 128, 16, 128), (262144, 2048, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_4 = rand_strided((1, 128, 16, 128), (262144, 2048, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_5 = rand_strided((1, 16, 128), (2048, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.int64)
    mm_3 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_2 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_14 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_4 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_5 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_18 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_6 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_3 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_9 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_13 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_18 = rand_strided((8192, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_22 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_30 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_34 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_38 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((1, 128, 2048), (262144, 2048, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_5, primals_6, primals_7, primals_8, primals_10, primals_14, view, mm, mm_1, rsqrt, rsqrt_1, view_8, convert_element_type_14, convert_element_type_15, getitem_4, getitem_5, getitem_7, mm_3, rsqrt_2, view_14, mm_4, mm_5, view_18, mm_6, rsqrt_3, permute_9, permute_13, permute_18, permute_22, permute_30, permute_34, permute_38, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
