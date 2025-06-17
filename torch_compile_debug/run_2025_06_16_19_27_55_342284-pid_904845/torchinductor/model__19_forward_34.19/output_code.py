# AOT ID: ['19_forward']
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


# kernel path: /tmp/torchinductor_karen/gt/cgt7pxuuspgj2w2hmst5ux2ozo5ukoevnrhalm46emozgjsutrjb.py
# Topologically Sorted Source Nodes: [sum_1], Original ATen: [aten.sum]
# Source node to ATen node mapping:
#   sum_1 => sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%primals_23, [-1]), kwargs = {dtype: torch.float32})
triton_red_fused_sum_0 = async_compile.triton('triton_red_fused_sum_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 36
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 2048*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_karen/f3/cf3onyffs5vvrrd3jxrsoby3dixqduzodooeskk4qyjmiisejvqd.py
# Topologically Sorted Source Nodes: [mse], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mse => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_1,), kwargs = {})
triton_per_fused_mean_1 = async_compile.triton('triton_per_fused_mean_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_1(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 36.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_karen/kc/ckckfdouzblosl66bgvso574ufcmdejum6ehswp2kdjmiyl3xzif.py
# Topologically Sorted Source Nodes: [mul_1, new_surprise_1, clamp_1], Original ATen: [aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   clamp_1 => clamp_max_1, clamp_min_1
#   mul_1 => mul_1
#   new_surprise_1 => sub_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %primals_7), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_1, %primals_5), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_1, -100.0), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 100.0), kwargs = {})
triton_poi_fused_clamp_mul_sub_2 = async_compile.triton('triton_poi_fused_clamp_mul_sub_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_mul_sub_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_mul_sub_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp4 = tl.load(in_ptr2 + (x0), xmask)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 - tmp4
    tmp6 = -100.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 100.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_karen/ee/ceewnyu5dkv7qmdwnjgrr4tne66b4hfe7yagstghzqtbb44clhfo.py
# Topologically Sorted Source Nodes: [mul, new_surprise, clamp], Original ATen: [aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   clamp => clamp_max, clamp_min
#   mul => mul
#   new_surprise => sub
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %primals_3), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %primals_1), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, -100.0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 100.0), kwargs = {})
triton_poi_fused_clamp_mul_sub_3 = async_compile.triton('triton_poi_fused_clamp_mul_sub_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_mul_sub_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '845BD750D40B8118FA308A2BADCFC77F52F612CF9AC4B9D320C19C4934EE1026', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_mul_sub_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (x0), None)
    tmp4 = tl.load(in_ptr2 + (x0), None)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 - tmp4
    tmp6 = -100.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 100.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (x0), tmp9, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23 = args
    args.clear()
    assert_size_stride(primals_1, (2048, 2048), (2048, 1))
    assert_size_stride(primals_2, (2048, 2048), (2048, 1))
    assert_size_stride(primals_3, (2048, 2048), (2048, 1))
    assert_size_stride(primals_4, (1, ), (36, ))
    assert_size_stride(primals_5, (2048, ), (1, ))
    assert_size_stride(primals_6, (2048, ), (1, ))
    assert_size_stride(primals_7, (2048, ), (1, ))
    assert_size_stride(primals_8, (2048, ), (1, ))
    assert_size_stride(primals_9, (2048, ), (1, ))
    assert_size_stride(primals_10, (2048, ), (1, ))
    assert_size_stride(primals_11, (2048, 2048), (2048, 1))
    assert_size_stride(primals_12, (2048, 2048), (2048, 1))
    assert_size_stride(primals_13, (2048, 2048), (2048, 1))
    assert_size_stride(primals_14, (2048, ), (1, ))
    assert_size_stride(primals_15, (2048, ), (1, ))
    assert_size_stride(primals_16, (2048, ), (1, ))
    assert_size_stride(primals_17, (2048, ), (1, ))
    assert_size_stride(primals_18, (2048, ), (1, ))
    assert_size_stride(primals_19, (2048, ), (1, ))
    assert_size_stride(primals_20, (2048, ), (1, ))
    assert_size_stride(primals_21, (2048, ), (1, ))
    assert_size_stride(primals_22, (2048, ), (1, ))
    assert_size_stride(primals_23, (1, 36, 2048), (73728, 2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf7 = empty_strided_cuda((1, 36), (36, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sum_1], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_0.run(primals_23, buf7, 36, 2048, grid=grid(36), stream=stream0)
        del primals_23
        buf8 = empty_strided_cuda((), (), torch.float32)
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [mse], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_1.run(buf9, buf7, 1, 36, grid=grid(1), stream=stream0)
        del buf7
        buf1 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1, new_surprise_1, clamp_1], Original ATen: [aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_mul_sub_2.run(primals_4, primals_7, primals_5, buf1, 2048, grid=grid(2048), stream=stream0)
        del primals_5
        del primals_7
        buf2 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mul_2, new_surprise_2, clamp_2], Original ATen: [aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_mul_sub_2.run(primals_4, primals_10, primals_8, buf2, 2048, grid=grid(2048), stream=stream0)
        del primals_10
        del primals_8
        buf4 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mul_4, new_surprise_4, clamp_4], Original ATen: [aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_mul_sub_2.run(primals_4, primals_16, primals_14, buf4, 2048, grid=grid(2048), stream=stream0)
        del primals_14
        del primals_16
        buf5 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mul_5, new_surprise_5, clamp_5], Original ATen: [aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_mul_sub_2.run(primals_4, primals_19, primals_17, buf5, 2048, grid=grid(2048), stream=stream0)
        del primals_17
        del primals_19
        buf6 = empty_strided_cuda((2048, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mul_6, new_surprise_6, clamp_6], Original ATen: [aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_mul_sub_2.run(primals_4, primals_22, primals_20, buf6, 2048, grid=grid(2048), stream=stream0)
        del primals_20
        del primals_22
        buf0 = empty_strided_cuda((2048, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, new_surprise, clamp], Original ATen: [aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_mul_sub_3.run(primals_4, primals_3, primals_1, buf0, 4194304, grid=grid(4194304), stream=stream0)
        del primals_1
        del primals_3
        buf3 = empty_strided_cuda((2048, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_3, new_surprise_3, clamp_3], Original ATen: [aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_mul_sub_3.run(primals_4, primals_13, primals_11, buf3, 4194304, grid=grid(4194304), stream=stream0)
        del primals_11
        del primals_13
        del primals_4
    return (buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1, ), (36, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((1, 36, 2048), (73728, 2048, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
