
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.traceable_tensor_subclasses = set()
torch._dynamo.config.allowed_functions_module_string_ignorelist = {'torch.distributions', 'torch._decomp', 'torch._prims', 'torch.testing', 'torch._refs'}
torch._dynamo.config.optimize_ddp = 'ddp_optimizer'
torch._dynamo.config._ddp_optimization_mode = ['ddp_optimizer', 'python_reducer', 'python_reducer_without_compiled_forward', 'no_optimization']
torch._dynamo.config._save_config_ignore = {'constant_functions', 'skipfiles_inline_module_allowlist', 'repro_after', 'repro_level'}
torch._dynamo.config.reorderable_logging_functions = set()
torch._dynamo.config.ignore_logger_methods = set()
torch._dynamo.config._autograd_backward_strict_mode_banned_ops = ['stride', 'requires_grad', 'storage_offset', 'layout', 'data', 'is_coalesced', 'is_complex', 'is_conj', 'is_contiguous', 'is_cpu', 'is_cuda', 'is_distributed', 'is_floating_point', 'is_inference', 'is_ipu', 'is_leaf', 'is_maia', 'is_meta', 'is_mkldnn', 'is_mps', 'is_mtia', 'is_neg', 'is_nested', 'is_nonzero', 'is_pinned', 'is_quantized', 'is_same_size', 'is_set_to', 'is_shared', 'is_signed', 'is_sparse', 'is_sparse_csr', 'is_vulkan', 'is_xla', 'is_xpu']
torch._dynamo.config.compiled_autograd_kwargs_override = {}
torch._inductor.config.pre_grad_fusion_options = {}
torch._inductor.config.post_grad_fusion_options = {}
torch._inductor.config.fx_passes_numeric_check = {'pre_grad': False, 'precision': 0.0001, 'num_iterations': 1, 'requires_optimizer': True}
torch._inductor.config.reorder_for_compute_comm_overlap_passes = ['reorder_compute_for_overlap', 'sink_waits', 'raise_comms']
torch._inductor.config._fuse_ddp_communication_passes = ['fuse_ddp_with_concat_op', 'schedule_comm_wait']
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.aot_inductor.metadata = {}
torch._inductor.config.aot_inductor.presets = {}
torch._inductor.config.rocm.arch = []
torch._inductor.config.rocm.ck_supported_arch = ['gfx90a', 'gfx940', 'gfx941', 'gfx942']
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config._save_config_ignore = ['trace.upload_tar', 'joint_custom_pre_pass', 'joint_custom_post_pass', 'pre_grad_custom_pass']
torch._inductor.config._cache_config_ignore_prefix = ['trace', 'cuda.cutlass_dir', 'worker_start_method', 'compile_threads', 'post_grad_custom_post_pass', 'post_grad_custom_pre_pass', 'always_complex_memory_overlap_TESTING_ONLY']
torch._inductor.config.external_matmul = []
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = False



isolate_fails_code_str = None




# torch version: 2.6.0+cu124
# torch cuda version: 12.4
# torch git version: 2236df1770800ffea5697b11b0bb0d910b2e59e1


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Thu_Mar_28_02:18:24_PDT_2024 
# Cuda compilation tools, release 12.4, V12.4.131 
# Build cuda_12.4.r12.4/compiler.34097967_0 

# GPU Hardware Info: 
# NVIDIA RTX A6000 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_4, primals_5, primals_8, view, mm, pow_2, convert_element_type_7, mm_1, getitem_1, rsqrt, convert_element_type_13, addmm, getitem_3, rsqrt_1, permute_3, permute_10, permute_15, tangents_1):
        select_1 = torch.ops.aten.select.int(tangents_1, 0, 0)
        mul_7 = torch.ops.aten.mul.Tensor(select_1, primals_8);  primals_8 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, 2048)
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_7, [1], True)
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_17, getitem_3);  convert_element_type_17 = getitem_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_7, mul_4);  mul_7 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_9, [1], True);  mul_9 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_4, sum_3);  sum_3 = None
        sub_3 = torch.ops.aten.sub.Tensor(mul_8, sum_2);  mul_8 = sum_2 = None
        sub_4 = torch.ops.aten.sub.Tensor(sub_3, mul_10);  sub_3 = mul_10 = None
        div_1 = torch.ops.aten.div.Tensor(rsqrt_1, 2048);  rsqrt_1 = None
        mul_11 = torch.ops.aten.mul.Tensor(div_1, sub_4);  div_1 = sub_4 = None
        mul_12 = torch.ops.aten.mul.Tensor(select_1, mul_4);  mul_4 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(mul_12, [0]);  mul_12 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(select_1, [0]);  select_1 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        mm_2 = torch.ops.aten.mm.default(convert_element_type_18, permute_3);  permute_3 = None
        permute_4 = torch.ops.aten.permute.default(convert_element_type_18, [1, 0])
        mm_3 = torch.ops.aten.mm.default(permute_4, convert_element_type_13);  permute_4 = convert_element_type_13 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(convert_element_type_18, [0], True, dtype = torch.float32);  convert_element_type_18 = None
        view_3 = torch.ops.aten.view.default(sum_6, [2048]);  sum_6 = None
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        full_default = torch.ops.aten.full.default([32, 2048], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        sub = torch.ops.aten.sub.Tensor(convert_element_type_10, getitem_1);  convert_element_type_10 = getitem_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, primals_4)
        add_1 = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = primals_5 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(add_1)
        sub_5 = torch.ops.aten.sub.Tensor(full_default, sigmoid_1);  full_default = None
        mul_13 = torch.ops.aten.mul.Tensor(add_1, sub_5);  add_1 = sub_5 = None
        add_5 = torch.ops.aten.add.Scalar(mul_13, 1);  mul_13 = None
        mul_14 = torch.ops.aten.mul.Tensor(sigmoid_1, add_5);  sigmoid_1 = add_5 = None
        mul_15 = torch.ops.aten.mul.Tensor(convert_element_type_24, mul_14);  convert_element_type_24 = mul_14 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_15, primals_4);  primals_4 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, 2048)
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_17, [1], True)
        mul_19 = torch.ops.aten.mul.Tensor(mul_17, mul_1);  mul_17 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(mul_19, [1], True);  mul_19 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_1, sum_8);  sum_8 = None
        sub_7 = torch.ops.aten.sub.Tensor(mul_18, sum_7);  mul_18 = sum_7 = None
        sub_8 = torch.ops.aten.sub.Tensor(sub_7, mul_20);  sub_7 = mul_20 = None
        div_2 = torch.ops.aten.div.Tensor(rsqrt, 2048);  rsqrt = None
        mul_21 = torch.ops.aten.mul.Tensor(div_2, sub_8);  div_2 = sub_8 = None
        mul_22 = torch.ops.aten.mul.Tensor(mul_15, mul_1);  mul_1 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_22, [0]);  mul_22 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(mul_15, [0]);  mul_15 = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        permute_8 = torch.ops.aten.permute.default(convert_element_type_27, [1, 0])
        mm_4 = torch.ops.aten.mm.default(permute_8, convert_element_type_7);  permute_8 = convert_element_type_7 = None
        mm_5 = torch.ops.aten.mm.default(convert_element_type_27, permute_10);  convert_element_type_27 = permute_10 = None
        convert_element_type_32 = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_33 = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        full_default_1 = torch.ops.aten.full.default([1, 32, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter = torch.ops.aten.select_scatter.default(full_default_1, convert_element_type_32, 0, 0);  full_default_1 = convert_element_type_32 = None
        add_6 = torch.ops.aten.add.Tensor(tangents_1, select_scatter);  tangents_1 = select_scatter = None
        view_1 = torch.ops.aten.view.default(mm, [1, 32, 2048]);  mm = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(view_1, torch.float32)
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type_3)
        mul = torch.ops.aten.mul.Tensor(convert_element_type_3, sigmoid);  convert_element_type_3 = sigmoid = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        clamp_min = torch.ops.aten.clamp_min.default(pow_2, 1e-08)
        expand = torch.ops.aten.expand.default(clamp_min, [1, 32, 2048]);  clamp_min = None
        div = torch.ops.aten.div.Tensor(convert_element_type_4, expand)
        div_4 = torch.ops.aten.div.Tensor(div, expand);  div = None
        neg = torch.ops.aten.neg.default(add_6)
        mul_23 = torch.ops.aten.mul.Tensor(neg, div_4);  neg = div_4 = None
        div_5 = torch.ops.aten.div.Tensor(add_6, expand);  add_6 = expand = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(mul_23, [1], True, dtype = torch.float32);  mul_23 = None
        full_default_2 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ge = torch.ops.aten.ge.Scalar(pow_2, 1e-08)
        where = torch.ops.aten.where.self(ge, sum_11, full_default_2);  ge = sum_11 = None
        div_6 = torch.ops.aten.div.Tensor(convert_element_type_4, pow_2);  convert_element_type_4 = None
        eq = torch.ops.aten.eq.Scalar(pow_2, 0);  pow_2 = None
        where_1 = torch.ops.aten.where.self(eq, full_default_2, div_6);  eq = full_default_2 = div_6 = None
        mul_24 = torch.ops.aten.mul.Tensor(where, where_1);  where = where_1 = None
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(mul_24, torch.bfloat16);  mul_24 = None
        add_7 = torch.ops.aten.add.Tensor(convert_element_type_34, convert_element_type_35);  convert_element_type_34 = convert_element_type_35 = None
        sigmoid_3 = torch.ops.aten.sigmoid.default(view_1)
        full_default_4 = torch.ops.aten.full.default([1, 32, 2048], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        sub_9 = torch.ops.aten.sub.Tensor(full_default_4, sigmoid_3);  full_default_4 = None
        mul_25 = torch.ops.aten.mul.Tensor(view_1, sub_9);  view_1 = sub_9 = None
        add_8 = torch.ops.aten.add.Scalar(mul_25, 1);  mul_25 = None
        mul_26 = torch.ops.aten.mul.Tensor(sigmoid_3, add_8);  sigmoid_3 = add_8 = None
        mul_27 = torch.ops.aten.mul.Tensor(add_7, mul_26);  add_7 = mul_26 = None
        view_4 = torch.ops.aten.view.default(mul_27, [32, 2048]);  mul_27 = None
        permute_13 = torch.ops.aten.permute.default(view_4, [1, 0])
        mm_6 = torch.ops.aten.mm.default(permute_13, view);  permute_13 = view = None
        mm_7 = torch.ops.aten.mm.default(view_4, permute_15);  view_4 = permute_15 = None
        view_5 = torch.ops.aten.view.default(mm_7, [1, 32, 2048]);  mm_7 = None
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        return (convert_element_type_40, view_5, convert_element_type_33, sum_9, sum_10, convert_element_type_25, convert_element_type_default, sum_4, sum_5)
        
def load_args(reader):
    buf0 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf0, (2048,), is_leaf=True)  # primals_4
    buf1 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf1, (2048,), is_leaf=True)  # primals_5
    buf2 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf2, (2048,), is_leaf=True)  # primals_8
    buf3 = reader.storage(None, 4096*s0, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (32, 2048), dtype=torch.bfloat16, is_leaf=True)  # view
    buf4 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (32, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm
    buf5 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf5, (1, 1, 2048), is_leaf=True)  # pow_2
    buf6 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (32, 2048), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_7
    buf7 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (32, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_1
    buf8 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf8, (32, 1), is_leaf=True)  # getitem_1
    buf9 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf9, (32, 1), is_leaf=True)  # rsqrt
    buf10 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf10, (32, 2048), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_13
    buf11 = reader.storage(None, 131072, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf11, (32, 2048), dtype=torch.bfloat16, is_leaf=True)  # addmm
    buf12 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf12, (32, 1), is_leaf=True)  # getitem_3
    buf13 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf13, (32, 1), is_leaf=True)  # rsqrt_1
    buf14 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf14, (2048, 2048), dtype=torch.bfloat16, is_leaf=True)  # permute_3
    buf15 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf15, (2048, 2048), dtype=torch.bfloat16, is_leaf=True)  # permute_10
    buf16 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf16, (2048, 2048), dtype=torch.bfloat16, is_leaf=True)  # permute_15
    buf17 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf17, (1, 32, 2048), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)