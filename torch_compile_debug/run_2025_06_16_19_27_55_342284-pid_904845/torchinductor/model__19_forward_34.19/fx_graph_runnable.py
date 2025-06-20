
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
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23):
        select = torch.ops.aten.select.int(primals_4, 0, 0);  primals_4 = None
        mul = torch.ops.aten.mul.Tensor(select, primals_3);  primals_3 = None
        sub = torch.ops.aten.sub.Tensor(mul, primals_1);  mul = primals_1 = None
        clamp_min = torch.ops.aten.clamp_min.default(sub, -100.0);  sub = None
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 100.0);  clamp_min = None
        mul_1 = torch.ops.aten.mul.Tensor(select, primals_7);  primals_7 = None
        sub_1 = torch.ops.aten.sub.Tensor(mul_1, primals_5);  mul_1 = primals_5 = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(sub_1, -100.0);  sub_1 = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_1, 100.0);  clamp_min_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(select, primals_10);  primals_10 = None
        sub_2 = torch.ops.aten.sub.Tensor(mul_2, primals_8);  mul_2 = primals_8 = None
        clamp_min_2 = torch.ops.aten.clamp_min.default(sub_2, -100.0);  sub_2 = None
        clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 100.0);  clamp_min_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(select, primals_13);  primals_13 = None
        sub_3 = torch.ops.aten.sub.Tensor(mul_3, primals_11);  mul_3 = primals_11 = None
        clamp_min_3 = torch.ops.aten.clamp_min.default(sub_3, -100.0);  sub_3 = None
        clamp_max_3 = torch.ops.aten.clamp_max.default(clamp_min_3, 100.0);  clamp_min_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(select, primals_16);  primals_16 = None
        sub_4 = torch.ops.aten.sub.Tensor(mul_4, primals_14);  mul_4 = primals_14 = None
        clamp_min_4 = torch.ops.aten.clamp_min.default(sub_4, -100.0);  sub_4 = None
        clamp_max_4 = torch.ops.aten.clamp_max.default(clamp_min_4, 100.0);  clamp_min_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(select, primals_19);  primals_19 = None
        sub_5 = torch.ops.aten.sub.Tensor(mul_5, primals_17);  mul_5 = primals_17 = None
        clamp_min_5 = torch.ops.aten.clamp_min.default(sub_5, -100.0);  sub_5 = None
        clamp_max_5 = torch.ops.aten.clamp_max.default(clamp_min_5, 100.0);  clamp_min_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(select, primals_22);  select = primals_22 = None
        sub_6 = torch.ops.aten.sub.Tensor(mul_6, primals_20);  mul_6 = primals_20 = None
        clamp_min_6 = torch.ops.aten.clamp_min.default(sub_6, -100.0);  sub_6 = None
        clamp_max_6 = torch.ops.aten.clamp_max.default(clamp_min_6, 100.0);  clamp_min_6 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(primals_23, [-1], dtype = torch.float32);  primals_23 = None
        mean = torch.ops.aten.mean.default(sum_1);  sum_1 = None
        return (clamp_max, clamp_max_1, clamp_max_2, clamp_max_3, clamp_max_4, clamp_max_5, clamp_max_6, mean)
        
def load_args(reader):
    buf0 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf0, (2048, 2048), storage_offset=4194304, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf1, (2048, 2048), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf2, (2048, 2048), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1,), (36,), storage_offset=35, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf4, (2048,), storage_offset=2048, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf5, (2048,), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf6, (2048,), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf7, (2048,), storage_offset=2048, is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf8, (2048,), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf9, (2048,), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf10, (2048, 2048), storage_offset=4194304, is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf11, (2048, 2048), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf12, (2048, 2048), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf13, (2048,), storage_offset=2048, is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf14, (2048,), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf15, (2048,), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf16, (2048,), storage_offset=2048, is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf17, (2048,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf18, (2048,), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf19, (2048,), storage_offset=2048, is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf20, (2048,), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf21, (2048,), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf22, (1, 36, 2048), is_leaf=True)  # primals_23
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)