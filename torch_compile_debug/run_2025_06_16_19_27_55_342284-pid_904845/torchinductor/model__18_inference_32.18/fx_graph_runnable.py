
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
        select = torch.ops.aten.select.int(arg3_1, 0, 0);  arg3_1 = None
        mul = torch.ops.aten.mul.Tensor(select, 1.0);  select = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        select_1 = torch.ops.aten.select.int(arg4_1, 0, 0);  arg4_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(select_1, arg2_1);  select_1 = arg2_1 = None
        add = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        sub = torch.ops.aten.sub.Tensor(add, arg0_1);  add = arg0_1 = None
        clamp_min = torch.ops.aten.clamp_min.default(sub, -100.0);  sub = None
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 100.0);  clamp_min = None
        return (clamp_max,)
        
def load_args(reader):
    buf0 = reader.storage(None, 4*s1*s2*(s0 - 1) + 4*s1*(s2 - 1) + 4*s1, device=device(type='cuda', index=0))
    reader.tensor(buf0, (2048, 2048), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf1, (2048, 2048), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf2, (2048, 2048), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1,), (36,), storage_offset=35, is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1,), is_leaf=True)  # arg4_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)