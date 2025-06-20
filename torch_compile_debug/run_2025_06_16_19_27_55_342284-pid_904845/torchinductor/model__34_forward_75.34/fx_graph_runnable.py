
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9):
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        permute = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        view = torch.ops.aten.view.default(convert_element_type_1, [32, 2048]);  convert_element_type_1 = None
        mm = torch.ops.aten.mm.default(view, permute)
        view_1 = torch.ops.aten.view.default(mm, [1, 32, 2048])
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type_4)
        mul = torch.ops.aten.mul.Tensor(convert_element_type_4, sigmoid);  convert_element_type_4 = sigmoid = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 2.0);  convert_element_type_6 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(pow_1, [1], True);  pow_1 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
        clamp_min = torch.ops.aten.clamp_min.default(pow_2, 1e-08)
        expand = torch.ops.aten.expand.default(clamp_min, [1, 32, 2048]);  clamp_min = None
        div = torch.ops.aten.div.Tensor(convert_element_type_5, expand);  convert_element_type_5 = expand = None
        select = torch.ops.aten.select.int(div, 0, 0)
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(select, torch.bfloat16);  select = None
        permute_1 = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        mm_1 = torch.ops.aten.mm.default(convert_element_type_8, permute_1)
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(mm_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_11, [1], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        sub = torch.ops.aten.sub.Tensor(convert_element_type_11, getitem_1);  convert_element_type_11 = None
        mul_1 = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, primals_4);  mul_1 = None
        add_1 = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(add_1)
        mul_3 = torch.ops.aten.mul.Tensor(add_1, sigmoid_1);  add_1 = sigmoid_1 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        permute_2 = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        addmm = torch.ops.aten.addmm.default(convert_element_type_12, convert_element_type_14, permute_2);  convert_element_type_12 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_18, [1], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_2 = torch.ops.aten.add.Tensor(getitem_2, 0.0001);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_18, getitem_3);  convert_element_type_18 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, primals_8);  mul_4 = None
        add_3 = torch.ops.aten.add.Tensor(mul_5, primals_9);  mul_5 = primals_9 = None
        view_2 = torch.ops.aten.view.default(add_3, [1, 32, 2048]);  add_3 = None
        add_4 = torch.ops.aten.add.Tensor(view_2, div);  view_2 = div = None
        permute_3 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_10 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        permute_15 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (add_4, primals_4, primals_5, primals_8, view, mm, pow_2, convert_element_type_8, mm_1, getitem_1, rsqrt, convert_element_type_14, addmm, getitem_3, rsqrt_1, permute_3, permute_10, permute_15)
        
def load_args(reader):
    buf0 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf0, (2048, 2048), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 8192*s0, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1, 32, 2048), (262144, 2048, 1), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf2, (2048, 2048), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf3, (2048,), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf4, (2048,), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 16777216, device=device(type='cuda', index=0))
    reader.tensor(buf5, (2048, 2048), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf6, (2048,), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf7, (2048,), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf8, (2048,), is_leaf=True)  # primals_9
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)