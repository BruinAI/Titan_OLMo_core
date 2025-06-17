
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

    
    
    def forward(self, primals_5, primals_6, primals_7, primals_8, primals_10, primals_14, view, mm, mm_1, rsqrt, rsqrt_1, view_8, convert_element_type_14, convert_element_type_15, getitem_4, getitem_5, getitem_7, mm_3, rsqrt_2, view_14, mm_4, mm_5, view_18, mm_6, rsqrt_3, permute_9, permute_13, permute_18, permute_22, permute_30, permute_34, permute_38, tangents_1):
        convert_element_type_32 = torch.ops.prims.convert_element_type.default(tangents_1, torch.float32)
        convert_element_type_30 = torch.ops.prims.convert_element_type.default(primals_14, torch.float32);  primals_14 = None
        mul_14 = torch.ops.aten.mul.Tensor(convert_element_type_32, convert_element_type_30);  convert_element_type_30 = None
        view_19 = torch.ops.aten.view.default(mm_6, [1, 128, 2048]);  mm_6 = None
        convert_element_type_29 = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_12 = torch.ops.aten.mul.Tensor(convert_element_type_29, rsqrt_3)
        mul_15 = torch.ops.aten.mul.Tensor(convert_element_type_32, mul_12);  convert_element_type_32 = mul_12 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(mul_15, [0, 1], True, dtype = torch.float32);  mul_15 = None
        view_20 = torch.ops.aten.view.default(sum_1, [2048]);  sum_1 = None
        convert_element_type_33 = torch.ops.prims.convert_element_type.default(view_20, torch.bfloat16);  view_20 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_14, convert_element_type_29)
        mul_17 = torch.ops.aten.mul.Tensor(mul_14, rsqrt_3);  mul_14 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_16, [2], True, dtype = torch.float32);  mul_16 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(rsqrt_3, 3);  rsqrt_3 = None
        mul_18 = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, pow_5);  mul_18 = pow_5 = None
        expand = torch.ops.aten.expand.default(mul_19, [1, 128, 2048]);  mul_19 = None
        div = torch.ops.aten.div.Scalar(expand, 2048);  expand = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_29, 1.0);  convert_element_type_29 = None
        mul_20 = torch.ops.aten.mul.Scalar(pow_6, 2.0);  pow_6 = None
        mul_21 = torch.ops.aten.mul.Tensor(div, mul_20);  div = mul_20 = None
        add_8 = torch.ops.aten.add.Tensor(mul_17, mul_21);  mul_17 = mul_21 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None
        view_21 = torch.ops.aten.view.default(convert_element_type_34, [128, 2048]);  convert_element_type_34 = None
        permute_7 = torch.ops.aten.permute.default(view_21, [1, 0])
        mm_7 = torch.ops.aten.mm.default(permute_7, view_18);  permute_7 = view_18 = None
        mm_8 = torch.ops.aten.mm.default(view_21, permute_9);  view_21 = permute_9 = None
        view_22 = torch.ops.aten.view.default(mm_8, [1, 128, 8192]);  mm_8 = None
        view_15 = torch.ops.aten.view.default(mm_4, [1, 128, 8192]);  mm_4 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(view_15, torch.float32)
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type_23)
        mul_10 = torch.ops.aten.mul.Tensor(convert_element_type_23, sigmoid);  convert_element_type_23 = sigmoid = None
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        mul_22 = torch.ops.aten.mul.Tensor(view_22, convert_element_type_24);  convert_element_type_24 = None
        view_17 = torch.ops.aten.view.default(mm_5, [1, 128, 8192]);  mm_5 = None
        mul_23 = torch.ops.aten.mul.Tensor(view_22, view_17);  view_22 = view_17 = None
        view_23 = torch.ops.aten.view.default(mul_22, [128, 8192]);  mul_22 = None
        permute_11 = torch.ops.aten.permute.default(view_23, [1, 0])
        mm_9 = torch.ops.aten.mm.default(permute_11, view_14);  permute_11 = None
        mm_10 = torch.ops.aten.mm.default(view_23, permute_13);  view_23 = permute_13 = None
        view_24 = torch.ops.aten.view.default(mm_10, [1, 128, 2048]);  mm_10 = None
        add_9 = torch.ops.aten.add.Tensor(tangents_1, view_24);  tangents_1 = view_24 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(view_15)
        full_default = torch.ops.aten.full.default([1, 128, 8192], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        sub = torch.ops.aten.sub.Tensor(full_default, sigmoid_1);  full_default = None
        mul_24 = torch.ops.aten.mul.Tensor(view_15, sub);  view_15 = sub = None
        add_10 = torch.ops.aten.add.Scalar(mul_24, 1);  mul_24 = None
        mul_25 = torch.ops.aten.mul.Tensor(sigmoid_1, add_10);  sigmoid_1 = add_10 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_23, mul_25);  mul_23 = mul_25 = None
        view_25 = torch.ops.aten.view.default(mul_26, [128, 8192]);  mul_26 = None
        permute_16 = torch.ops.aten.permute.default(view_25, [1, 0])
        mm_11 = torch.ops.aten.mm.default(permute_16, view_14);  permute_16 = view_14 = None
        mm_12 = torch.ops.aten.mm.default(view_25, permute_18);  view_25 = permute_18 = None
        view_26 = torch.ops.aten.view.default(mm_12, [1, 128, 2048]);  mm_12 = None
        add_11 = torch.ops.aten.add.Tensor(add_9, view_26);  add_9 = view_26 = None
        convert_element_type_47 = torch.ops.prims.convert_element_type.default(add_11, torch.float32)
        convert_element_type_19 = torch.ops.prims.convert_element_type.default(primals_10, torch.float32);  primals_10 = None
        mul_27 = torch.ops.aten.mul.Tensor(convert_element_type_47, convert_element_type_19);  convert_element_type_19 = None
        view_13 = torch.ops.aten.view.default(mm_3, [1, 128, 2048]);  mm_3 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        mul_8 = torch.ops.aten.mul.Tensor(convert_element_type_18, rsqrt_2)
        mul_28 = torch.ops.aten.mul.Tensor(convert_element_type_47, mul_8);  convert_element_type_47 = mul_8 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_28, [0, 1], True, dtype = torch.float32);  mul_28 = None
        view_27 = torch.ops.aten.view.default(sum_3, [2048]);  sum_3 = None
        convert_element_type_48 = torch.ops.prims.convert_element_type.default(view_27, torch.bfloat16);  view_27 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_27, convert_element_type_18)
        mul_30 = torch.ops.aten.mul.Tensor(mul_27, rsqrt_2);  mul_27 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(mul_29, [2], True, dtype = torch.float32);  mul_29 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(rsqrt_2, 3);  rsqrt_2 = None
        mul_31 = torch.ops.aten.mul.Scalar(sum_4, -0.5);  sum_4 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, pow_7);  mul_31 = pow_7 = None
        expand_1 = torch.ops.aten.expand.default(mul_32, [1, 128, 2048]);  mul_32 = None
        div_1 = torch.ops.aten.div.Scalar(expand_1, 2048);  expand_1 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_18, 1.0);  convert_element_type_18 = None
        mul_33 = torch.ops.aten.mul.Scalar(pow_8, 2.0);  pow_8 = None
        mul_34 = torch.ops.aten.mul.Tensor(div_1, mul_33);  div_1 = mul_33 = None
        add_12 = torch.ops.aten.add.Tensor(mul_30, mul_34);  mul_30 = mul_34 = None
        convert_element_type_49 = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16);  add_12 = None
        view_28 = torch.ops.aten.view.default(convert_element_type_49, [128, 2048]);  convert_element_type_49 = None
        permute_20 = torch.ops.aten.permute.default(view_28, [1, 0])
        view_11 = torch.ops.aten.view.default(getitem_4, [1, 128, -1])
        view_12 = torch.ops.aten.view.default(view_11, [128, 2048]);  view_11 = None
        mm_13 = torch.ops.aten.mm.default(permute_20, view_12);  permute_20 = view_12 = None
        mm_14 = torch.ops.aten.mm.default(view_28, permute_22);  view_28 = permute_22 = None
        view_29 = torch.ops.aten.view.default(mm_14, [1, 128, 2048]);  mm_14 = None
        view_30 = torch.ops.aten.view.default(view_29, [1, 128, 16, 128]);  view_29 = None
        empty_1 = torch.ops.aten.empty.memory_format([1, 128, 16, 128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_24 = torch.ops.aten.permute.default(empty_1, [0, 1, 2, 3]);  empty_1 = None
        empty_2 = torch.ops.aten.empty.memory_format([1, 128, 16, 128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_25 = torch.ops.aten.permute.default(empty_2, [0, 1, 2, 3]);  empty_2 = None
        empty_3 = torch.ops.aten.empty.memory_format([1, 128, 16, 128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_26 = torch.ops.aten.permute.default(empty_3, [0, 1, 2, 3]);  empty_3 = None
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.flash_attn._flash_attn_backward.default, dout = view_30, q = convert_element_type_14, k = convert_element_type_15, v = view_8, out = getitem_4, softmax_lse = getitem_5, dropout_p = 0.0, softmax_scale = 0.08838834764831845, causal = True, window_size_left = 64, window_size_right = 0, softcap = 0.0, alibi_slopes = None, deterministic = False, rng_state = getitem_7, _dq_base_index = 0, _dk_base_index = 1, _dv_base_index = 2, _all_bases = [permute_24, permute_25, permute_26]);  view_30 = convert_element_type_14 = convert_element_type_15 = view_8 = getitem_4 = getitem_5 = getitem_7 = permute_24 = permute_25 = permute_26 = None
        getitem_9 = auto_functionalized_v2[1]
        getitem_10 = auto_functionalized_v2[2]
        getitem_11 = auto_functionalized_v2[3];  auto_functionalized_v2 = None
        convert_element_type_54 = torch.ops.prims.convert_element_type.default(getitem_10, torch.float32);  getitem_10 = None
        convert_element_type_55 = torch.ops.prims.convert_element_type.default(getitem_9, torch.float32);  getitem_9 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_7, 0);  primals_7 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        mul_35 = torch.ops.aten.mul.Tensor(convert_element_type_54, unsqueeze_1)
        slice_9 = torch.ops.aten.slice.Tensor(mul_35, 3, 0, 64)
        slice_10 = torch.ops.aten.slice.Tensor(mul_35, 3, 64, 128);  mul_35 = None
        neg_2 = torch.ops.aten.neg.default(slice_9);  slice_9 = None
        cat_2 = torch.ops.aten.cat.default([slice_10, neg_2], 3);  slice_10 = neg_2 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(primals_8, 0);  primals_8 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, 2);  unsqueeze_2 = None
        mul_36 = torch.ops.aten.mul.Tensor(convert_element_type_54, unsqueeze_3);  convert_element_type_54 = None
        add_13 = torch.ops.aten.add.Tensor(cat_2, mul_36);  cat_2 = mul_36 = None
        mul_37 = torch.ops.aten.mul.Tensor(convert_element_type_55, unsqueeze_1);  unsqueeze_1 = None
        slice_11 = torch.ops.aten.slice.Tensor(mul_37, 3, 0, 64)
        slice_12 = torch.ops.aten.slice.Tensor(mul_37, 3, 64, 128);  mul_37 = None
        neg_3 = torch.ops.aten.neg.default(slice_11);  slice_11 = None
        cat_3 = torch.ops.aten.cat.default([slice_12, neg_3], 3);  slice_12 = neg_3 = None
        mul_38 = torch.ops.aten.mul.Tensor(convert_element_type_55, unsqueeze_3);  convert_element_type_55 = unsqueeze_3 = None
        add_14 = torch.ops.aten.add.Tensor(cat_3, mul_38);  cat_3 = mul_38 = None
        convert_element_type_56 = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None
        convert_element_type_57 = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None
        view_36 = torch.ops.aten.view.default(convert_element_type_56, [1, 128, 2048]);  convert_element_type_56 = None
        view_37 = torch.ops.aten.view.default(convert_element_type_57, [1, 128, 2048]);  convert_element_type_57 = None
        convert_element_type_58 = torch.ops.prims.convert_element_type.default(view_36, torch.float32);  view_36 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(primals_6, torch.float32);  primals_6 = None
        mul_39 = torch.ops.aten.mul.Tensor(convert_element_type_58, convert_element_type_10);  convert_element_type_10 = None
        view_3 = torch.ops.aten.view.default(mm_1, [1, 128, 2048]);  mm_1 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_9, rsqrt_1)
        mul_40 = torch.ops.aten.mul.Tensor(convert_element_type_58, mul_2);  convert_element_type_58 = mul_2 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_40, [0, 1], True, dtype = torch.float32);  mul_40 = None
        view_38 = torch.ops.aten.view.default(sum_5, [2048]);  sum_5 = None
        convert_element_type_59 = torch.ops.prims.convert_element_type.default(view_38, torch.bfloat16);  view_38 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_39, convert_element_type_9)
        mul_42 = torch.ops.aten.mul.Tensor(mul_39, rsqrt_1);  mul_39 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(mul_41, [2], True, dtype = torch.float32);  mul_41 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(rsqrt_1, 3);  rsqrt_1 = None
        mul_43 = torch.ops.aten.mul.Scalar(sum_6, -0.5);  sum_6 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_43, pow_9);  mul_43 = pow_9 = None
        expand_2 = torch.ops.aten.expand.default(mul_44, [1, 128, 2048]);  mul_44 = None
        div_2 = torch.ops.aten.div.Scalar(expand_2, 2048);  expand_2 = None
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_9, 1.0);  convert_element_type_9 = None
        mul_45 = torch.ops.aten.mul.Scalar(pow_10, 2.0);  pow_10 = None
        mul_46 = torch.ops.aten.mul.Tensor(div_2, mul_45);  div_2 = mul_45 = None
        add_15 = torch.ops.aten.add.Tensor(mul_42, mul_46);  mul_42 = mul_46 = None
        convert_element_type_60 = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None
        convert_element_type_61 = torch.ops.prims.convert_element_type.default(view_37, torch.float32);  view_37 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(primals_5, torch.float32);  primals_5 = None
        mul_47 = torch.ops.aten.mul.Tensor(convert_element_type_61, convert_element_type_7);  convert_element_type_7 = None
        view_1 = torch.ops.aten.view.default(mm, [1, 128, 2048]);  mm = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type_6, rsqrt)
        mul_48 = torch.ops.aten.mul.Tensor(convert_element_type_61, mul);  convert_element_type_61 = mul = None
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_48, [0, 1], True, dtype = torch.float32);  mul_48 = None
        view_39 = torch.ops.aten.view.default(sum_7, [2048]);  sum_7 = None
        convert_element_type_62 = torch.ops.prims.convert_element_type.default(view_39, torch.bfloat16);  view_39 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_47, convert_element_type_6)
        mul_50 = torch.ops.aten.mul.Tensor(mul_47, rsqrt);  mul_47 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(mul_49, [2], True, dtype = torch.float32);  mul_49 = None
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_51 = torch.ops.aten.mul.Scalar(sum_8, -0.5);  sum_8 = None
        mul_52 = torch.ops.aten.mul.Tensor(mul_51, pow_11);  mul_51 = pow_11 = None
        expand_3 = torch.ops.aten.expand.default(mul_52, [1, 128, 2048]);  mul_52 = None
        div_3 = torch.ops.aten.div.Scalar(expand_3, 2048);  expand_3 = None
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 1.0);  convert_element_type_6 = None
        mul_53 = torch.ops.aten.mul.Scalar(pow_12, 2.0);  pow_12 = None
        mul_54 = torch.ops.aten.mul.Tensor(div_3, mul_53);  div_3 = mul_53 = None
        add_16 = torch.ops.aten.add.Tensor(mul_50, mul_54);  mul_50 = mul_54 = None
        convert_element_type_63 = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16);  add_16 = None
        view_41 = torch.ops.aten.view.default(getitem_11, [1, 128, 2048]);  getitem_11 = None
        view_42 = torch.ops.aten.view.default(view_41, [128, 2048]);  view_41 = None
        permute_28 = torch.ops.aten.permute.default(view_42, [1, 0])
        mm_15 = torch.ops.aten.mm.default(permute_28, view);  permute_28 = None
        mm_16 = torch.ops.aten.mm.default(view_42, permute_30);  view_42 = permute_30 = None
        view_45 = torch.ops.aten.view.default(mm_16, [1, 128, 2048]);  mm_16 = None
        add_17 = torch.ops.aten.add.Tensor(add_11, view_45);  add_11 = view_45 = None
        view_46 = torch.ops.aten.view.default(convert_element_type_60, [128, 2048]);  convert_element_type_60 = None
        permute_32 = torch.ops.aten.permute.default(view_46, [1, 0])
        mm_17 = torch.ops.aten.mm.default(permute_32, view);  permute_32 = None
        mm_18 = torch.ops.aten.mm.default(view_46, permute_34);  view_46 = permute_34 = None
        view_47 = torch.ops.aten.view.default(mm_18, [1, 128, 2048]);  mm_18 = None
        add_18 = torch.ops.aten.add.Tensor(add_17, view_47);  add_17 = view_47 = None
        view_48 = torch.ops.aten.view.default(convert_element_type_63, [128, 2048]);  convert_element_type_63 = None
        permute_36 = torch.ops.aten.permute.default(view_48, [1, 0])
        mm_19 = torch.ops.aten.mm.default(permute_36, view);  permute_36 = view = None
        mm_20 = torch.ops.aten.mm.default(view_48, permute_38);  view_48 = permute_38 = None
        view_49 = torch.ops.aten.view.default(mm_20, [1, 128, 2048]);  mm_20 = None
        add_19 = torch.ops.aten.add.Tensor(add_18, view_49);  add_18 = view_49 = None
        return (add_19, mm_19, mm_17, mm_15, convert_element_type_62, convert_element_type_59, None, None, mm_13, convert_element_type_48, mm_11, mm_9, mm_7, convert_element_type_33)
        
def load_args(reader):
    buf0 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (2048,), dtype=torch.bfloat16, is_leaf=True)  # primals_5
    buf1 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (2048,), dtype=torch.bfloat16, is_leaf=True)  # primals_6
    buf2 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf2, (128, 128), is_leaf=True)  # primals_7
    buf3 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf3, (128, 128), is_leaf=True)  # primals_8
    buf4 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (2048,), dtype=torch.bfloat16, is_leaf=True)  # primals_10
    buf5 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (2048,), dtype=torch.bfloat16, is_leaf=True)  # primals_14
    buf6 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (128, 2048), dtype=torch.bfloat16, is_leaf=True)  # view
    buf7 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (128, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm
    buf8 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf8, (128, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_1
    buf9 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf9, (1, 128, 1), is_leaf=True)  # rsqrt
    buf10 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf10, (1, 128, 1), is_leaf=True)  # rsqrt_1
    buf11 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf11, (1, 128, 16, 128), dtype=torch.bfloat16, is_leaf=True)  # view_8
    buf12 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf12, (1, 128, 16, 128), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_14
    buf13 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf13, (1, 128, 16, 128), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_15
    buf14 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf14, (1, 128, 16, 128), dtype=torch.bfloat16, is_leaf=True)  # getitem_4
    buf15 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf15, (1, 16, 128), is_leaf=True)  # getitem_5
    buf16 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf16, (2,), dtype=torch.int64, is_leaf=True)  # getitem_7
    buf17 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf17, (128, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_3
    buf18 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf18, (1, 128, 1), is_leaf=True)  # rsqrt_2
    buf19 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf19, (128, 2048), dtype=torch.bfloat16, is_leaf=True)  # view_14
    buf20 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf20, (128, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_4
    buf21 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf21, (128, 8192), dtype=torch.bfloat16, is_leaf=True)  # mm_5
    buf22 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf22, (128, 8192), dtype=torch.bfloat16, is_leaf=True)  # view_18
    buf23 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf23, (128, 2048), dtype=torch.bfloat16, is_leaf=True)  # mm_6
    buf24 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf24, (1, 128, 1), is_leaf=True)  # rsqrt_3
    buf25 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf25, (2048, 8192), dtype=torch.bfloat16, is_leaf=True)  # permute_9
    buf26 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf26, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # permute_13
    buf27 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf27, (8192, 2048), dtype=torch.bfloat16, is_leaf=True)  # permute_18
    buf28 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf28, (2048, 2048), dtype=torch.bfloat16, is_leaf=True)  # permute_22
    buf29 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf29, (2048, 2048), dtype=torch.bfloat16, is_leaf=True)  # permute_30
    buf30 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf30, (2048, 2048), dtype=torch.bfloat16, is_leaf=True)  # permute_34
    buf31 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf31, (2048, 2048), dtype=torch.bfloat16, is_leaf=True)  # permute_38
    buf32 = reader.storage(None, 524288, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf32, (1, 128, 2048), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)