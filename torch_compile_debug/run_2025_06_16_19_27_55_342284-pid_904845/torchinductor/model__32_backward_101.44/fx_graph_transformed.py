class GraphModule(torch.nn.Module):
    def forward(self, primals_5: "bf16[2048]", primals_6: "bf16[2048]", primals_7: "f32[128, 128]", primals_8: "f32[128, 128]", primals_10: "bf16[2048]", primals_14: "bf16[2048]", view: "bf16[128, 2048]", mm: "bf16[128, 2048]", mm_1: "bf16[128, 2048]", rsqrt: "f32[1, 128, 1]", rsqrt_1: "f32[1, 128, 1]", view_8: "bf16[1, 128, 16, 128]", convert_element_type_17: "bf16[1, 128, 16, 128]", convert_element_type_18: "bf16[1, 128, 16, 128]", getitem_4: "bf16[1, 128, 16, 128]", getitem_5: "f32[1, 16, 128]", getitem_7: "i64[2]", mm_3: "bf16[128, 2048]", rsqrt_2: "f32[1, 128, 1]", view_14: "bf16[128, 2048]", mm_4: "bf16[128, 8192]", mm_5: "bf16[128, 8192]", view_18: "bf16[128, 8192]", mm_6: "bf16[128, 2048]", rsqrt_3: "f32[1, 128, 1]", permute_9: "bf16[2048, 8192]", permute_13: "bf16[8192, 2048]", permute_18: "bf16[8192, 2048]", permute_22: "bf16[2048, 2048]", permute_30: "bf16[2048, 2048]", permute_34: "bf16[2048, 2048]", permute_38: "bf16[2048, 2048]", tangents_1: "f32[1, 128, 2048]"):
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:212 in forward, code: x = self.weight.type_as(x) * x
        convert_element_type_35: "f32[2048]" = torch.ops.prims.convert_element_type.default(primals_14, torch.float32);  primals_14 = None
        mul_14: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(tangents_1, convert_element_type_35);  convert_element_type_35 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/feed_forward.py:128 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        view_19: "bf16[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_6, [1, 128, 2048]);  mm_6 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:203 in forward, code: x = x.float()
        convert_element_type_34: "f32[1, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:206 in forward, code: x = x * torch.rsqrt(variance + self.eps)
        mul_12: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_34, rsqrt_3)
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:212 in forward, code: x = self.weight.type_as(x) * x
        mul_15: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(tangents_1, mul_12);  mul_12 = None
        sum_1: "f32[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_15, [0, 1], True, dtype = torch.float32);  mul_15 = None
        view_20: "f32[2048]" = torch.ops.aten.reshape.default(sum_1, [2048]);  sum_1 = None
        convert_element_type_39: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_20, torch.bfloat16);  view_20 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:206 in forward, code: x = x * torch.rsqrt(variance + self.eps)
        mul_16: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_14, convert_element_type_34)
        mul_17: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_14, rsqrt_3);  mul_14 = None
        sum_2: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_16, [2], True, dtype = torch.float32);  mul_16 = None
        pow_5: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_3, 3);  rsqrt_3 = None
        mul_18: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None
        mul_19: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_18, pow_5);  mul_18 = pow_5 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:205 in forward, code: variance = x.pow(2).mean(-1, keepdim=True)
        expand: "f32[1, 128, 2048]" = torch.ops.aten.expand.default(mul_19, [1, 128, 2048]);  mul_19 = None
        div: "f32[1, 128, 2048]" = torch.ops.aten.div.Scalar(expand, 2048);  expand = None
        pow_6: "f32[1, 128, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_34, 1.0);  convert_element_type_34 = None
        mul_20: "f32[1, 128, 2048]" = torch.ops.aten.mul.Scalar(pow_6, 2.0);  pow_6 = None
        mul_21: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div, mul_20);  div = mul_20 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:205 in forward, code: variance = x.pow(2).mean(-1, keepdim=True)
        add_8: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_17, mul_21);  mul_17 = mul_21 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:203 in forward, code: x = x.float()
        convert_element_type_40: "bf16[1, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/feed_forward.py:128 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        view_21: "bf16[128, 2048]" = torch.ops.aten.reshape.default(convert_element_type_40, [128, 2048]);  convert_element_type_40 = None
        permute_7: "bf16[2048, 128]" = torch.ops.aten.permute.default(view_21, [1, 0])
        mm_7: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_7, view_18);  permute_7 = view_18 = None
        mm_8: "bf16[128, 8192]" = torch.ops.aten.mm.default(view_21, permute_9);  view_21 = permute_9 = None
        view_22: "bf16[1, 128, 8192]" = torch.ops.aten.reshape.default(mm_8, [1, 128, 8192]);  mm_8 = None
        view_15: "bf16[1, 128, 8192]" = torch.ops.aten.reshape.default(mm_4, [1, 128, 8192]);  mm_4 = None
        convert_element_type_27: "f32[1, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_15, torch.float32)
        sigmoid: "f32[1, 128, 8192]" = torch.ops.aten.sigmoid.default(convert_element_type_27)
        mul_10: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_27, sigmoid);  convert_element_type_27 = sigmoid = None
        convert_element_type_28: "bf16[1, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        mul_22: "bf16[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_22, convert_element_type_28);  convert_element_type_28 = None
        view_17: "bf16[1, 128, 8192]" = torch.ops.aten.reshape.default(mm_5, [1, 128, 8192]);  mm_5 = None
        mul_23: "bf16[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_22, view_17);  view_22 = view_17 = None
        view_23: "bf16[128, 8192]" = torch.ops.aten.reshape.default(mul_22, [128, 8192]);  mul_22 = None
        permute_11: "bf16[8192, 128]" = torch.ops.aten.permute.default(view_23, [1, 0])
        mm_9: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_11, view_14);  permute_11 = None
        mm_10: "bf16[128, 2048]" = torch.ops.aten.mm.default(view_23, permute_13);  view_23 = permute_13 = None
        view_24: "bf16[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_10, [1, 128, 2048]);  mm_10 = None
        convert_element_type_49: "f32[1, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_24, torch.float32);  view_24 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/feed_forward.py:128 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        add_9: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(tangents_1, convert_element_type_49);  tangents_1 = convert_element_type_49 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/feed_forward.py:128 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        sigmoid_1: "bf16[1, 128, 8192]" = torch.ops.aten.sigmoid.default(view_15)
        full_default: "bf16[1, 128, 8192]" = torch.ops.aten.full.default([1, 128, 8192], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        sub: "bf16[1, 128, 8192]" = torch.ops.aten.sub.Tensor(full_default, sigmoid_1);  full_default = None
        mul_24: "bf16[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_15, sub);  view_15 = sub = None
        add_10: "bf16[1, 128, 8192]" = torch.ops.aten.add.Scalar(mul_24, 1);  mul_24 = None
        mul_25: "bf16[1, 128, 8192]" = torch.ops.aten.mul.Tensor(sigmoid_1, add_10);  sigmoid_1 = add_10 = None
        mul_26: "bf16[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_23, mul_25);  mul_23 = mul_25 = None
        view_25: "bf16[128, 8192]" = torch.ops.aten.reshape.default(mul_26, [128, 8192]);  mul_26 = None
        permute_16: "bf16[8192, 128]" = torch.ops.aten.permute.default(view_25, [1, 0])
        mm_11: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_16, view_14);  permute_16 = view_14 = None
        mm_12: "bf16[128, 2048]" = torch.ops.aten.mm.default(view_25, permute_18);  view_25 = permute_18 = None
        view_26: "bf16[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_12, [1, 128, 2048]);  mm_12 = None
        convert_element_type_54: "f32[1, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_26, torch.float32);  view_26 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/feed_forward.py:128 in forward, code: return self.w2(F.silu(self.w1(x)) * self.w3(x))
        add_11: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_9, convert_element_type_54);  add_9 = convert_element_type_54 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:212 in forward, code: x = self.weight.type_as(x) * x
        convert_element_type_22: "f32[2048]" = torch.ops.prims.convert_element_type.default(primals_10, torch.float32);  primals_10 = None
        mul_27: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_11, convert_element_type_22);  convert_element_type_22 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/attention/__init__.py:549 in forward, code: return self.w_out(att)
        view_13: "bf16[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_3, [1, 128, 2048]);  mm_3 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:203 in forward, code: x = x.float()
        convert_element_type_21: "f32[1, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:206 in forward, code: x = x * torch.rsqrt(variance + self.eps)
        mul_8: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_21, rsqrt_2)
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:212 in forward, code: x = self.weight.type_as(x) * x
        mul_28: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(add_11, mul_8);  mul_8 = None
        sum_3: "f32[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_28, [0, 1], True, dtype = torch.float32);  mul_28 = None
        view_27: "f32[2048]" = torch.ops.aten.reshape.default(sum_3, [2048]);  sum_3 = None
        convert_element_type_57: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_27, torch.bfloat16);  view_27 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:206 in forward, code: x = x * torch.rsqrt(variance + self.eps)
        mul_29: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_27, convert_element_type_21)
        mul_30: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_27, rsqrt_2);  mul_27 = None
        sum_4: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_29, [2], True, dtype = torch.float32);  mul_29 = None
        pow_7: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_2, 3);  rsqrt_2 = None
        mul_31: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_4, -0.5);  sum_4 = None
        mul_32: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_31, pow_7);  mul_31 = pow_7 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:205 in forward, code: variance = x.pow(2).mean(-1, keepdim=True)
        expand_1: "f32[1, 128, 2048]" = torch.ops.aten.expand.default(mul_32, [1, 128, 2048]);  mul_32 = None
        div_1: "f32[1, 128, 2048]" = torch.ops.aten.div.Scalar(expand_1, 2048);  expand_1 = None
        pow_8: "f32[1, 128, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_21, 1.0);  convert_element_type_21 = None
        mul_33: "f32[1, 128, 2048]" = torch.ops.aten.mul.Scalar(pow_8, 2.0);  pow_8 = None
        mul_34: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_1, mul_33);  div_1 = mul_33 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:205 in forward, code: variance = x.pow(2).mean(-1, keepdim=True)
        add_12: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_30, mul_34);  mul_30 = mul_34 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:203 in forward, code: x = x.float()
        convert_element_type_58: "bf16[1, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16);  add_12 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/attention/__init__.py:549 in forward, code: return self.w_out(att)
        view_28: "bf16[128, 2048]" = torch.ops.aten.reshape.default(convert_element_type_58, [128, 2048]);  convert_element_type_58 = None
        permute_20: "bf16[2048, 128]" = torch.ops.aten.permute.default(view_28, [1, 0])
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/attention/__init__.py:546 in forward, code: att = att.view(B, T, -1)
        view_11: "bf16[1, 128, 2048]" = torch.ops.aten.reshape.default(getitem_4, [1, 128, -1])
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/attention/__init__.py:549 in forward, code: return self.w_out(att)
        view_12: "bf16[128, 2048]" = torch.ops.aten.reshape.default(view_11, [128, 2048]);  view_11 = None
        mm_13: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_20, view_12);  permute_20 = view_12 = None
        mm_14: "bf16[128, 2048]" = torch.ops.aten.mm.default(view_28, permute_22);  view_28 = permute_22 = None
        view_29: "bf16[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_14, [1, 128, 2048]);  mm_14 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/attention/__init__.py:546 in forward, code: att = att.view(B, T, -1)
        view_30: "bf16[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_29, [1, 128, 16, 128]);  view_29 = None
        
         # File: /ssd/karen/miniconda3/envs/TOLMo/lib/python3.11/site-packages/flash_attn/flash_attn_interface.py:1201 in flash_attn_func, code: return FlashAttnFunc.apply(
        empty_1: "bf16[1, 128, 16, 128]" = torch.ops.aten.empty.memory_format([1, 128, 16, 128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_24: "bf16[1, 128, 16, 128]" = torch.ops.aten.permute.default(empty_1, [0, 1, 2, 3]);  empty_1 = None
        empty_2: "bf16[1, 128, 16, 128]" = torch.ops.aten.empty.memory_format([1, 128, 16, 128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_25: "bf16[1, 128, 16, 128]" = torch.ops.aten.permute.default(empty_2, [0, 1, 2, 3]);  empty_2 = None
        empty_3: "bf16[1, 128, 16, 128]" = torch.ops.aten.empty.memory_format([1, 128, 16, 128], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_26: "bf16[1, 128, 16, 128]" = torch.ops.aten.permute.default(empty_3, [0, 1, 2, 3]);  empty_3 = None
        
        # No stacktrace found for following nodes
        _flash_attn_backward_default: "f32[1, 16, 128]" = torch.ops.flash_attn._flash_attn_backward.default(view_30, convert_element_type_17, convert_element_type_18, view_8, getitem_4, getitem_5, permute_24, permute_25, permute_26, 0.0, 0.08838834764831845, True, 64, 0, 0.0, None, False, getitem_7);  view_30 = convert_element_type_17 = convert_element_type_18 = view_8 = getitem_4 = getitem_5 = getitem_7 = _flash_attn_backward_default = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/rope.py:290 in forward, code: return q_.type_as(q), k_.type_as(k)
        convert_element_type_63: "f32[1, 128, 16, 128]" = torch.ops.prims.convert_element_type.default(permute_25, torch.float32);  permute_25 = None
        convert_element_type_64: "f32[1, 128, 16, 128]" = torch.ops.prims.convert_element_type.default(permute_24, torch.float32);  permute_24 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/rope.py:282 in forward, code: pos_sin[None, k_len - q_len : k_len, None, :],
        unsqueeze: "f32[1, 128, 128]" = torch.ops.aten.unsqueeze.default(primals_7, 0);  primals_7 = None
        unsqueeze_1: "f32[1, 128, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/rope.py:227 in _apply_rotary_pos_emb, code: return ((t * pos_cos) + (self._rotate_half(t) * pos_sin)).to(t.dtype)
        mul_35: "f32[1, 128, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_63, unsqueeze_1)
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/rope.py:222 in _rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_9: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(mul_35, 3, 0, 64)
        slice_10: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(mul_35, 3, 64, 128);  mul_35 = None
        neg_2: "f32[1, 128, 16, 64]" = torch.ops.aten.neg.default(slice_9);  slice_9 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/rope.py:221 in _rotate_half, code: x1, x2 = x.unbind(dim=-2)
        cat_2: "f32[1, 128, 16, 128]" = torch.ops.aten.cat.default([slice_10, neg_2], 3);  slice_10 = neg_2 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/rope.py:283 in forward, code: pos_cos[None, k_len - q_len : k_len, None, :],
        unsqueeze_2: "f32[1, 128, 128]" = torch.ops.aten.unsqueeze.default(primals_8, 0);  primals_8 = None
        unsqueeze_3: "f32[1, 128, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 2);  unsqueeze_2 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/rope.py:227 in _apply_rotary_pos_emb, code: return ((t * pos_cos) + (self._rotate_half(t) * pos_sin)).to(t.dtype)
        mul_36: "f32[1, 128, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_63, unsqueeze_3);  convert_element_type_63 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/rope.py:227 in _apply_rotary_pos_emb, code: return ((t * pos_cos) + (self._rotate_half(t) * pos_sin)).to(t.dtype)
        add_13: "f32[1, 128, 16, 128]" = torch.ops.aten.add.Tensor(cat_2, mul_36);  cat_2 = mul_36 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/rope.py:227 in _apply_rotary_pos_emb, code: return ((t * pos_cos) + (self._rotate_half(t) * pos_sin)).to(t.dtype)
        mul_37: "f32[1, 128, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_64, unsqueeze_1);  unsqueeze_1 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/rope.py:222 in _rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        slice_11: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(mul_37, 3, 0, 64)
        slice_12: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(mul_37, 3, 64, 128);  mul_37 = None
        neg_3: "f32[1, 128, 16, 64]" = torch.ops.aten.neg.default(slice_11);  slice_11 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/rope.py:221 in _rotate_half, code: x1, x2 = x.unbind(dim=-2)
        cat_3: "f32[1, 128, 16, 128]" = torch.ops.aten.cat.default([slice_12, neg_3], 3);  slice_12 = neg_3 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/rope.py:227 in _apply_rotary_pos_emb, code: return ((t * pos_cos) + (self._rotate_half(t) * pos_sin)).to(t.dtype)
        mul_38: "f32[1, 128, 16, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_64, unsqueeze_3);  convert_element_type_64 = unsqueeze_3 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/rope.py:227 in _apply_rotary_pos_emb, code: return ((t * pos_cos) + (self._rotate_half(t) * pos_sin)).to(t.dtype)
        add_14: "f32[1, 128, 16, 128]" = torch.ops.aten.add.Tensor(cat_3, mul_38);  cat_3 = mul_38 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/rope.py:261 in forward, code: q_, k_ = q.float(), k.float()
        convert_element_type_65: "bf16[1, 128, 16, 128]" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None
        convert_element_type_66: "bf16[1, 128, 16, 128]" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/attention/__init__.py:513 in forward, code: k = k.view(B, T, -1, self.head_dim)
        view_36: "bf16[1, 128, 2048]" = torch.ops.aten.reshape.default(convert_element_type_65, [1, 128, 2048]);  convert_element_type_65 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/attention/__init__.py:511 in forward, code: q = q.view(B, T, -1, self.head_dim)
        view_37: "bf16[1, 128, 2048]" = torch.ops.aten.reshape.default(convert_element_type_66, [1, 128, 2048]);  convert_element_type_66 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:214 in forward, code: return x.to(og_dtype)
        convert_element_type_67: "f32[1, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_36, torch.float32);  view_36 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:212 in forward, code: x = self.weight.type_as(x) * x
        convert_element_type_13: "f32[2048]" = torch.ops.prims.convert_element_type.default(primals_6, torch.float32);  primals_6 = None
        mul_39: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_67, convert_element_type_13);  convert_element_type_13 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/attention/__init__.py:496 in forward, code: q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        view_3: "bf16[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_1, [1, 128, 2048]);  mm_1 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:203 in forward, code: x = x.float()
        convert_element_type_12: "f32[1, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:206 in forward, code: x = x * torch.rsqrt(variance + self.eps)
        mul_2: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_12, rsqrt_1)
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:212 in forward, code: x = self.weight.type_as(x) * x
        mul_40: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_67, mul_2);  convert_element_type_67 = mul_2 = None
        sum_5: "f32[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_40, [0, 1], True, dtype = torch.float32);  mul_40 = None
        view_38: "f32[2048]" = torch.ops.aten.reshape.default(sum_5, [2048]);  sum_5 = None
        convert_element_type_68: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_38, torch.bfloat16);  view_38 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:206 in forward, code: x = x * torch.rsqrt(variance + self.eps)
        mul_41: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_39, convert_element_type_12)
        mul_42: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_39, rsqrt_1);  mul_39 = None
        sum_6: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_41, [2], True, dtype = torch.float32);  mul_41 = None
        pow_9: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_1, 3);  rsqrt_1 = None
        mul_43: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_6, -0.5);  sum_6 = None
        mul_44: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_43, pow_9);  mul_43 = pow_9 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:205 in forward, code: variance = x.pow(2).mean(-1, keepdim=True)
        expand_2: "f32[1, 128, 2048]" = torch.ops.aten.expand.default(mul_44, [1, 128, 2048]);  mul_44 = None
        div_2: "f32[1, 128, 2048]" = torch.ops.aten.div.Scalar(expand_2, 2048);  expand_2 = None
        pow_10: "f32[1, 128, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_12, 1.0);  convert_element_type_12 = None
        mul_45: "f32[1, 128, 2048]" = torch.ops.aten.mul.Scalar(pow_10, 2.0);  pow_10 = None
        mul_46: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_2, mul_45);  div_2 = mul_45 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:205 in forward, code: variance = x.pow(2).mean(-1, keepdim=True)
        add_15: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_42, mul_46);  mul_42 = mul_46 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:203 in forward, code: x = x.float()
        convert_element_type_69: "bf16[1, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:214 in forward, code: return x.to(og_dtype)
        convert_element_type_70: "f32[1, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_37, torch.float32);  view_37 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:212 in forward, code: x = self.weight.type_as(x) * x
        convert_element_type_10: "f32[2048]" = torch.ops.prims.convert_element_type.default(primals_5, torch.float32);  primals_5 = None
        mul_47: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_70, convert_element_type_10);  convert_element_type_10 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/attention/__init__.py:496 in forward, code: q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        view_1: "bf16[1, 128, 2048]" = torch.ops.aten.reshape.default(mm, [1, 128, 2048]);  mm = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:203 in forward, code: x = x.float()
        convert_element_type_9: "f32[1, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:206 in forward, code: x = x * torch.rsqrt(variance + self.eps)
        mul: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_9, rsqrt)
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:212 in forward, code: x = self.weight.type_as(x) * x
        mul_48: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_70, mul);  convert_element_type_70 = mul = None
        sum_7: "f32[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_48, [0, 1], True, dtype = torch.float32);  mul_48 = None
        view_39: "f32[2048]" = torch.ops.aten.reshape.default(sum_7, [2048]);  sum_7 = None
        convert_element_type_71: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_39, torch.bfloat16);  view_39 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:206 in forward, code: x = x * torch.rsqrt(variance + self.eps)
        mul_49: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_47, convert_element_type_9)
        mul_50: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_47, rsqrt);  mul_47 = None
        sum_8: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_49, [2], True, dtype = torch.float32);  mul_49 = None
        pow_11: "f32[1, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_51: "f32[1, 128, 1]" = torch.ops.aten.mul.Scalar(sum_8, -0.5);  sum_8 = None
        mul_52: "f32[1, 128, 1]" = torch.ops.aten.mul.Tensor(mul_51, pow_11);  mul_51 = pow_11 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:205 in forward, code: variance = x.pow(2).mean(-1, keepdim=True)
        expand_3: "f32[1, 128, 2048]" = torch.ops.aten.expand.default(mul_52, [1, 128, 2048]);  mul_52 = None
        div_3: "f32[1, 128, 2048]" = torch.ops.aten.div.Scalar(expand_3, 2048);  expand_3 = None
        pow_12: "f32[1, 128, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_9, 1.0);  convert_element_type_9 = None
        mul_53: "f32[1, 128, 2048]" = torch.ops.aten.mul.Scalar(pow_12, 2.0);  pow_12 = None
        mul_54: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(div_3, mul_53);  div_3 = mul_53 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:205 in forward, code: variance = x.pow(2).mean(-1, keepdim=True)
        add_16: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_50, mul_54);  mul_50 = mul_54 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/layer_norm.py:203 in forward, code: x = x.float()
        convert_element_type_72: "bf16[1, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16);  add_16 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/attention/__init__.py:496 in forward, code: q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        view_41: "bf16[1, 128, 2048]" = torch.ops.aten.reshape.default(permute_26, [1, 128, 2048]);  permute_26 = None
        view_42: "bf16[128, 2048]" = torch.ops.aten.reshape.default(view_41, [128, 2048]);  view_41 = None
        permute_28: "bf16[2048, 128]" = torch.ops.aten.permute.default(view_42, [1, 0])
        mm_15: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_28, view);  permute_28 = None
        mm_16: "bf16[128, 2048]" = torch.ops.aten.mm.default(view_42, permute_30);  view_42 = permute_30 = None
        view_45: "bf16[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_16, [1, 128, 2048]);  mm_16 = None
        convert_element_type_77: "f32[1, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/attention/__init__.py:496 in forward, code: q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        add_17: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_11, convert_element_type_77);  add_11 = convert_element_type_77 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/attention/__init__.py:496 in forward, code: q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        view_46: "bf16[128, 2048]" = torch.ops.aten.reshape.default(convert_element_type_69, [128, 2048]);  convert_element_type_69 = None
        permute_32: "bf16[2048, 128]" = torch.ops.aten.permute.default(view_46, [1, 0])
        mm_17: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_32, view);  permute_32 = None
        mm_18: "bf16[128, 2048]" = torch.ops.aten.mm.default(view_46, permute_34);  view_46 = permute_34 = None
        view_47: "bf16[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_18, [1, 128, 2048]);  mm_18 = None
        convert_element_type_82: "f32[1, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_47, torch.float32);  view_47 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/attention/__init__.py:496 in forward, code: q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        add_18: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_17, convert_element_type_82);  add_17 = convert_element_type_82 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/attention/__init__.py:496 in forward, code: q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        view_48: "bf16[128, 2048]" = torch.ops.aten.reshape.default(convert_element_type_72, [128, 2048]);  convert_element_type_72 = None
        permute_36: "bf16[2048, 128]" = torch.ops.aten.permute.default(view_48, [1, 0])
        mm_19: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_36, view);  permute_36 = view = None
        mm_20: "bf16[128, 2048]" = torch.ops.aten.mm.default(view_48, permute_38);  view_48 = permute_38 = None
        view_49: "bf16[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_20, [1, 128, 2048]);  mm_20 = None
        convert_element_type_87: "f32[1, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_49, torch.float32);  view_49 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/attention/__init__.py:496 in forward, code: q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        add_19: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_18, convert_element_type_87);  add_18 = convert_element_type_87 = None
        return (add_19, mm_19, mm_17, mm_15, convert_element_type_71, convert_element_type_68, None, None, mm_13, convert_element_type_57, mm_11, mm_9, mm_7, convert_element_type_39)
        