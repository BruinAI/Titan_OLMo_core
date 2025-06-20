class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "f32[2048]", primals_5: "f32[2048]", primals_8: "f32[2048]", view: "bf16[32, 2048]", mm: "bf16[32, 2048]", pow_2: "f32[1, 1, 2048]", convert_element_type_7: "bf16[32, 2048]", mm_1: "bf16[32, 2048]", getitem_1: "f32[32, 1]", rsqrt: "f32[32, 1]", convert_element_type_13: "bf16[32, 2048]", addmm: "bf16[32, 2048]", getitem_3: "f32[32, 1]", rsqrt_1: "f32[32, 1]", permute_3: "bf16[2048, 2048]", permute_10: "bf16[2048, 2048]", permute_15: "bf16[2048, 2048]", tangents_1: "f32[1, 32, 2048]"):
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:88 in forward, code: return torch.stack(outputs, dim=0) + x  # residual connection
        select_1: "f32[32, 2048]" = torch.ops.aten.select.int(tangents_1, 0, 0)
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:87 in <listcomp>, code: outputs = [self.mlps[i](x[i]) for i in range(x.shape[0])]
        mul_7: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(select_1, primals_8);  primals_8 = None
        mul_8: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(mul_7, 2048)
        sum_2: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(mul_7, [1], True)
        convert_element_type_17: "f32[32, 2048]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        sub_1: "f32[32, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_17, getitem_3);  convert_element_type_17 = getitem_3 = None
        mul_4: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_9: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(mul_7, mul_4);  mul_7 = None
        sum_3: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(mul_9, [1], True);  mul_9 = None
        mul_10: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(mul_4, sum_3);  sum_3 = None
        sub_3: "f32[32, 2048]" = torch.ops.aten.sub.Tensor(mul_8, sum_2);  mul_8 = sum_2 = None
        sub_4: "f32[32, 2048]" = torch.ops.aten.sub.Tensor(sub_3, mul_10);  sub_3 = mul_10 = None
        div_1: "f32[32, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 2048);  rsqrt_1 = None
        mul_11: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(div_1, sub_4);  div_1 = sub_4 = None
        mul_12: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(select_1, mul_4);  mul_4 = None
        sum_4: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_12, [0]);  mul_12 = None
        sum_5: "f32[2048]" = torch.ops.aten.sum.dim_IntList(select_1, [0]);  select_1 = None
        convert_element_type_18: "bf16[32, 2048]" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        mm_2: "bf16[32, 2048]" = torch.ops.aten.mm.default(convert_element_type_18, permute_3);  permute_3 = None
        permute_4: "bf16[2048, 32]" = torch.ops.aten.permute.default(convert_element_type_18, [1, 0])
        mm_3: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_4, convert_element_type_13);  permute_4 = convert_element_type_13 = None
        sum_6: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_18, [0], True, dtype = torch.float32);  convert_element_type_18 = None
        view_3: "f32[2048]" = torch.ops.aten.view.default(sum_6, [2048]);  sum_6 = None
        convert_element_type_24: "f32[32, 2048]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_25: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_default: "f32[2048]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        full_default: "f32[32, 2048]" = torch.ops.aten.full.default([32, 2048], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_10: "f32[32, 2048]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        sub: "f32[32, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_10, getitem_1);  convert_element_type_10 = getitem_1 = None
        mul_1: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_2: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(mul_1, primals_4)
        add_1: "f32[32, 2048]" = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = primals_5 = None
        sigmoid_1: "f32[32, 2048]" = torch.ops.aten.sigmoid.default(add_1)
        sub_5: "f32[32, 2048]" = torch.ops.aten.sub.Tensor(full_default, sigmoid_1);  full_default = None
        mul_13: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(add_1, sub_5);  add_1 = sub_5 = None
        add_5: "f32[32, 2048]" = torch.ops.aten.add.Scalar(mul_13, 1);  mul_13 = None
        mul_14: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(sigmoid_1, add_5);  sigmoid_1 = add_5 = None
        mul_15: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_24, mul_14);  convert_element_type_24 = mul_14 = None
        mul_17: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(mul_15, primals_4);  primals_4 = None
        mul_18: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(mul_17, 2048)
        sum_7: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(mul_17, [1], True)
        mul_19: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(mul_17, mul_1);  mul_17 = None
        sum_8: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(mul_19, [1], True);  mul_19 = None
        mul_20: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(mul_1, sum_8);  sum_8 = None
        sub_7: "f32[32, 2048]" = torch.ops.aten.sub.Tensor(mul_18, sum_7);  mul_18 = sum_7 = None
        sub_8: "f32[32, 2048]" = torch.ops.aten.sub.Tensor(sub_7, mul_20);  sub_7 = mul_20 = None
        div_2: "f32[32, 1]" = torch.ops.aten.div.Tensor(rsqrt, 2048);  rsqrt = None
        mul_21: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(div_2, sub_8);  div_2 = sub_8 = None
        mul_22: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(mul_15, mul_1);  mul_1 = None
        sum_9: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_22, [0]);  mul_22 = None
        sum_10: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_15, [0]);  mul_15 = None
        convert_element_type_27: "bf16[32, 2048]" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        permute_8: "bf16[2048, 32]" = torch.ops.aten.permute.default(convert_element_type_27, [1, 0])
        mm_4: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_8, convert_element_type_7);  permute_8 = convert_element_type_7 = None
        mm_5: "bf16[32, 2048]" = torch.ops.aten.mm.default(convert_element_type_27, permute_10);  convert_element_type_27 = permute_10 = None
        convert_element_type_32: "f32[32, 2048]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_33: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        full_default_1: "f32[1, 32, 2048]" = torch.ops.aten.full.default([1, 32, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "f32[1, 32, 2048]" = torch.ops.aten.select_scatter.default(full_default_1, convert_element_type_32, 0, 0);  full_default_1 = convert_element_type_32 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:87 in <listcomp>, code: outputs = [self.mlps[i](x[i]) for i in range(x.shape[0])]
        add_6: "f32[1, 32, 2048]" = torch.ops.aten.add.Tensor(tangents_1, select_scatter);  tangents_1 = select_scatter = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:490 in forward, code: queries = self.silu(self.Q(x))
        view_1: "bf16[1, 32, 2048]" = torch.ops.aten.view.default(mm, [1, 32, 2048]);  mm = None
        convert_element_type_3: "f32[1, 32, 2048]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32)
        sigmoid: "f32[1, 32, 2048]" = torch.ops.aten.sigmoid.default(convert_element_type_3)
        mul: "f32[1, 32, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_3, sigmoid);  convert_element_type_3 = sigmoid = None
        convert_element_type_4: "bf16[1, 32, 2048]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:497 in forward, code: queries = F.normalize(queries, eps=1e-8) # Normalize after convolution
        clamp_min: "f32[1, 1, 2048]" = torch.ops.aten.clamp_min.default(pow_2, 1e-08)
        expand: "f32[1, 32, 2048]" = torch.ops.aten.expand.default(clamp_min, [1, 32, 2048]);  clamp_min = None
        div: "f32[1, 32, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_4, expand)
        div_4: "f32[1, 32, 2048]" = torch.ops.aten.div.Tensor(div, expand);  div = None
        neg: "f32[1, 32, 2048]" = torch.ops.aten.neg.default(add_6)
        mul_23: "f32[1, 32, 2048]" = torch.ops.aten.mul.Tensor(neg, div_4);  neg = div_4 = None
        div_5: "f32[1, 32, 2048]" = torch.ops.aten.div.Tensor(add_6, expand);  add_6 = expand = None
        convert_element_type_34: "bf16[1, 32, 2048]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        sum_11: "f32[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(mul_23, [1], True, dtype = torch.float32);  mul_23 = None
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ge: "b8[1, 1, 2048]" = torch.ops.aten.ge.Scalar(pow_2, 1e-08)
        where: "f32[1, 1, 2048]" = torch.ops.aten.where.self(ge, sum_11, full_default_2);  ge = sum_11 = None
        div_6: "f32[1, 32, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_4, pow_2);  convert_element_type_4 = None
        eq: "b8[1, 1, 2048]" = torch.ops.aten.eq.Scalar(pow_2, 0);  pow_2 = None
        where_1: "f32[1, 32, 2048]" = torch.ops.aten.where.self(eq, full_default_2, div_6);  eq = full_default_2 = div_6 = None
        mul_24: "f32[1, 32, 2048]" = torch.ops.aten.mul.Tensor(where, where_1);  where = where_1 = None
        convert_element_type_35: "bf16[1, 32, 2048]" = torch.ops.prims.convert_element_type.default(mul_24, torch.bfloat16);  mul_24 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:497 in forward, code: queries = F.normalize(queries, eps=1e-8) # Normalize after convolution
        add_7: "bf16[1, 32, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_34, convert_element_type_35);  convert_element_type_34 = convert_element_type_35 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:490 in forward, code: queries = self.silu(self.Q(x))
        sigmoid_3: "bf16[1, 32, 2048]" = torch.ops.aten.sigmoid.default(view_1)
        full_default_4: "bf16[1, 32, 2048]" = torch.ops.aten.full.default([1, 32, 2048], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        sub_9: "bf16[1, 32, 2048]" = torch.ops.aten.sub.Tensor(full_default_4, sigmoid_3);  full_default_4 = None
        mul_25: "bf16[1, 32, 2048]" = torch.ops.aten.mul.Tensor(view_1, sub_9);  view_1 = sub_9 = None
        add_8: "bf16[1, 32, 2048]" = torch.ops.aten.add.Scalar(mul_25, 1);  mul_25 = None
        mul_26: "bf16[1, 32, 2048]" = torch.ops.aten.mul.Tensor(sigmoid_3, add_8);  sigmoid_3 = add_8 = None
        mul_27: "bf16[1, 32, 2048]" = torch.ops.aten.mul.Tensor(add_7, mul_26);  add_7 = mul_26 = None
        view_4: "bf16[32, 2048]" = torch.ops.aten.view.default(mul_27, [32, 2048]);  mul_27 = None
        permute_13: "bf16[2048, 32]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm_6: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_13, view);  permute_13 = view = None
        mm_7: "bf16[32, 2048]" = torch.ops.aten.mm.default(view_4, permute_15);  view_4 = permute_15 = None
        view_5: "bf16[1, 32, 2048]" = torch.ops.aten.view.default(mm_7, [1, 32, 2048]);  mm_7 = None
        convert_element_type_40: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        return (convert_element_type_40, view_5, convert_element_type_33, sum_9, sum_10, convert_element_type_25, convert_element_type_default, sum_4, sum_5)
        