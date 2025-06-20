class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2048, 2048]", primals_2: "f32[1, 32, 2048]", primals_3: "f32[2048, 2048]", primals_4: "f32[2048]", primals_5: "f32[2048]", primals_6: "f32[2048, 2048]", primals_7: "f32[2048]", primals_8: "f32[2048]", primals_9: "f32[2048]"):
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:490 in forward, code: queries = self.silu(self.Q(x))
        convert_element_type: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_1: "bf16[1, 32, 2048]" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        permute: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        view: "bf16[32, 2048]" = torch.ops.aten.view.default(convert_element_type_1, [32, 2048]);  convert_element_type_1 = None
        mm: "bf16[32, 2048]" = torch.ops.aten.mm.default(view, permute)
        view_1: "bf16[1, 32, 2048]" = torch.ops.aten.view.default(mm, [1, 32, 2048])
        convert_element_type_4: "f32[1, 32, 2048]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        sigmoid: "f32[1, 32, 2048]" = torch.ops.aten.sigmoid.default(convert_element_type_4)
        mul: "f32[1, 32, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_4, sigmoid);  convert_element_type_4 = sigmoid = None
        convert_element_type_5: "bf16[1, 32, 2048]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:497 in forward, code: queries = F.normalize(queries, eps=1e-8) # Normalize after convolution
        convert_element_type_6: "f32[1, 32, 2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32)
        pow_1: "f32[1, 32, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 2.0);  convert_element_type_6 = None
        sum_1: "f32[1, 1, 2048]" = torch.ops.aten.sum.dim_IntList(pow_1, [1], True);  pow_1 = None
        pow_2: "f32[1, 1, 2048]" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
        clamp_min: "f32[1, 1, 2048]" = torch.ops.aten.clamp_min.default(pow_2, 1e-08)
        expand: "f32[1, 32, 2048]" = torch.ops.aten.expand.default(clamp_min, [1, 32, 2048]);  clamp_min = None
        div: "f32[1, 32, 2048]" = torch.ops.aten.div.Tensor(convert_element_type_5, expand);  convert_element_type_5 = expand = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:87 in <listcomp>, code: outputs = [self.mlps[i](x[i]) for i in range(x.shape[0])]
        select: "f32[32, 2048]" = torch.ops.aten.select.int(div, 0, 0)
        convert_element_type_7: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        convert_element_type_8: "bf16[32, 2048]" = torch.ops.prims.convert_element_type.default(select, torch.bfloat16);  select = None
        permute_1: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        mm_1: "bf16[32, 2048]" = torch.ops.aten.mm.default(convert_element_type_8, permute_1)
        convert_element_type_11: "f32[32, 2048]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_11, [1], correction = 0, keepdim = True)
        getitem: "f32[32, 1]" = var_mean[0]
        getitem_1: "f32[32, 1]" = var_mean[1];  var_mean = None
        add: "f32[32, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[32, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_11, getitem_1);  convert_element_type_11 = None
        mul_1: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_2: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(mul_1, primals_4);  mul_1 = None
        add_1: "f32[32, 2048]" = torch.ops.aten.add.Tensor(mul_2, primals_5);  mul_2 = None
        sigmoid_1: "f32[32, 2048]" = torch.ops.aten.sigmoid.default(add_1)
        mul_3: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(add_1, sigmoid_1);  add_1 = sigmoid_1 = None
        convert_element_type_12: "bf16[2048]" = torch.ops.prims.convert_element_type.default(primals_7, torch.bfloat16);  primals_7 = None
        convert_element_type_13: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        convert_element_type_14: "bf16[32, 2048]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        permute_2: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        addmm: "bf16[32, 2048]" = torch.ops.aten.addmm.default(convert_element_type_12, convert_element_type_14, permute_2);  convert_element_type_12 = None
        convert_element_type_18: "f32[32, 2048]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_18, [1], correction = 0, keepdim = True)
        getitem_2: "f32[32, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 1]" = var_mean_1[1];  var_mean_1 = None
        add_2: "f32[32, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.0001);  getitem_2 = None
        rsqrt_1: "f32[32, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1: "f32[32, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_18, getitem_3);  convert_element_type_18 = None
        mul_4: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_5: "f32[32, 2048]" = torch.ops.aten.mul.Tensor(mul_4, primals_8);  mul_4 = None
        add_3: "f32[32, 2048]" = torch.ops.aten.add.Tensor(mul_5, primals_9);  mul_5 = primals_9 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:88 in forward, code: return torch.stack(outputs, dim=0) + x  # residual connection
        view_2: "f32[1, 32, 2048]" = torch.ops.aten.view.default(add_3, [1, 32, 2048]);  add_3 = None
        add_4: "f32[1, 32, 2048]" = torch.ops.aten.add.Tensor(view_2, div);  view_2 = div = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:87 in <listcomp>, code: outputs = [self.mlps[i](x[i]) for i in range(x.shape[0])]
        permute_3: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_10: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:490 in forward, code: queries = self.silu(self.Q(x))
        permute_15: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (add_4, primals_4, primals_5, primals_8, view, mm, pow_2, convert_element_type_8, mm_1, getitem_1, rsqrt, convert_element_type_14, addmm, getitem_3, rsqrt_1, permute_3, permute_10, permute_15)
        