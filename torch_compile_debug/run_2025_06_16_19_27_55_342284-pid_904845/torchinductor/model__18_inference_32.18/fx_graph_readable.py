class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2048, 2048]", arg1_1: "f32[2048, 2048]", arg2_1: "f32[2048, 2048]", arg3_1: "f32[1]", arg4_1: "f32[1]"):
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:293 in update_param, code: orig_weight_coeff = p_T[idx] * self.l2_factor
        select: "f32[]" = torch.ops.aten.select.int(arg3_1, 0, 0);  arg3_1 = None
        mul: "f32[]" = torch.ops.aten.mul.Tensor(select, 1.0);  select = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:294 in update_param, code: new_param = orig_weight_coeff * orig_param + A_T[idx] * surprise - grad
        mul_1: "f32[2048, 2048]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        select_1: "f32[]" = torch.ops.aten.select.int(arg4_1, 0, 0);  arg4_1 = None
        mul_2: "f32[2048, 2048]" = torch.ops.aten.mul.Tensor(select_1, arg2_1);  select_1 = arg2_1 = None
        add: "f32[2048, 2048]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        sub: "f32[2048, 2048]" = torch.ops.aten.sub.Tensor(add, arg0_1);  add = arg0_1 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:299 in update_param, code: return new_param.detach().clamp(-clamp_w, clamp_w).requires_grad_(True)
        clamp_min: "f32[2048, 2048]" = torch.ops.aten.clamp_min.default(sub, -100.0);  sub = None
        clamp_max: "f32[2048, 2048]" = torch.ops.aten.clamp_max.default(clamp_min, 100.0);  clamp_min = None
        return (clamp_max,)
        