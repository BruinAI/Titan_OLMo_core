class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2048, 2048]", primals_2: "f32[2048, 2048]", primals_3: "f32[2048, 2048]", primals_4: "f32[1]", primals_5: "f32[2048]", primals_6: "f32[2048]", primals_7: "f32[2048]", primals_8: "f32[2048]", primals_9: "f32[2048]", primals_10: "f32[2048]", primals_11: "f32[2048, 2048]", primals_12: "f32[2048, 2048]", primals_13: "f32[2048, 2048]", primals_14: "f32[2048]", primals_15: "f32[2048]", primals_16: "f32[2048]", primals_17: "f32[2048]", primals_18: "f32[2048]", primals_19: "f32[2048]", primals_20: "f32[2048]", primals_21: "f32[2048]", primals_22: "f32[2048]", primals_23: "f32[1, 36, 2048]"):
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:308 in update_surprise, code: new_surprise = q_T[idx] * old_surprise - grad
        select: "f32[]" = torch.ops.aten.select.int(primals_4, 0, 0);  primals_4 = None
        mul: "f32[2048, 2048]" = torch.ops.aten.mul.Tensor(select, primals_3);  primals_3 = None
        sub: "f32[2048, 2048]" = torch.ops.aten.sub.Tensor(mul, primals_1);  mul = primals_1 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:310 in update_surprise, code: return new_surprise.detach().clamp(-clamp_w, clamp_w)
        clamp_min: "f32[2048, 2048]" = torch.ops.aten.clamp_min.default(sub, -100.0);  sub = None
        clamp_max: "f32[2048, 2048]" = torch.ops.aten.clamp_max.default(clamp_min, 100.0);  clamp_min = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:308 in update_surprise, code: new_surprise = q_T[idx] * old_surprise - grad
        mul_1: "f32[2048]" = torch.ops.aten.mul.Tensor(select, primals_7);  primals_7 = None
        sub_1: "f32[2048]" = torch.ops.aten.sub.Tensor(mul_1, primals_5);  mul_1 = primals_5 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:310 in update_surprise, code: return new_surprise.detach().clamp(-clamp_w, clamp_w)
        clamp_min_1: "f32[2048]" = torch.ops.aten.clamp_min.default(sub_1, -100.0);  sub_1 = None
        clamp_max_1: "f32[2048]" = torch.ops.aten.clamp_max.default(clamp_min_1, 100.0);  clamp_min_1 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:308 in update_surprise, code: new_surprise = q_T[idx] * old_surprise - grad
        mul_2: "f32[2048]" = torch.ops.aten.mul.Tensor(select, primals_10);  primals_10 = None
        sub_2: "f32[2048]" = torch.ops.aten.sub.Tensor(mul_2, primals_8);  mul_2 = primals_8 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:310 in update_surprise, code: return new_surprise.detach().clamp(-clamp_w, clamp_w)
        clamp_min_2: "f32[2048]" = torch.ops.aten.clamp_min.default(sub_2, -100.0);  sub_2 = None
        clamp_max_2: "f32[2048]" = torch.ops.aten.clamp_max.default(clamp_min_2, 100.0);  clamp_min_2 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:308 in update_surprise, code: new_surprise = q_T[idx] * old_surprise - grad
        mul_3: "f32[2048, 2048]" = torch.ops.aten.mul.Tensor(select, primals_13);  primals_13 = None
        sub_3: "f32[2048, 2048]" = torch.ops.aten.sub.Tensor(mul_3, primals_11);  mul_3 = primals_11 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:310 in update_surprise, code: return new_surprise.detach().clamp(-clamp_w, clamp_w)
        clamp_min_3: "f32[2048, 2048]" = torch.ops.aten.clamp_min.default(sub_3, -100.0);  sub_3 = None
        clamp_max_3: "f32[2048, 2048]" = torch.ops.aten.clamp_max.default(clamp_min_3, 100.0);  clamp_min_3 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:308 in update_surprise, code: new_surprise = q_T[idx] * old_surprise - grad
        mul_4: "f32[2048]" = torch.ops.aten.mul.Tensor(select, primals_16);  primals_16 = None
        sub_4: "f32[2048]" = torch.ops.aten.sub.Tensor(mul_4, primals_14);  mul_4 = primals_14 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:310 in update_surprise, code: return new_surprise.detach().clamp(-clamp_w, clamp_w)
        clamp_min_4: "f32[2048]" = torch.ops.aten.clamp_min.default(sub_4, -100.0);  sub_4 = None
        clamp_max_4: "f32[2048]" = torch.ops.aten.clamp_max.default(clamp_min_4, 100.0);  clamp_min_4 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:308 in update_surprise, code: new_surprise = q_T[idx] * old_surprise - grad
        mul_5: "f32[2048]" = torch.ops.aten.mul.Tensor(select, primals_19);  primals_19 = None
        sub_5: "f32[2048]" = torch.ops.aten.sub.Tensor(mul_5, primals_17);  mul_5 = primals_17 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:310 in update_surprise, code: return new_surprise.detach().clamp(-clamp_w, clamp_w)
        clamp_min_5: "f32[2048]" = torch.ops.aten.clamp_min.default(sub_5, -100.0);  sub_5 = None
        clamp_max_5: "f32[2048]" = torch.ops.aten.clamp_max.default(clamp_min_5, 100.0);  clamp_min_5 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:308 in update_surprise, code: new_surprise = q_T[idx] * old_surprise - grad
        mul_6: "f32[2048]" = torch.ops.aten.mul.Tensor(select, primals_22);  select = primals_22 = None
        sub_6: "f32[2048]" = torch.ops.aten.sub.Tensor(mul_6, primals_20);  mul_6 = primals_20 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:310 in update_surprise, code: return new_surprise.detach().clamp(-clamp_w, clamp_w)
        clamp_min_6: "f32[2048]" = torch.ops.aten.clamp_min.default(sub_6, -100.0);  sub_6 = None
        clamp_max_6: "f32[2048]" = torch.ops.aten.clamp_max.default(clamp_min_6, 100.0);  clamp_min_6 = None
        
         # File: /ssd/karen/Titan_OLMo_core/src/olmo_core/nn/titans/neural_memory.py:322 in torch_dynamo_resume_in_update_memory_at_312, code: mse = sqerr.sum(dim=-1).mean()
        sum_1: "f32[1, 36]" = torch.ops.aten.sum.dim_IntList(primals_23, [-1], dtype = torch.float32);  primals_23 = None
        mean: "f32[]" = torch.ops.aten.mean.default(sum_1);  sum_1 = None
        return (clamp_max, clamp_max_1, clamp_max_2, clamp_max_3, clamp_max_4, clamp_max_5, clamp_max_6, mean)
        