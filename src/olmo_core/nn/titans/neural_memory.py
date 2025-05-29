import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call
import torch.utils.checkpoint as cp
from typing import List
import regex as re


class ParallelMLPs(nn.Module):
    _mlp_index_regex = re.compile(r"\.(\d+)\.")

    """
    ParallelMLPs is a wrapper for multiple MLPs that allows for parallel processing of inputs with torch.compile.
    """
    def __init__(self, mlp_list: List[nn.Module], template_weights: nn.ParameterDict, memory_l2_weight: float):
        super().__init__()
        self.mlps = nn.ModuleList(mlp_list)
        self.reset_weights_from_template(template_weights)
        self.name_to_batch_idx = {  # the index of the MLP the param is from which = its batch index
            name: self.get_name_idx(name) for name, _ in self.named_parameters()
        }
        self.memory_l2_weight = memory_l2_weight

    @staticmethod
    def get_name_idx(name: str) -> int:
        if idx := ParallelMLPs._mlp_index_regex.search(name):
            return int(idx.group(1))
        raise ValueError(f"Parameter name {name} does not match expected format.")
    
    @staticmethod
    @torch.compile()
    def soft_sqrt_clip(g: torch.Tensor, eps=1e-8) -> torch.Tensor:
        if g is None:
            return g
        norm = g.norm()
        if norm <= 1.0:
            return g
        return g / torch.sqrt(norm + eps)

    @staticmethod
    @torch.compile()
    def scaled_root_excess_norm(g: torch.Tensor, alpha=5, eps=1e-8) -> torch.Tensor:
        """
        Soft clipping function such that g is rescale to have a norm of:
            2 * alpha * (sqrt(1 + ||g|| / (alpha + eps)) - 1)
        """
        if g is None:
            return g
        norm = g.norm()
        final_norm = 2 * alpha * (torch.sqrt(1 + norm / (alpha + eps)) - 1)
        return g * (final_norm / (norm + eps)) if norm > 0 else g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != len(self.mlps):
            raise ValueError(
                f"Input batch size {x.shape[0]} must match the number of MLPs {len(self.mlps)}"
            )
        outputs = [self.mlps[i](x[i]) for i in range(x.shape[0])]
        return torch.stack(outputs, dim=0) + x  # residual connection
    
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     if x.shape[0] != len(self.mlps):
    #         raise ValueError(f"Input batch size {x.shape[0]} must match the number of MLPs {len(self.mlps)}")
        
    #     # Debug: Check input for extreme values
    #     if torch.isnan(x).any():
    #         raise ValueError("Input x contains NaN")
    #     if torch.isinf(x).any():
    #         raise ValueError("Input x contains Inf")
        
    #     # Check for extreme values that might cause numerical issues
    #     x_min, x_max = x.min(), x.max()
    #     if x_max > 1e10 or x_min < -1e10:
    #         print(f"WARNING: Input x has extreme values: min={x_min}, max={x_max}")
        
    #     outputs = []
    #     for i in range(x.shape[0]):
    #         single_input = x[i]
            
    #         # Debug single input
    #         if torch.isnan(single_input).any():
    #             raise ValueError(f"Input x[{i}] contains NaN")
    #         if torch.isinf(single_input).any():
    #             raise ValueError(f"Input x[{i}] contains Inf")
                
    #         try:
    #             single_output = self.mlps[i](single_input)
                
    #             # Debug single output
    #             if torch.isnan(single_output).any():
    #                 print(f"ERROR: MLP {i} produced NaN output")
    #                 print(f"  Input stats: min={single_input.min()}, max={single_input.max()}, mean={single_input.mean()}, std={single_input.std()}")
                    
    #                 # Debug each layer in the MLP
    #                 temp_x = single_input
    #                 for j, layer in enumerate(self.mlps[i]):
    #                     temp_x = layer(temp_x)
    #                     if torch.isnan(temp_x).any():
    #                         print(f"  Layer {j} ({type(layer).__name__}) produced NaN")
    #                         print(f"    Input to this layer: min={temp_x.min()}, max={temp_x.max()}")
    #                         break
    #                     else:
    #                         print(f"  Layer {j} ({type(layer).__name__}) OK: min={temp_x.min()}, max={temp_x.max()}")
                    
    #                 raise ValueError(f"MLP {i} produced NaN output")
                
    #             if torch.isinf(single_output).any():
    #                 raise ValueError(f"MLP {i} produced Inf output")
                    
    #             outputs.append(single_output)
                
    #         except Exception as e:
    #             print(f"Error in MLP {i}: {e}")
    #             print(f"  Input shape: {single_input.shape}")
    #             print(f"  Input stats: min={single_input.min()}, max={single_input.max()}")
    #             raise
        
    #     stacked_outputs = torch.stack(outputs, dim=0)
        
    #     # Debug the residual connection
    #     if torch.isnan(stacked_outputs).any():
    #         raise ValueError("Stacked outputs contain NaN before residual")
        
    #     final_output = stacked_outputs + x
        
    #     if torch.isnan(final_output).any():
    #         print("ERROR: Final output contains NaN after residual connection")
    #         print(f"  stacked_outputs stats: min={stacked_outputs.min()}, max={stacked_outputs.max()}")
    #         print(f"  x stats: min={x.min()}, max={x.max()}")
    #         raise ValueError("Final output contains NaN after residual connection")
        
    #     return final_output
    
    @torch.compile()
    def calculate_coeffs(self, beta_vecs, eta_vecs, theta_vecs):
        # ---------------------------------------------------------
        eps = 1e-7
        beta_vecs = beta_vecs.clamp(min=eps, max=1.0 - eps)
        eta_vecs  = eta_vecs.clamp(min=eps, max=1.0 - eps)
        
        # ---------- prefix / suffix cumulative products ----------
        p_prefix = beta_vecs.cumprod(1)                                 # p_t  = β_0⋯β_t: B, T
        p_suffix = beta_vecs.flip(1).cumprod(1).flip(1)                 # β_t⋯β_{T-1}, B: T

        q_prefix = eta_vecs.cumprod(1)                                  # q_t  = η_0⋯η_t: B, T
        q_suffix = eta_vecs.flip(1).cumprod(1).flip(1)                  # η_t⋯η_{T-1}: B, T

        p_T = p_prefix[:, -1]                                           # final p_T:  B
        q_T = q_prefix[:, -1]                                           # final q_T:  B

        # ---------- w_k  (shape: T) ----------
        w = (p_suffix / beta_vecs + eps) * q_prefix                           # β^{T-1-k} · η^{k+1}: B, T

        # ---------- A_T (scalar) ----------
        A_T = w.sum(dim=1)                                              # A_T:  B

        # ---------- B_{T,j}  (shape: T) ----------
        partial_sum = w.flip(1).cumsum(dim=1).flip(1)                   # Σ_{k=j}^{T-1} w_k: B, T
        B_coeffs = theta_vecs * partial_sum / (q_prefix + eps)         # −θ · B_{T,j}: B, T

        # ---------- D_{T,j}  (shape: T) ----------
        D_coeffs = theta_vecs * q_suffix                                # −θ · D_{T,j}: B, T

        return p_T, q_T, A_T, B_coeffs.unsqueeze(-1), D_coeffs.unsqueeze(-1)
    
    # @torch.compile()
    def update_memory(self, current_params, surprises, keys, values, beta_vecs, eta_vecs, theta_vecs, ckpt_memory=True, audit_grad=True):
        with torch.enable_grad():  # Enable gradients for this specific block
            new_params = {}
            
            # ================================================================
            # Recurrence definitions (for context)
            # --------------------------------------------------------
            #   S_t = η_t · S_{t-1} − θ_t · u_t
            #   M_t = (1 − α_t) · M_{t-1} + S_t
            # --------------------------------------------------------
            # Closed-form coefficients
            # --------------------------------------------------------
            #   p_t              = ∏_{i=0}^{t} β_i
            #   q_t              = ∏_{i=0}^{t} η_i
            #
            #   w_k              =  (∏_{i=k+1}^{T-1} β_i) · (∏_{i=0}^{k} η_i)
            #                    =  β_{k+1} β_{k+2} … β_{T-1} · η_0 η_1 … η_k
            #
            #   A_T              =  Σ_{k=0}^{T-1} w_k
            #
            #   B_{T,j}          =  (Σ_{k=j}^{T-1} w_k) · q_j⁻¹
            #   D_{T,j}          =  q_T · q_j⁻¹
            #
            #   M_T (param)      =  p_T·M_0  +  A_T·S_0  −  Σ_j θ·B_{T,j}·u_j
            #   S_T (surprise)   =  q_T·S_0  −  Σ_j θ·D_{T,j}·u_j
            # ================================================================
            p_T, q_T, A_T, B_coeffs, D_coeffs = self.calculate_coeffs(beta_vecs, eta_vecs, theta_vecs)

            if ckpt_memory:
                # Using checkpointing to save memory (recomputes forward pass during back-prop instead of storing all activations)
                def _fwd(keys):
                    return functional_call(self, current_params, keys)
                outputs = cp.checkpoint(_fwd, keys.detach().requires_grad_(), use_reentrant=False)
            else:
                outputs = functional_call(self, current_params, keys)
            
            # assert not torch.isnan(outputs).any(), "Outputs contain NaN values, which is unexpected."
            # assert not torch.isinf(outputs).any(), "Outputs contain NaN or Inf values, which is unexpected."
            sqerr = (outputs - values).pow(2)  # squared error
            #if not self.training:
            #    print(outputs, values)
            
            input_params = tuple(current_params.values())
            mem_grads = torch.autograd.grad(
                outputs=sqerr, inputs=input_params, grad_outputs=B_coeffs.expand_as(sqerr), 
                allow_unused=True, retain_graph=True
            )
            
            
            surp_grads = torch.autograd.grad(
                outputs=sqerr, inputs=input_params, grad_outputs=D_coeffs.expand_as(sqerr), 
                allow_unused=True
            )
            
            if audit_grad:
                # Log L2 norm of mem_grads
                mem_grad_norms = []
                for i, grad in enumerate(mem_grads):
                    if grad is not None:
                        norm = grad.norm().item()
                        mem_grad_norms.append(norm)
                        if torch.isnan(grad).any():
                            print(f"NaN detected in mem_grad[{i}] with norm {norm}")
                        elif norm > 1000:
                            print(f"Large gradient detected in mem_grad[{i}] with norm {norm}")
                
                print(f"Memory gradient norms: {mem_grad_norms}")
                
                # Log L2 norm of surp_grads
                surp_grad_norms = []
                for i, grad in enumerate(surp_grads):
                    if grad is not None:
                        norm = grad.norm().item()
                        surp_grad_norms.append(norm)
                        if torch.isnan(grad).any():
                            print(f"NaN detected in surp_grad[{i}] with norm {norm}")
                        elif norm > 1000:
                            print(f"Large gradient detected in surp_grad[{i}] with norm {norm}")
                
                print(f"Surprise gradient norms: {surp_grad_norms}")
                
                # Log B_coeffs and D_coeffs stats
                print(f"B_coeffs stats - Min: {B_coeffs.min().item():.6f}, Max: {B_coeffs.max().item():.6f}, Mean: {B_coeffs.mean().item():.6f}")
                print(f"D_coeffs stats - Min: {D_coeffs.min().item():.6f}, Max: {D_coeffs.max().item():.6f}, Mean: {D_coeffs.mean().item():.6f}")

            # clip norms
            clip_mem_g = 5
            clip_sur_g = 10
            clip_w = 20
            
            # mem_grads = [self.soft_sqrt_clip(g) if g is not None else g for g in mem_grads]
            # mem_grads = [nn.utils.clip_grad_norm_(g, clip_g) if g is not None else g for g in mem_grads]
            mem_grads = [self.scaled_root_excess_norm(g, clip_mem_g) if g is not None else g for g in mem_grads]
            surp_grads = [self.scaled_root_excess_norm(g, clip_sur_g) if g is not None else g for g in surp_grads]

            # p_T * M_0 + A_T[idx] * S_0 + gradient_term
            def update_param(name, grad):
                if grad is None:
                    return current_params[name].detach().clone().requires_grad_(True)
                idx = self.name_to_batch_idx[name]
                orig_param = current_params[name]  # (*param_shape)
                surprise = surprises.get(name, torch.zeros_like(orig_param))
                # M_T (param) = p_T·M_0 + A_T·S_0 − Σ_j θ·B_{T,j}·u_j
                orig_weight_coeff = p_T[idx] * (1 - 2 * self.memory_l2_weight)
                new_param = orig_weight_coeff * orig_param + A_T[idx] * surprise - grad
                return new_param.detach().clamp(-clip_w, clip_w).requires_grad_(True)

            # q_T * S_0 - Σ_j θ·D_{T,j}·u_j
            def update_surprise(name, grad):
                if grad is None:
                    return torch.zeros_like(current_params[name]).detach()
                idx = self.name_to_batch_idx[name]
                old_surprise = surprises.get(name, torch.zeros_like(current_params[name]))
                # S_T (surprise) = q_T·S_0 − Σ_j θ·D_{T,j}·u_j
                new_surprise = q_T[idx] * old_surprise - grad
                return new_surprise.detach().clamp(-clip_w, clip_w)

            new_params = {
                name: update_param(name, grad)
                for name, grad in zip(current_params.keys(), mem_grads)
            }

            surprises = {
                name: update_surprise(name, grad)
                for name, grad in zip(current_params.keys(), surp_grads)
            }

            mse = sqerr.sum(dim=-1).mean()
            return surprises, mse, new_params
        
    def reset_weights_from_template(self, template_weights: nn.ParameterDict):
        """
        Copies the learned template weights to all MLP instances.
        This is used for resetting the MLPs to their learned base state.
        Hooks are NOT re-registered here.
        """
        with torch.no_grad(): # Ensure this operation is not tracked by autograd
            for mlp_instance in self.mlps:
                for name, param_instance in mlp_instance.named_parameters():
                    clean_name = name.replace('.', '_')
                    if clean_name in template_weights:
                        param_instance.data.copy_(template_weights[clean_name].data)
                    else:
                        # This case implies a mismatch if it occurs after successful initialization.
                        raise KeyError(
                            f"Instance parameter {name} (cleaned: {clean_name}) not found in template weights "
                            f"during reset. Ensure MLP structures remain consistent."
                        )


class NeuralMemory(nn.Module):
    """
    Neural Memory Module Requirements:
    1. can have any amount of mlp layers with same hyperparameters
    2. but has shared, learned K and V matrices ad 
    """

    def __init__(self, emb_dim = 16, n_layers = 2, 
                 hidden_dim = 32, nu = 0.001, l2_memory_weight = 0.,
                 use_global_sw = False, num_global_tokens = 0,
                 use_conv=False, retrieve_layer=True,
                 audit_grad=False):
        super().__init__()

        # Define the layers of the network
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.mlps_processor: nn.Module | None = None  # placeholder until MLPs init
        self.mlp_states = []
        self.surprise = {}
        self.mlp_template_weights = self.init_mlp_template_weights()
        self.mlp_reset = True
        self.nu = nu
        self.l2_memory_weight = l2_memory_weight
        self.use_conv = use_conv
        self.retrieve_layer = retrieve_layer
        self.audit_grad = audit_grad

        self.K = nn.Linear(emb_dim, emb_dim, bias = False)  # Mapping to keys
        self.Q = nn.Linear(emb_dim, emb_dim, bias = False)  # Mapping to queries
        self.V = nn.Linear(emb_dim, emb_dim, bias = False)  # Mapping to values
        if self.retrieve_layer:
            self.retrieve_to_gate = nn.Linear(emb_dim, emb_dim)  # Mapping to retrieve gate

        torch.nn.init.normal_(self.K.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.V.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.Q.weight, mean=0.0, std=0.02)
        if self.retrieve_layer:
            torch.nn.init.normal_(self.retrieve_to_gate.weight, mean=0.0, std=0.02)

        self.alpha = nn.Linear(emb_dim, 1, bias=True)
        self.eta = nn.Linear(emb_dim, 1, bias=True)
        self.theta = nn.Linear(emb_dim, 1, bias=True)

        torch.nn.init.normal_(self.alpha.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.eta.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.theta.weight, mean=0.0, std=0.02)

        # Initialize bias terms to -4.5 (sigmoid(-3.5)=0.01)
        with torch.no_grad():
            self.alpha.bias.fill_(-3) # alpha needs to be close to 0: beta = 1 - alpha close to 1
            self.eta.bias.fill_(2) # eta needs to be close to 1 for proper momentum
            self.theta.bias.fill_(-5) # theta needs to be close to 0 for low learning rate

        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

        if self.use_conv:
            # Depthwise-separable convolutions
            self.conv_q = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding='same', groups=emb_dim, bias=False)
            self.conv_k = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding='same', groups=emb_dim, bias=False)
            self.conv_v = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding='same', groups=emb_dim, bias=False)

            torch.nn.init.normal_(self.conv_q.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.conv_k.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.conv_v.weight, mean=0.0, std=0.02)

        self.use_global_sw = use_global_sw
        self.num_global_tokens = num_global_tokens
        
        if self.use_global_sw and self.num_global_tokens > 0:
            self.persistent_tokens = nn.Parameter(
            torch.empty(self.num_global_tokens, self.emb_dim)
            )
            torch.nn.init.normal_(self.persistent_tokens, mean=0.0, std=0.1)

        self.surprise = {}

    def build_mlp(self):
        """
        Build the MLP layers based on the specified architecture.
        This function is called during initialization to set up the MLP layers.
        """
        # Define the layers of the network
        if self.n_layers == 1:
            layers: List[nn.Module] = [nn.Linear(self.emb_dim, self.emb_dim)]
        else:
            layers: List[nn.Module] = [
                nn.Linear(self.emb_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.SiLU(),
            ]
            for k in range(self.n_layers - 2):
                layers += [
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.SiLU(),
                ]
            layers.append(nn.Linear(self.hidden_dim, self.emb_dim))
        layers.append(nn.LayerNorm(self.emb_dim, eps=1e-4))  # Layer normalization
        return nn.Sequential(*layers)

    def get_mlp_params(self):
        return {
            name.replace('_orig_mod.', ''): param.clone().detach().requires_grad_(True)
            for name, param in self.mlps_processor.named_parameters()  # type: ignore
        }

    # Ideally only called once, compiling slows it down, pad inputs instead
    def init_mlp(self, batch_size):
        del self.mlps_processor
        device = next(self.parameters()).device
        mlps = []
        for i in range(batch_size):
            mlp = self.build_mlp().to(device)  # build the mlp
            mlps.append(mlp)  # adding the mlp to the list
        parallel_mlps = ParallelMLPs(mlps, self.mlp_template_weights, memory_l2_weight=self.l2_memory_weight).to(device)  # type: ignore
        self.mlps_processor = torch.compile(parallel_mlps, mode="reduce-overhead")  # type: ignore
        # self.mlps_processor = parallel_mlps

        self.mlp_states.append(self.get_mlp_params())

    def reset_mlps(self):
        self.mlp_reset = True
        if self.mlps_processor is not None:
            self.mlps_processor.reset_weights_from_template(self.mlp_template_weights)  # type: ignore
            del self.mlp_states[:]
            self.mlp_states = [self.get_mlp_params()]
            self.surprise.clear()

    def retrieve(self, x):
        return self.forward(x)

    def forward(self, x):
        if self.mlps_processor is None or self.mlp_states[-1] is None:
            raise RuntimeError("MLPs not initialized. Call init_mlp(batch_size) first.")

        queries = self.silu(self.Q(x))
        if self.use_conv:
            # Apply 1D depthwise-separable convolution
            # Input to Conv1d: (B, N, L), current shape: (B, L, N)
            queries = queries.transpose(1, 2)  # B, N, L -> B, L, N
            queries = self.conv_q(queries)
            queries = queries.transpose(1, 2)  # B, L, N -> B, N, L
        queries = F.normalize(queries, eps=1e-8) # Normalize after convolution

        outputs = functional_call(self.mlps_processor, self.mlp_states[-1], queries)
        if not self.retrieve_layer:
            return outputs
        # transform retrieved memory to sigmoid gate inputs (not needed, but more intuitive imo)
        outputs = self.retrieve_to_gate(outputs)
        return outputs

    # @torch.compile()
    def update(self, x):
        if self.mlps_processor is None or self.mlp_states[-1] is None:
            raise RuntimeError("MLPs not initialized. Call init_mlp(batch_size) first.")
        
        self.mlp_reset = False
        z = x.detach()
           
        # NOT SURE IF THIS SHOULD GO BEFORE OR AFTER THE DETATCH
        
        if self.use_global_sw and self.num_global_tokens > 0:
            # Add batch dimension [num_global_tokens, emb_dim] -> [1, num_global_tokens, emb_dim]
            repeated_persistent_tokens = self.persistent_tokens.unsqueeze(0)
            
            # Expand to match batch size [1, num_global_tokens, emb_dim] -> [batch_size, num_global_tokens, emb_dim]
            repeated_persistent_tokens = repeated_persistent_tokens.expand(z.shape[0], -1, -1)
            
            # Concatenate with input along sequence dimension
            z = torch.cat([repeated_persistent_tokens, z], dim=1)

        # Evaluate the corresponding keys and values
        keys = self.silu(self.K(z))
        values = self.silu(self.V(z))

        if self.use_conv:
            # Apply 1D depthwise-separable convolution to keys
            # B, N, L -> B, L, N
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)

            keys = self.conv_k(keys)
            values = self.conv_v(values)

            # B, L, N -> B, N, L
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
        
        keys = F.normalize(keys, eps=1e-8)
        values = F.normalize(values, eps=1e-8)

        # Computing β, η, & θ vectors which are gated between 0 and 1: B, N, D -> B, N
        beta_vec = 1 - self.sigmoid(self.alpha(keys)).squeeze(-1)  # (B, N)
        eta_vec = self.sigmoid(self.eta(keys)).squeeze(-1)  # (B, N)
        theta_vec = self.sigmoid(self.theta(keys)).squeeze(-1)  # (B, N)

        self.surprise, losses, next_mlp_params = self.mlps_processor.update_memory(  # type: ignore
            self.mlp_states[-1], self.surprise, keys, values, beta_vec, eta_vec, theta_vec, audit_grad = self.audit_grad
        )
        if self.training:
            self.mlp_states.append(next_mlp_params)
        return losses

    def init_mlp_template_weights(self, seed=42):
        reference_mlp = self.build_mlp()
        template_weights = nn.ParameterDict()
        with torch.no_grad():
            torch.manual_seed(seed)
            for name, param_to_copy in reference_mlp.named_parameters():
                # Sanitize name for ParameterDict key, e.g., 'layers.0.weight' -> 'layers_0_weight'
                clean_name = name.replace('.', '_')
                # Create new parameters for the template
                new_param = nn.Parameter(torch.empty_like(param_to_copy.data))
                
                # Initialize the template parameters (e.g., normal distribution)
                # This initialization is for the *learnable template*.
                # Adjust mean and std as needed, or use other init functions.
                # if "weight" in name:
                #     torch.nn.init.normal_(new_param.data, mean=0, std=0.02)
                # elif "bias" in name: # Bias exists but is None (e.g. bias=False in Linear)
                #     torch.nn.init.zeros_(new_param.data) # Or some other default
                # else: # Fallback for other params
                #     raise ValueError(f"Unexpected parameter name: {name}")
                
                module_idx_str = name.split('.')[0]
                parent_module = None
                if module_idx_str.isdigit() and int(module_idx_str) < len(reference_mlp):
                    parent_module = reference_mlp[int(module_idx_str)]
                
                param_type = name.split('.')[-1] # "weight" or "bias"
                if param_type == "weight":
                    if isinstance(parent_module, nn.LayerNorm):
                        torch.nn.init.ones_(new_param.data)  # LayerNorm weight (gamma)
                    elif isinstance(parent_module, nn.Linear):
                        # Kaiming He initialization for Linear layers
                        torch.nn.init.kaiming_normal_(new_param.data, mode='fan_in', nonlinearity='relu')
                        # Alternative: torch.nn.init.normal_(new_param.data, mean=0, std=0.02)
                    else:
                        # Fallback for other types of weights if MLP structure changes
                        torch.nn.init.normal_(new_param.data, mean=0, std=0.02)
                elif param_type == "bias":
                    # For both LayerNorm and Linear biases
                    torch.nn.init.zeros_(new_param.data)
                else:
                    # This case should ideally not be hit if all params are standard 'weight' or 'bias'
                    raise ValueError(f"Unexpected parameter name structure: {name}")
                
                
                # new param with no gradients
                template_weights[clean_name] = new_param.detach()
        del reference_mlp
        return template_weights

    @torch.compile()
    def train_initial_mlp(self):
        """
        Aggregate parameters from every stored MLP state and update the
        shared template weights via an exponential moving-average.
        """
        num_states = len(self.mlp_states)
        if num_states == 0:
            return

        mlp_sums: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for state in self.mlp_states:
                for name, param in state.items():
                    match = ParallelMLPs._mlp_index_regex.search(name)
                    if match is None:
                        raise ValueError(f"Parameter name {name} does not match expected format.")
                    clean_name = name[match.end():].replace('.', '_')

                    if not torch.isfinite(param).all():
                        raise ValueError(f"Parameter {name} contains NaN or Inf values.")

                    if clean_name not in mlp_sums:
                        mlp_sums[clean_name] = param.detach().clone()  # initializing with clone for running sum
                    else:
                        mlp_sums[clean_name].add_(param.detach())  # accumulate in‑place

            for clean_name, summed_param in mlp_sums.items():
                if clean_name not in self.mlp_template_weights:
                    raise AssertionError(f"Parameter {clean_name} not found in template weights.")

                avg_param = summed_param / num_states
                template_weight = self.mlp_template_weights[clean_name]

                new_weight = (1.0 - self.nu) * template_weight + self.nu * avg_param
                if not torch.isfinite(new_weight).all():
                    raise ValueError(f"Updated weight for {clean_name} contains NaN or Inf values.")

                template_weight.data.copy_(new_weight)  # copy into Parameter to preserve state‑dict link
