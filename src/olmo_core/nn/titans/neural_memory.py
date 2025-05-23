import torch
from torch import nn
from torch.nn.functional import normalize
from torch.func import functional_call
from typing import List
import regex as re


# def loss_one(model, param_dict, x, target):
#     """
#     Loss for each index in seq
#     Args:
#         model: The model to be used.
#         param_dict: The parameters of the model.
#         x: The input tensor.  (B x seq_len x emb_dim)
#         target: The target tensor.  (B x seq_len x emb_dim)
#     """
#     logits = functional_call(model, param_dict, x)
#     return nn.functional.mse_loss(logits, target)

# grad_and_loss_one = grad_and_value(loss_one, argnums=1)  # takes the gradient and value of the loss w.r.t. the model parameters
# per_sample_grads_and_losses = vmap(grad_and_loss_one, in_dims=(None, None, 1, 1))  # creates a function to get per sample gradients and value

class ParallelMLPs(nn.Module):
    """
    ParallelMLPs is a wrapper for multiple MLPs that allows for parallel processing of inputs with torch.compile.
    """
    def __init__(self, mlp_list: List[nn.Module], template_weights: nn.ParameterDict):
        super().__init__()
        self.mlps = nn.ModuleList(mlp_list)
        self.reset_weights_from_template(template_weights)
        self.name_to_batch_idx = {  # the index of the MLP the param is from which = its batch index
            name: self.get_name_idx(name) for name, _ in self.named_parameters()
        }

    @staticmethod
    def get_name_idx( name: str) -> int:
        if idx := re.search(r"\.(\d+)\.", name):
            return int(idx.group(1))
        raise ValueError(f"Parameter name {name} does not match expected format.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != len(self.mlps):
            raise ValueError(
                f"Input batch size {x.shape[0]} must match the number of MLPs {len(self.mlps)}"
            )
        outputs = [self.mlps[i](x[i]) for i in range(x.shape[0])]
        return torch.stack(outputs, dim=0) + x  # residual connection
    
    @torch.compile()
    def calculate_coeffs(self, beta_vecs, eta_vecs, theta_vecs):
        # ---------- prefix / suffix cumulative products ----------
        p_prefix = beta_vecs.cumprod(1)                                 # p_t  = β_0⋯β_t: B, T
        p_suffix = beta_vecs.flip(1).cumprod(1).flip(1)                 # β_t⋯β_{T-1}, B: T

        q_prefix = eta_vecs.cumprod(1)                                  # q_t  = η_0⋯η_t: B, T
        q_suffix = eta_vecs.flip(1).cumprod(1).flip(1)                  # η_t⋯η_{T-1}: B, T

        p_T = p_prefix[:, -1]                                           # final p_T:  B
        q_T = q_prefix[:, -1]                                           # final q_T:  B

        # ---------- w_k  (shape: T) ----------
        w = (p_suffix / beta_vecs) * q_prefix                           # β^{T-1-k} · η^{k+1}: B, T

        # ---------- A_T (scalar) ----------
        A_T = w.sum(dim=1)                                              # A_T:  B

        # ---------- B_{T,j}  (shape: T) ----------
        partial_sum = w.flip(1).cumsum(dim=1).flip(1)                   # Σ_{k=j}^{T-1} w_k: B, T
        B_coeffs = theta_vecs * partial_sum / q_prefix                  # −θ · B_{T,j}: B, T

        # ---------- D_{T,j}  (shape: T) ----------
        D_coeffs = theta_vecs * q_suffix                                # −θ · D_{T,j}: B, T
        
        return p_T, q_T, A_T, B_coeffs, D_coeffs
    
    # @torch.compile()
    def update_memory(self, current_params, surprises, keys, values, beta_vecs, eta_vecs, theta_vecs):
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

            outputs = functional_call(self, current_params, keys)

            # GRAD DESC FOR MEMORY UPDATE
            # scaling loss by B_coeffs
            scaled_outputs = outputs * torch.sqrt(B_coeffs.unsqueeze(-1))
            scaled_values = values * torch.sqrt(B_coeffs.unsqueeze(-1))
            
            mem_loss = nn.functional.mse_loss(scaled_outputs, scaled_values)    # Compute loss
            grads = torch.autograd.grad(outputs=mem_loss, inputs=current_params.values(), retain_graph=True)

            # p_T * M_0 + A_T[idx] * S_0 + gradient_term
            def update_param(name, grad):
                if grad is None:
                    return current_params[name].detach().clone()
                idx = self.name_to_batch_idx[name]
                old_param = current_params[name]  # (*param_shape)
                surprise = surprises.get(name, torch.zeros_like(old_param))
                # M_T (param) = p_T·M_0 + A_T·S_0 − Σ_j θ·B_{T,j}·u_j
                return p_T[idx] * old_param + A_T[idx] * surprise - grad

            new_params = {
                name: update_param(name, grad)
                for name, grad in zip(current_params.keys(), grads)
            }

            # GRAD DESC FOR SURPRISE UPDATE
            # scaling loss by D_coeffs
            scaled_outputs = outputs * torch.sqrt(D_coeffs.unsqueeze(-1))
            scaled_values = values * torch.sqrt(D_coeffs.unsqueeze(-1))

            surprise_loss = nn.functional.mse_loss(scaled_outputs, scaled_values)    # Compute loss
            grads = torch.autograd.grad(outputs=surprise_loss, inputs=current_params.values())

            # q_T * S_0 - Σ_j θ·D_{T,j}·u_j
            def update_suprise(name, grad):
                if grad is None:
                    return torch.zeros_like(current_params[name])
                idx = self.name_to_batch_idx[name]
                old_surprise = surprises.get(name, torch.zeros_like(current_params[name]))
                # S_T (surprise) = q_T·S_0 − Σ_j θ·D_{T,j}·u_j
                return q_T[idx] * old_surprise - grad

            surprises = {
                name: update_suprise(name, grad)
                for name, grad in zip(current_params.keys(), grads)
            }

            return surprises, mem_loss, new_params
        
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
                 hidden_dim = 32, nu = 0.01,
                 use_global_sw = False, num_global_tokens = 0,
                 use_conv=False):
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
        self.use_conv = use_conv
        self.nu = nu

        self.K = nn.Linear(emb_dim, emb_dim, bias = False)  # Mapping to keys
        self.Q = nn.Linear(emb_dim, emb_dim, bias = False)  # Mapping to queries
        self.V = nn.Linear(emb_dim, emb_dim, bias = False)  # Mapping to values
        
        self.use_global_sw = use_global_sw
        self.num_global_tokens = num_global_tokens
        
        if self.use_global_sw and self.num_global_tokens > 0:
            self.persistent_tokens = nn.Parameter(
            torch.empty(self.num_global_tokens, self.emb_dim)
            )
            torch.nn.init.normal_(self.persistent_tokens, mean=0.0, std=0.2)

        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.xavier_uniform_(self.K.weight)
        torch.nn.init.xavier_uniform_(self.V.weight)
        torch.nn.init.xavier_uniform_(self.Q.weight)

        if self.use_conv:
            # Depthwise-separable convolutions
            self.conv_q = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding='same', groups=emb_dim, bias=False)
            self.conv_k = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding='same', groups=emb_dim, bias=False)
            self.conv_v = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding='same', groups=emb_dim, bias=False)

            torch.nn.init.xavier_uniform_(self.conv_q.weight)
            torch.nn.init.xavier_uniform_(self.conv_k.weight)
            torch.nn.init.xavier_uniform_(self.conv_v.weight)

        self.alpha = nn.Linear(emb_dim, 1, bias=False)
        self.eta = nn.Linear(emb_dim, 1, bias=False)
        self.theta = nn.Linear(emb_dim, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.alpha.weight)
        torch.nn.init.xavier_uniform_(self.eta.weight)
        torch.nn.init.xavier_uniform_(self.theta.weight)

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
                nn.SiLU()
            ]
            for k in range(self.n_layers - 2):
                layers += [
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.SiLU()
                ]
            layers.append(nn.Linear(self.hidden_dim, self.emb_dim))
        layers.append(nn.LayerNorm(self.emb_dim))  # Layer normalization
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
        parallel_mlps = ParallelMLPs(mlps, self.mlp_template_weights).to(device)  # type: ignore
        self.mlps_processor = torch.compile(parallel_mlps)  # type: ignore
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
        queries = normalize(queries) # Normalize after convolution

        return functional_call(self.mlps_processor, self.mlp_states[-1], queries)

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
        
        keys = normalize(keys)
        values = normalize(values)

        # Computing β, η, & θ vectors which are gated between 0 and 1: B, N, D -> B, N
        beta_vec = 1 - self.sigmoid(self.alpha(keys)).squeeze(-1)  # (B, N)
        eta_vec = self.sigmoid(self.eta(keys)).squeeze(-1)  # (B, N)
        theta_vec = self.sigmoid(self.theta(keys)).squeeze(-1)  # (B, N)

        self.surprise, losses, next_mlp_params = self.mlps_processor.update_memory(  # type: ignore
            self.mlp_states[-1], self.surprise, keys, values, beta_vec, eta_vec, theta_vec
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
                if "weight" in name:
                    torch.nn.init.normal_(new_param.data, mean=0, std=0.02)
                elif "bias" in name: # Bias exists but is None (e.g. bias=False in Linear)
                    torch.nn.init.zeros_(new_param.data) # Or some other default
                else: # Fallback for other params
                    raise ValueError(f"Unexpected parameter name: {name}")
                # new param with no gradients
                template_weights[clean_name] = new_param.detach()
        del reference_mlp
        return template_weights
    
    def train_initial_mlp(self):
        mlp_updates = {}
        for mlp in self.mlps_processor.mlps:  # type: ignore
            for name, param in mlp.named_parameters():
                clean_name = name.replace('.', '_')
                mlp_updates[clean_name] = mlp_updates.get(clean_name, []) + [param.data.clone()]
        for name, params in mlp_updates.items():
            assert name in self.mlp_template_weights, f"Parameter {name} not found in template weights, but in MLPs."
            avg_param = torch.mean(torch.stack(params), dim=0) if len(params) > 1 else params[0]
            old_weight = self.mlp_template_weights[name].data
            self.mlp_template_weights[name].data = (1 - self.nu) * old_weight + self.nu * avg_param
