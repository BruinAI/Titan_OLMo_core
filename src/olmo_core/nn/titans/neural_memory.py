import torch
from torch import nn, unsqueeze
from torch.nn.functional import normalize
from torch.func import functional_call, grad, vmap
from wandb.cli import beta
from xdist.scheduler import each
if torch.cuda.is_available():
    from accelerated_scan.ref import scan  # Temp changed from warp to ref to test memory
else:
    from accelerated_scan.ref import scan
from typing import List


def loss_one(model, param_dict, x, target):
    """
    Loss for each index in seq
    Args:
        model: The model to be used.
        param_dict: The parameters of the model.
        x: The input tensor.  (B x seq_len x emb_dim)
        target: The target tensor.  (B x seq_len x emb_dim)
    """
    logits = functional_call(model, param_dict, x)
    return nn.functional.mse_loss(logits, target)

grad_one = grad(loss_one, argnums=1)  # takes the gradient of the loss w.r.t. the model parameters
per_sample_grads = vmap(grad_one, in_dims=(None, None, 1, 1))  # creates a function to get per sample gradients

def next_power_of_2(x: int) -> int:
    if x < 1:
        raise ValueError("Input must be a positive integer.")
    return 1 << (x - 1).bit_length()  # Find the next power of 2 using bit manipulation


class ParallelMLPs(nn.Module):
    """
    ParallelMLPs is a wrapper for multiple MLPs that allows for parallel processing of inputs with torch.compile.
    It supports a learnable initial state for the weights of the MLPs.
    """
    def __init__(self, mlp_list: List[nn.Module], mlp_template_weights: nn.ParameterDict):
        super().__init__()
        
        if not mlp_list:
            raise ValueError("mlp_list cannot be empty")
        self.mlps = nn.ModuleList(mlp_list)
        self._template_weights = mlp_template_weights

        # Perform initial copy of template weights to instance MLP parameters
        # This uses the same logic as reset_weights_from_template but is done once.
        with torch.no_grad():
            for mlp_instance in self.mlps:
                for name, p_inst in mlp_instance.named_parameters():
                    clean_name = name.replace('.', '_')
                    if clean_name in self._template_weights:
                        template_param = self._template_weights[clean_name]
                        p_inst.data.copy_(template_param.data)
                    else:
                        raise KeyError(
                            f"Instance parameter {name} (cleaned: {clean_name}) not found in template weights "
                            f"during initial data copy. Ensure MLP structures are consistent."
                        )
        
        # TODO: Karen plz check this I don't understand hooks tbh
        # Register hooks ONCE during initialization.
        # This ensures gradients from instance MLPs flow to the template weights.
        for mlp_instance in self.mlps:
            for name, p_inst in mlp_instance.named_parameters():
                clean_name = name.replace('.', '_')
                # We assume clean_name is in _template_weights due to the check above,
                # but for safety, you could re-check or ensure consistency.
                template_param = self._template_weights[clean_name]

                # Define the hook function using a closure to capture the correct template_param
                def make_hook(tmpl_param_to_update):
                    def hook(grad_from_instance):
                        if grad_from_instance is not None: # Check if gradient exists
                            print(f"Updating template parameter {tmpl_param_to_update} with gradient from instance.")
                            if tmpl_param_to_update.grad is None:
                                tmpl_param_to_update.grad = torch.zeros_like(tmpl_param_to_update.data)
                            tmpl_param_to_update.grad.add_(grad_from_instance)
                        # The hook should return None or the original grad.
                        # Returning None is fine if the instance grad isn't modified by the hook.
                        return None
                    return hook

                p_inst.register_hook(make_hook(template_param))

    def reset_weights_from_template(self):
        """
        Copies the learned template weights to all MLP instances.
        This is used for resetting the MLPs to their learned base state.
        Hooks are NOT re-registered here.
        """
        with torch.no_grad(): # Ensure this operation is not tracked by autograd
            for mlp_instance in self.mlps:
                for name, param_instance in mlp_instance.named_parameters():
                    clean_name = name.replace('.', '_')
                    if clean_name in self._template_weights:
                        param_instance.data.copy_(self._template_weights[clean_name].data)
                    else:
                        # This case implies a mismatch if it occurs after successful initialization.
                        raise KeyError(
                            f"Instance parameter {name} (cleaned: {clean_name}) not found in template weights "
                            f"during reset. Ensure MLP structures remain consistent."
                        )

    # The _template_weights are nn.Parameters, so they will be learned.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != len(self.mlps):
            raise ValueError(
                f"Input batch size {x.shape[0]} must match the number of MLPs {len(self.mlps)}"
            )
        outputs = [self.mlps[i](x[i]) for i in range(x.shape[0])]
        return torch.stack(outputs, dim=0)

class NeuralMemory(nn.Module):
    """
    Neural Memory Module Requirements:
    1. can have any amount of mlp layers with same hyperparameters
    2. but has shared, learned K and V matrices ad 
    """

    def __init__(self, emb_dim = 16, n_layers = 2, hidden_dim = 32, alpha = 0.999, eta = 0.60, theta = 0.05):
        super().__init__()

        # Define the layers of the network
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.mlps_processor: nn.Module | None = None
        self.mlp_template_weights: nn.ParameterDict = self.init_mlp_template_weights()
        self.mlp_reset = True

        self.K = nn.Linear(emb_dim, emb_dim, bias = False)  # Mapping to keys
        self.Q = nn.Linear(emb_dim, emb_dim, bias = False)  # Mapping to queries
        self.V = nn.Linear(emb_dim, emb_dim, bias = False)  # Mapping to values

        torch.nn.init.xavier_uniform_(self.K.weight)
        torch.nn.init.xavier_uniform_(self.V.weight)

        self.alpha = alpha
        self.beta = 1 - alpha
        self.eta = eta
        self.theta = theta

        self.silu = nn.SiLU()
        self.surprise = {}

    def build_mlp(self):
        """
        Build the MLP layers based on the specified architecture.
        This function is called during initialization to set up the MLP layers.
        """
        # Define the layers of the network
        if self.n_layers == 1:
            layers = [nn.Linear(self.emb_dim, self.emb_dim)]
        else:
            layers = [
                nn.Linear(self.emb_dim, self.hidden_dim),
                nn.SiLU()
            ]
            for k in range(self.n_layers - 2):
                layers += [
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.SiLU()
                ]
            layers.append(nn.Linear(self.hidden_dim, self.emb_dim))
        return nn.Sequential(*layers)

    def init_mlp_template_weights(self):
        reference_mlp = self.build_mlp()
        template_weights = nn.ParameterDict()
        with torch.no_grad():
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
                
                template_weights[clean_name] = new_param
        del reference_mlp
        return template_weights


    # Ideally only called once, compiling slows it down, pad inputs instead
    def init_mlp(self, batch_size):
        del self.mlps_processor
        device = next(self.parameters()).device
        mlps = []
        for i in range(batch_size):
            mlp = self.build_mlp().to(device)  # build the mlp
            mlps.append(mlp)  # adding the mlp to the list
        parallel_mlps = ParallelMLPs(mlps, self.mlp_template_weights).to(device)
        self.mlps_processor = parallel_mlps  # torch.compile(parallel_mlps)  # type: ignore

    def reset_mlps(self):
        self.mlp_reset = True
        if self.mlps_processor is not None:
            self.mlps_processor.reset_weights_from_template()  # type: ignore

    def retrieve(self, x, new_params=None):
        if new_params is None:
            return self.forward(x)  # same thing as functional_call just clearer code
        else:
            return functional_call(self, dict(self.named_parameters()), x)

    def forward(self, x):
        if self.mlps_processor is None:
            raise RuntimeError("MLPs not initialized. Call init_mlp(batch_size) first.")
        queries = normalize(self.silu(self.Q(x)))
        return self.mlps_processor(queries)
    
    def update(self, x):
        if x.shape[1] > 1:
            return self.update_seq(x)
        else:
            return self.update_single(x)
        
    def update_single(self, x):
        self.mlp_reset = False
        z = x.detach()

        # Evaluate the corresponding keys and values
        keys = normalize(self.silu(self.K(z)))
        vals = normalize(self.silu(self.V(z)))

        with torch.enable_grad():  # Enable gradients for this specific block
            # Propagate the keys through the model
            """
            Updates sequence with accelerated scan
            1. Calculates grad_t
            2. calculates S_t = eta * S_{t-1} - theta * grad_t
            3. calculate M_t = (1-alpha_t) * M_{t-1} + S_t
            """

            keys = self.forward(keys)

            # Calculate the loss || M(keys) - vals ||_2 ^2
            loss = ((keys - vals) ** 2).mean(axis=0).sum()

            # Compute gradients of aux loss w.r.t. NMM's parameters
            # Ensure parameters of NeuralMemory (self.K, self.V, self.layers) have requires_grad=True
            grads = torch.autograd.grad(loss, self.parameters(), allow_unused=True)  # type: ignore

            for (name, param), grad in zip(self.named_parameters(), grads):
                if grad is None or name[0] in ['K', 'V']:
                    continue
                if self.surprise.get(name, None) is None:
                    self.surprise[name] = torch.zeros_like(grad)
                self.surprise[name] = self.surprise[name] * self.eta - self.theta * grad
                param.data = self.alpha * param.data + self.surprise[name]

            return loss
    
    def update_seq(self, x):
        self.mlp_reset = False
        z = x.detach()

        # Evaluate the corresponding keys and values
        keys = normalize(self.silu(self.K(z)))
        vals = normalize(self.silu(self.V(z)))

        with torch.enable_grad():  # Enable gradients for this specific block
            # Dict[parameter name] -> (seq len, *param shape)
            grads = per_sample_grads(self.mlps_processor, dict(self.mlps_processor.named_parameters()), keys, vals)  # type: ignore
            for (name, param), grad_name in zip(self.mlps_processor.named_parameters(), grads):  # type: ignore
                grad = grads[grad_name]  # getting grad: (T, *param shape)
                if grad is None or name[0] in ['K', 'V'] or name.startswith("_template_weights.") or name.startswith("_orig_mod._template_weights."):
                    continue

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
                #                     =  β_{k+1} β_{k+2} … β_{T-1} · η_0 η_1 … η_k
                #
                #   A_T              =  Σ_{k=0}^{T-1} w_k
                #
                #   B_{T,j}          =  (Σ_{k=j}^{T-1} w_k) · q_j⁻¹
                #   D_{T,j}          =  q_T · q_j⁻¹
                #
                #   M_T (=param)     =  p_T·M_0  +  A_T·S_0  −  Σ_j θ·B_{T,j}·u_j
                #   S_T (surprise)   =  q_T·S_0  −  Σ_j θ·D_{T,j}·u_j
                # ================================================================

                T = grad.size(0)                                       # length of sequence  (T, *)
                device = grad.device

                # ---------- constant-valued coefficient vectors ----------
                beta_vec = torch.full((T,), self.beta, device=device)   # β_0 … β_{T-1}
                eta_vec = torch.full((T,), self.eta,  device=device)    # η_0 … η_{T-1}
                theta_vec = torch.full((T,), self.theta, device=device) # θ_0 … θ_{T-1}

                # ---------- prefix / suffix cumulative products ----------
                p_prefix = beta_vec.cumprod(0)                          # p_t  = β_0⋯β_t
                p_suffix = beta_vec.flip(0).cumprod(0).flip(0)          # β_t⋯β_{T-1}

                q_prefix = eta_vec.cumprod(0)                           # q_t  = η_0⋯η_t
                q_suffix = eta_vec.flip(0).cumprod(0).flip(0)           # η_t⋯η_{T-1}

                p_T = p_prefix[-1]                                      # final p_T  (scalar)
                q_T = q_prefix[-1]                                      # final q_T  (scalar)

                # ---------- w_k  (shape: T) ----------
                w = (p_suffix / beta_vec) * q_prefix                    # β^{T-1-k} · η^{k+1}

                # ---------- A_T (scalar) ----------
                A_T = w.sum()

                # ---------- B_{T,j}  (shape: T) ----------
                partial_sum = torch.cumsum(w.flip(0), dim=0).flip(0)    # Σ_{k=j}^{T-1} w_k
                B_coeffs = -theta_vec * partial_sum / q_prefix          # −θ · B_{T,j}

                # ---------- D_{T,j}  (shape: T) ----------
                D_coeffs = -theta_vec * q_suffix                        # −θ · D_{T,j}

                # ---------- initial states ----------
                M_0 = param.data                                        # (*param_shape)
                S_0 = self.surprise.get(name, torch.zeros_like(param))  # (*param_shape)

                # broadcast helper: reshape coeffs to (T, 1, 1, …)
                expand = lambda v: v.view(T, *([1] * M_0.dim()))

                # ---------- final memory  M_T ----------
                gradient_term = (grad * expand(B_coeffs)).sum(dim=0)    # Σ −θ B u
                param.data = p_T * M_0 + A_T * S_0 + gradient_term      # M_T

                # ---------- updated surprise  S_T ----------
                surprise_term = (grad * expand(D_coeffs)).sum(dim=0)    # Σ −θ D u
                self.surprise[name] = q_T * S_0 + surprise_term         # S_T              

            return 0  # TODO: return loss?
