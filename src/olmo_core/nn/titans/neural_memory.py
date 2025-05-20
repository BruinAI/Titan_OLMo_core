import torch
from torch import nn, unsqueeze
from torch.nn.functional import normalize
from torch.func import functional_call, grad, vmap
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
    """Find the next power of 2 that's at least 32. Associative Scan demands at least 32 on CUDA"""
    if x < 1:
        raise ValueError("Input must be a positive integer.")
    power_of_2 = 1 << (x - 1).bit_length()
    return max(power_of_2, 2)  # Ensure minimum length of 2


class ParallelMLPs(nn.Module):
    """
    ParallelMLPs is a wrapper for multiple MLPs that allows for parallel processing of inputs with torch.compile.
    """
    def __init__(self, mlp_list: List[nn.Module]):
        super().__init__()
        self.mlps = nn.ModuleList(mlp_list)
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != len(self.mlps):
            raise ValueError(
                f"Input batch size {x.shape[0]} must match the number of MLPs {len(self.mlps)}"
            )
        outputs = [self.mlps[i](x[i]) for i in range(x.shape[0])]
        
        return torch.stack(outputs, dim=0)
    
    # TODO: add learned init weights, 0 weights cannot learn for mlp w/ >= 2 layers
    def init_weights(self, mean=0, std=10):
        for mlp in self.mlps:
            for layer in mlp.modules():  # Iterate through modules, not parameters
                if isinstance(layer, nn.Linear):
                    torch.nn.init.normal_(layer.weight, mean=mean, std=std)
                    if layer.bias is not None:
                        torch.nn.init.normal_(layer.bias, mean=mean, std=std)

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
        self.mlp_reset = True

        self.K = nn.Linear(emb_dim, emb_dim, bias = False)  # Mapping to keys
        self.Q = nn.Linear(emb_dim, emb_dim, bias = False)  # Mapping to queries
        self.V = nn.Linear(emb_dim, emb_dim, bias = False)  # Mapping to values

        torch.nn.init.xavier_uniform_(self.K.weight)
        torch.nn.init.xavier_uniform_(self.V.weight)

        self.alpha = alpha
        self.eta = eta
        self.theta = theta

        self.silu = nn.SiLU()
        self.surprise = {}

    # Ideally only called once, compiling slows it down, pad inputs instead
    def init_mlp(self, batch_size):
        del self.mlps_processor
        
        device = next(self.parameters()).device # Adding CUDA support
        
        mlps = []
        for i in range(batch_size):
            # Building the layers in the MLP
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
            # Create the MLP as a sequential model
            mlp = nn.Sequential(
                *layers
            )
            mlps.append(mlp)  # adding the mlp to the list
        parallel_mlps = ParallelMLPs(mlps).to(device)
        parallel_mlps.init_weights(mean=0, std=10) 
        self.mlps_processor = torch.compile(parallel_mlps)  # type: ignore

    def reset_mlps(self):
        if self.mlps_processor is not None:
            self.mlps_processor.init_weights()  # type: ignore
            self.mlp_reset = True

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

        with torch.enable_grad():
            # compute per-sample gradients: dict[name] → (seq_len, *param_shape)
            grads = per_sample_grads(
                self.mlps_processor,
                dict(self.mlps_processor.named_parameters()),
                keys,
                vals,
            )

            for (name, param), grad_name in zip(
                self.mlps_processor.named_parameters(), grads
            ):
                grad = grads[grad_name]  # (T, ...)
                if grad is None or name[0] in ['K', 'V']:
                    continue

                # ------ reshape to (*param_shape, T) -------
                squeezed = False
                if grad.ndim == 2:  # (T, hidden_dim) → (T, 1, hidden_dim)
                    grad = grad.unsqueeze(1)
                    squeezed = True
                # move time-axis (0) to the last dim:
                perm = list(range(1, grad.ndim)) + [0]
                grad = grad.permute(*perm)  # now shape (*param_shape, T)

                # ------ closed-form coefficients -------------
                T = grad.shape[-1]
                device, dtype = grad.device, grad.dtype
                p = 1.0 - self.alpha
                η = self.eta
                θ = self.theta

                # time indices 1…T
                ar = torch.arange(1, T+1, device=device, dtype=dtype)

                # powers of p and η
                p_t = p ** T
                η_t = η ** T
                pows_p   = p ** ar        # [p¹, p², …, pᵀ]
                pows_η   = η ** ar        # [η¹, η², …, ηᵀ]

                # base states
                M0 = param.data                            # (*param_shape)
                S0 = self.surprise.get(name, torch.zeros_like(M0))

                # 1) final surprise:  S_T = ηᵀ S₀  − θ Σ_{j=1}ᵀ η^{T−j} grad_j
                S_coeffs = η ** (T - ar)                   # [η^{T−1}, η^{T−2}, …, η⁰]
                S_sum    = torch.sum(S_coeffs * grad, dim=-1)
                S_T      = η_t * S0 - θ * S_sum

                # 2) final memory:   M_T = pᵀ M₀
                #                 + η (pᵀ − ηᵀ)/(p−η) · S₀
                #                 − θ Σ_{j=1}ᵀ (p^{T−j+1}−η^{T−j+1})/(p−η) · grad_j
                A_pref   = η * (p_t - η_t) / (p - η)        # scalar
                coeffs   = (pows_p - pows_η) / (p - η)     # shape (T,)
                M_sum    = torch.sum(coeffs * grad, dim=-1)
                M_T      = p_t * M0 + A_pref * S0 - θ * M_sum

                # write back
                self.surprise[name] = S_T
                param.data = M_T if not squeezed else M_T.squeeze(1)

            return 0  # or return whatever you need
