import torch
from torch import nn, unsqueeze
from torch.nn.functional import normalize
from torch.func import functional_call, grad, vmap
from xdist.scheduler import each
if torch.cuda.is_available():
    from accelerated_scan.warp import scan  # can only be used on CUDA
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
    """
    def __init__(self, mlp_list: List[nn.Module]):
        super().__init__()
        self.mlps = nn.ModuleList(mlp_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != len(self.mlps):
            raise ValueError(
                f"Input batch size {x.shape[0]} must match the number of MLPs {len(self.mlps)}"
            )
        outputs = [self.mlps[i](x[i]) for i in range(x.shape[0])]
        return torch.stack(outputs, dim=0)
    
    # TODO: add learned init weights, 0 weights cannot learn for mlp w/ >= 2 layers
    def init_weights(self):
        for mlp in self.mlps:
            for layer in mlp.parameters():
                if isinstance(layer, nn.Linear):
                    torch.nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

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
        parallel_mlps = ParallelMLPs(mlps)
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

        with torch.enable_grad():  # Enable gradients for this specific block
            # Dict[parameter name] -> (seq len, *param shape)
            grads = per_sample_grads(self.mlps_processor, dict(self.mlps_processor.named_parameters()), keys, vals)  # type: ignore
            for (name, param), grad_name in zip(self.mlps_processor.named_parameters(), grads):  # type: ignore
                grad = grads[grad_name]  # getting grad: (seq_len, *param shape)
                if grad is None or name[0] in ['K', 'V']:  # skip K and V and not-computed grads
                    continue

                # checking if grad is 2D (like biases) and converting to 3D for scan
                unsqueezed = False
                if len(grad.shape) == 2:  # (seq_len, hidden_dim)
                    grad = grad.unsqueeze(1)  # (seq_len, 1, hidden_dim)
                    unsqueezed = True
                grad = grad.permute(1, 2, 0)  # (*param shape, seq_len)

                # 
                seq_len = grad.shape[-1]
                pad_seq_len = next_power_of_2(seq_len)  # find the next power of 2
                if pad_seq_len > seq_len:
                    pad = torch.zeros(grad.shape[0], grad.shape[1], pad_seq_len - seq_len, device=grad.device)
                    grad = torch.cat([grad, pad], dim=-1)

                
                base_surprise = self.surprise.get(name, torch.zeros_like(grad))  # (*param shape, seq_len)
                eta_t = torch.ones_like(grad, device=grad.device) * self.eta  # (*param shape, seq_len)
                each_surprise = -grad * self.theta  # multiplying by -theta to scale grads according to the formula
                # surprises: s_t = eta * s_{t-1} - theta * grad
                # SCAN EXPECTS INPUTS OF (B, C, T) and OUTPUTS (B, C, T)
                surprises = scan(eta_t, each_surprise)  # (*param shape, seq_len)

                # Adding the base surprise to the surprises and storing results
                eta_cumprod = torch.cumprod(eta_t, dim=0)
                base_surprise = base_surprise * eta_cumprod
                surprises += base_surprise
                self.surprise[name] = surprises[-1]

                # calculating the updates
                beta_t = torch.ones_like(surprises, device=grad.device) * (1 - self.alpha)  # (*param shape, seq_len)
                updates = scan(beta_t, surprises)

                # squeezing the updates if they are 2D (like biases), (hidden_dim, 1, seq_len) -> (hidden_dim, seq_len)
                if unsqueezed:
                    updates = updates.squeeze(1)  # type: ignore
                    beta_t = beta_t.squeeze(1)  # squeeze beta_t as well for cumprod update
                
                # Adding the base memory to the updates and storing results
                beta_cumprod = torch.cumprod(beta_t, dim=-1)
                expanded_base_memory = param.data.unsqueeze(-1).expand_as(beta_cumprod)  # (*param shape, seq_len)
                base_memory = expanded_base_memory * beta_cumprod
                update_data = base_memory + updates  # type: ignore
                if seq_len < pad_seq_len:
                    update_data = update_data[..., :seq_len]
                param.data = update_data.sum(dim=-1)  # type: ignore
                

            return 0  # TODO: return loss?
