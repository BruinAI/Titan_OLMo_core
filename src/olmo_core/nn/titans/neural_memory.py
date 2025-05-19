import torch
from torch import gt, nn, unsqueeze
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
        
    def updatex(self, x):
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
    
    # TODO: better understand scan channel formatting + optimize code (very slow)
    def update(self, x):
        self.mlp_reset = False
        z = x.detach()

        # Evaluate the corresponding keys and values
        keys = normalize(self.silu(self.K(z)))
        vals = normalize(self.silu(self.V(z)))

        with torch.enable_grad():  # Enable gradients for this specific block
            # Dict[parameter name] -> (seq len, *param shape)
            grads = per_sample_grads(self.mlps_processor, dict(self.mlps_processor.named_parameters()), keys, vals)  # type: ignore
            for (name, param), grad_name in zip(self.mlps_processor.named_parameters(), grads):  # type: ignore
                grad = grads[grad_name]
                if grad is None or name[0] in ['K', 'V']:
                    continue
                unsqueezed = False
                if len(grad.shape) == 2:
                    grad = grad.unsqueeze(1)
                    unsqueezed = True
                # grad: (seq_len, *param shape)
                if name not in self.surprise:
                    self.surprise[name] = torch.zeros_like(grad)
                base_surprise = self.surprise[name]
                # surprises: s_t = eta * s_{t-1} - theta * grad
                eta_t = torch.ones_like(grad, device=grad.device) * self.eta  # (seq_len, *param shape)
                each_surprise = -grad * self.theta
                assert eta_t.shape == each_surprise.shape, f"eta_t shape {eta_t.shape} != each_surprise shape {each_surprise.shape}"
                surprises = scan(eta_t, each_surprise)

                eta_cum_prod = torch.cumprod(eta_t, dim=0)  # not sure if this is parallel but also not sure if it matters
                base_surprise = base_surprise * eta_cum_prod
                surprises += base_surprise

                beta_t = torch.ones_like(surprises, device=grad.device) * (1 - self.alpha)  # (seq_len, *param shape)
                assert beta_t.shape == surprises.shape, f"beta_t shape {beta_t.shape} != surprises shape {surprises.shape}"
                updates = scan(beta_t, surprises)
                if unsqueezed:
                    updates = updates.squeeze(1)  # type: ignore
                param.data = param.data + updates.sum(dim=0)  # type: ignore
                self.surprise[name] = surprises[-1]

            return 0
