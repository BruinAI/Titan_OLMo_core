import torch
from torch import nn
from torch.nn.functional import normalize
from torch.func import functional_call
from typing import List


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

        # Mapping to keys
        self.K = nn.Linear(emb_dim, emb_dim, bias = False)

        # Mapping to values
        self.V = nn.Linear(emb_dim, emb_dim, bias = False)

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
                        nn.Linear(self.emb_dim, self.hidden_dim),
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

    def retrieve(self, x):
        return functional_call(self, dict(self.named_parameters()), x)

    def forward(self, x):
        if self.mlps_processor is None:
            raise RuntimeError("MLPs not initialized. Call init_mlp(batch_size) first.")
        return self.mlps_processor(x)

    def update(self, x):

        z = x.detach()

        # Evaluate the corresponding keys and values
        keys = normalize(self.silu(self.K(z)))
        vals = self.silu(self.V(z))

        with torch.enable_grad():  # Enable gradients for this specific block
            # Propagate the keys through the model
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