import torch
from torch import nn
from torch import Tensor
from typing import Callable
from .utils.time_embeddings import SinusoidalPosEmb


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        num_categories: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        encode_time: bool = True,
        n_hidden_layers: int = 2,
        probs_parametrization_fn: Callable[[Tensor, Tensor], Tensor] = lambda logits, x: torch.softmax(logits, dim=-1),
    ):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.num_categories = num_categories
        self.embedding = nn.Embedding(num_categories, embed_dim)
        self.encode_time = encode_time
        self.probs_parametrization_fn = probs_parametrization_fn
        
        L = 1
        for s in input_shape:
            L *= s
        
        self.mlp = nn.Sequential(
            nn.Linear(L * embed_dim, hidden_dim),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                for _ in range(n_hidden_layers - 1)
            ],
            nn.Linear(hidden_dim, L * num_categories),
        )
        
        if self.encode_time:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 2 * embed_dim),
            )
        
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        x: (B, input_shape, num_categories)
        t: (B)
        
        returns: (B, input_shape, num_categories)
        """
        B = x.shape[0]
        x = x.reshape(B, -1, self.num_categories)
        L = x.shape[1]
        
        x_emb = x @ self.embedding.weight # Shape: (B, L, embed_dim)
        
        if self.encode_time:
            t_emb = self.time_mlp(t) # Shape: (B, 2 * embed_dim)
            t_scale, t_shift = torch.chunk(t_emb, 2, dim=-1)
            x_emb = x_emb * t_scale[:, None, :] + t_shift[:, None, :]
        
        mlp_input = x_emb.flatten(start_dim=1)
        logits: Tensor = self.mlp(mlp_input).reshape(B, L, self.num_categories)
        
        probs = self.probs_parametrization_fn(logits, x)
        return probs.reshape(B, *self.input_shape, self.num_categories)
