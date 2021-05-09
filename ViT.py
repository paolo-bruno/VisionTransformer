import torch
import torch.nn as nn
from modules import MultiHeadAttention
from modules import FeedForward
from modules import LinearProjectionAndFlattenedPatches


class VisionTransformer(nn.Module):
    def __init__(self, embed_dim, num_cls, n_heads):
        super().__init__()
        self.proj_flatten = MultiHeadAttention(n_heads)
        self.mlp_head = nn.Linear(embed_dim, num_cls)

    def forward(self, x):
        z = self
        y = self.mlp_head(z)
