import torch
import torch.nn as nn
from modules import TransformerEncoder
from modules import LinearProjectionAndFlattenedPatches


class VisionTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_cls=1000, n_heads=8, depth=12):
        super().__init__()
        self.proj_flatten = LinearProjectionAndFlattenedPatches()
        self.transformers_encoders = nn.Sequential(
            *[TransformerEncoder(embed_dim, n_heads) for i in range(depth)]
        )
        self.norm = nn.LayerNorm()
        self.mlp_head = nn.Linear(embed_dim, num_cls)

    def forward(self, x):
        z = self.proj_flatten(x)
        z = self.transformers_encoders(z)
        z = self.norm(z)
        y = self.mlp_head(z[0])
        return y
