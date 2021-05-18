"""
Hierarchical Visual Transformer (HVC) in PyTorch
A PyTorch implement of Hierarchical Visual Transformer as described in
'Scalable Visual Transformers with Hierarchical Pooling' - https://arxiv.org/abs/2103.10619

Traditional blocks of the ViT are imported from our implementation of the ViT
"""
import torch
import torch.nn as nn
from ViT import TransformerEncoder
import math


class LinearProjAndFlattenedPatches_noClsToken(nn.Module):
    """
    Linear projection of Flattened patches without CLS token concatenation

    Reshape the input image (C, H, W) into a sequence of flattened 2D patches.
    Then map patches to a a vector with size D thanks to a trainable linear projection
    The result is the concatenation between input vector and positional embedding vector
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_size, patch_size),
                              stride=(patch_size, patch_size))
        self.n = img_size**2 // patch_size**2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n, embed_dim))

    def forward(self, x):
        # more details in ViT implementation of the linear projection
        b, c, h, w = x.shape
        x = self.proj(x)
        x = torch.flatten(x, start_dim=2)
        # transpose to have (B, N, D) as the shape for each batch of flattened patches
        x = torch.transpose(x, 1, 2)
        # In HVT there isn't cls_token (and relative concatenation between cls_token and x)
        return x + self.pos_embed


class Stage(nn.Module):
    """
    Stage of the HVT

    The Transformer blocks are partitioned in stages, in each stage is applied a downsampling operation to shrink
    the sequence length.
    1D max pooling is applied to achieve the reduce the length.
    """
    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4, pool_kernel_size=3,
                 pool_stride=2, num_transformers=4, n: list = None):
        super().__init__()
        self.first_trasformer = TransformerEncoder(embed_dim, n_heads, mlp_ratio)
        self.max_pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
        # the number of patches decrease after the pooling layer, it must be recalculated
        n.append(math.floor((n.pop() - (pool_kernel_size - 1) - 1) / pool_stride + 1))

        self.pos_embed = nn.Parameter(torch.zeros(1, n[-1], embed_dim))
        self.last_transformers = nn.Sequential(
            *[TransformerEncoder(embed_dim, n_heads, mlp_ratio) for i in range(num_transformers - 1)]
        )

    def forward(self, z):
        z = self.first_trasformer(z)  # (B, N, D)
        z = self.max_pool(z.transpose(-2, -1)).transpose(-2, -1)
        z = z + self.pos_embed
        z = self.last_transformers(z)
        return z


class HierarchicalVisualTransformer(nn.Module):
    """
    Hierarchical Visual Transformer

    A PyTorch imp of: 'Scalable Visual Transformers with Hierarchical Pooling' - https://arxiv.org/abs/2103.10619
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_cls=1000, embed_dim=768, depth=12, n_heads=12,
                 mlp_ratio=4, num_stages=3):
        super().__init__()
        self.patch_embed = LinearProjAndFlattenedPatches_noClsToken(img_size, patch_size, in_chans, embed_dim)
        self.n = [img_size ** 2 // patch_size ** 2]
        self.stages = nn.Sequential(
            *[Stage(embed_dim, n_heads=12, mlp_ratio=4, pool_kernel_size=3, pool_stride=2,
                    num_transformers=4, n=self.n) for i in range(num_stages)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.average_pool = nn.AvgPool1d(kernel_size=self.n[-1])
        self.linear = nn.Linear(embed_dim, num_cls)

    def forward(self, x):
        z = self.patch_embed(x)
        z = self.stages(z)
        z = self.average_pool(self.norm(z).transpose(-2, -1)).transpose(-2, -1)
        # forward to a head
        y = self.linear(z).squeeze(1)
        return y
