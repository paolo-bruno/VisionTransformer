""" Hierarchical Visual Transformer (HVC) in PyTorch

A PyTorch implement of Hierarchical Visual Transformer as described in
'Scalable Visual Transformers with Hierarchical Pooling' - https://arxiv.org/abs/2103.10619

Traditional blocks of the ViT are imported from my implementation of the ViT
The implementation follows some defs and templates for compatibility with timm library's scripts,
details in https://github.com/rwightman/pytorch-image-models
"""
import math
import logging
from copy import deepcopy

import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg
from timm.models.registry import register_model

from vision_transformer import TransformerEncoder


_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': InterpolationMode.BICUBIC, 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # List of the configuration available
    'my_hvt_224': _cfg(
        url="://data/pretrained_weights_hvt_224.pth"
    )
}


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
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4, pool_kernel_size=3,
                 pool_stride=2, num_transformers=4, n: list = None):
        super().__init__()
        self.first_trasformer = TransformerEncoder(embed_dim, num_heads, mlp_ratio)
        self.max_pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
        # the number of patches decrease after the pooling layer, it must be recalculated
        n.append(math.floor((n.pop() - (pool_kernel_size - 1) - 1) / pool_stride + 1))

        self.pos_embed = nn.Parameter(torch.zeros(1, n[-1], embed_dim))
        self.last_transformers = nn.Sequential(
            *[TransformerEncoder(embed_dim, num_heads, mlp_ratio) for i in range(num_transformers - 1)]
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
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4, num_stages=3, drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = LinearProjAndFlattenedPatches_noClsToken(img_size, patch_size, in_chans, embed_dim)
        self.n = [img_size ** 2 // patch_size ** 2]
        self.stages = nn.Sequential(
            *[Stage(embed_dim, mlp_ratio=4, pool_kernel_size=3, pool_stride=2, num_transformers=4, n=self.n)
              for i in range(num_stages)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.average_pool = nn.AvgPool1d(kernel_size=self.n[-1])
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        z = self.patch_embed(x)
        z = self.stages(z)
        z = self.average_pool(self.norm(z).transpose(-2, -1)).transpose(-2, -1)
        # forward to a head
        y = self.head(z).squeeze(1)
        return y


def _init_hvt_weigts():
    pass


def _create_hierarchical_visual_transformer(variant, pretrained=False, **kwargs):
    default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']

    num_classes = kwargs.pop('num_classes', default_num_classes)

    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Hierarchical Visual Transformer models.')

    model = build_model_with_cfg(
        HierarchicalVisualTransformer,
        variant,
        pretrained,
        default_cfg=default_cfg,
        num_classes=num_classes,
        **kwargs)

    return model


@register_model
def my_hvt_224(pretrained=False, **kwargs):
    """ My Hierarchical Visual Transformer (HVC) from original paper (https://arxiv.org/abs/2103.10619).

    HVT with image_size of 224 suitable for use by the timm library
    (https://github.com/rwightman/pytorch-image-models) and his script for train, validation and inference
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_hierarchical_visual_transformer('my_hvt_224', pretrained=pretrained, **model_kwargs)
    return model
