""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929
The implementation follows some defs and templates for compatibility with timm library's scripts,
details in https://github.com/rwightman/pytorch-image-models
"""
import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg
from timm.models.registry import register_model

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
    'my_vit_base_patch16_224': _cfg(
        url=""
    ),
    'my_vit_base_patch16_384': _cfg(
        url=""
    ),
    'my_vit_large_patch16_224': _cfg(
        url=""
    )
}


class LinearProjectionAndFlattenedPatches(nn.Module):
    """
    Linear projection of Flattened patches given an image

    Reshape the input image (C, H, W) into a sequence of flattened 2D patches.
    Then map patches to a a vector with size D thanks to a trainable linear projection
    Prepend a learnable embedding to the sequence of embedded patches
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_size, patch_size),
                              stride=(patch_size, patch_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.n = img_size**2 // patch_size**2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n+1, embed_dim))

    def forward(self, x):
        b, c, h, w = x.shape
        # Conv allow to convert image x (B, C, Hi, Wi) -> (B, D, Ho, Wo), in this manner we obtain
        # flattened 2D patches x1,x2,...,xn if we consider that (B, D, Ho, Wo) can be flattened as (B, D, N), whit
        # N = Ho*Wo
        x = self.proj(x)
        # now flatten: (B, D, Ho, Wo) -> (B, D, N)
        x = torch.flatten(x, start_dim=2)
        # to reproduce the order of the paper: (B, N, D) is the shape for each batch of flattened patches
        x = torch.transpose(x, 1, 2)

        x = torch.cat((self.cls_token.repeat(b, 1, 1), x), dim=1)
        return x + self.pos_embed


class ScaledDotProductAttentions(nn.Module):
    """
    Compute scaled dot produt attention

    q, k, v have shape (b, n, num_heads, d_h)
    """
    def __init__(self, d_h):
        super().__init__()
        self.radq_d_h = d_h**(-1/2)

    def forward(self, q, k, v):
        attentions = q @ k.transpose(-2, -1) * self.radq_d_h
        attentions = F.softmax(attentions, dim=-1) @ v
        return attentions


class MultiHeadAttention(nn.Module):
    """
    Multihead self-attention

    Compute k self-attention operations (heads) in parallel and project their concatenated outputs.
    For details:
        See appendix A - https://arxiv.org/abs/2010.11929
        See the paper: 'Attention Is All You Need' - https://arxiv.org/abs/1706.03762
    """
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.d_h = embed_dim//self.num_heads

        self.qkv = nn.Linear(embed_dim, 3 * self.d_h * self.num_heads)
        self.attentions = ScaledDotProductAttentions(self.d_h)
        self.proj = nn.Linear(self.num_heads * self.d_h, embed_dim)

    def forward(self, z):
        b, n, d = z.shape
        z = self.qkv(z).reshape(b, n, 3, self.num_heads, -1)
        z = z.permute(2, 0, 3, 1, 4)  # as in timm ViT implementation
        q, k, v = z[0], z[1], z[2]
        multihead_self_attention = self.attentions(q, k, v).transpose(1, 2).reshape(b, n, d)
        multihead_self_attention = self.proj(multihead_self_attention)
        return multihead_self_attention


class FeedForward(nn.Module):
    """
    MLP block

    Contains 2 layers and a GELU non linearity, as described in eq. 2, 3 - https://arxiv.org/abs/2010.11929
    """
    def __init__(self, embed_dim=768, mlp_ratio=4):
        super().__init__()
        self.dim_inner_layer = embed_dim*mlp_ratio
        self.fc1 = nn.Linear(embed_dim, self.dim_inner_layer)
        self.fc2 = nn.Linear(self.dim_inner_layer, embed_dim)

    def forward(self, x):
        z = self.fc2(F.gelu(self.fc1(x)))
        return z


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder

    Composed by tha alternation of multihead self-attention layers and MLP block (Feed Forward)
    LayerNorm is applied before every block, and residual connection after every block.
    """
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = FeedForward(embed_dim, mlp_ratio)

    def forward(self, z):
        z = self.attn(self.norm1(z)) + z
        z = self.mlp(self.norm2(z)) + z
        return z


class MLP_head(nn.Module):
    """
    Classification head attached to z0 (cls_token)

    Implemented by a MLP with one hidden layer
    """
    def __init__(self, embed_dim=768, num_classes=1000):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, num_classes)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, z):
        y = self.linear1(self.norm(z))
        return y


class VisionTransformer(nn.Module):
    """
    Vision Transformer

    A PyTorch imp of: 'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = LinearProjectionAndFlattenedPatches(img_size, patch_size, in_chans, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerEncoder(embed_dim, num_heads, mlp_ratio) for i in range(depth)]
        )
        self.head = MLP_head(embed_dim, num_classes)

    def forward(self, x):
        z = self.patch_embed(x)
        z = self.blocks(z)
        y = self.head(z[:, 0])
        return y


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']

    num_classes = kwargs.pop('num_classes', default_num_classes)

    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Hierarchical Visual Transformer models.')

    model = build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        default_cfg=default_cfg,
        num_classes=num_classes,
        **kwargs)

    return model


@register_model
def my_vit_base_patch16_224(pretrained=False, **kwargs):
    """ My Vision Transformer (ViT) from original paper (https://arxiv.org/abs/2010.11929)

    ViT base suitable for use by the timm library
    (https://github.com/rwightman/pytorch-image-models)
    """
    model_kwargs = dict(embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('my_vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def my_vit_base_patch16_384(pretrained=False, ** kwargs):
    """ My Vision Transformer (ViT) from original paper (https://arxiv.org/abs/2010.11929)

    ViT base with image_size 384 (in the base version is 224) suitable for use by the timm library
    """
    model_kwargs = dict(img_size=384, embed_dim=768, depth=12, n_heads=12, **kwargs)
    model = _create_vision_transformer('my_vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def my_vit_large_patch16_224(pretrained=False, ** kwargs):
    """ My Vision Transformer (ViT) from original paper (https://arxiv.org/abs/2010.11929)

    ViT large suitable for use by the timm library
    """
    model_kwargs = dict(embed_dim=1024, depth=24, n_heads=16, **kwargs)
    model = _create_vision_transformer('my_vit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model
