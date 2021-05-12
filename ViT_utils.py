""" Vision Transformer utils
A PyTorch utils for Vision Transformers (described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929)
"""
import torchvision
from torchvision.transforms import transforms


def _cfg(**kwargs):
    return {**kwargs}


cfg = {'vit_base_patch16_224': _cfg(embed_dim=768, depth=12, n_heads=12),   # with weights ported from official Google
                                                                            # JAX impl
       'vit_large_patch16_224': _cfg(embed_dim=1024, depth=24, n_heads=16),
       'vit_huge_patch16_224': _cfg(embed_dim=1280, depth=32, n_heads=16)}  # no weights available?!

ViT_transform = transforms.Compose([
                    transforms.Resize(size=248, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])

adapt_to_my_ViT = {
    'cls_token': 'patch_embed.cls_token',
    'pos_embed': 'patch_embed.pos_embed',
    'norm.weight': 'head.norm.weight',
    'norm.bias': 'head.norm.bias',
    'head.weight': 'head.linear1.weight',
    'head.bias': 'head.linear1.bias',
}


def adapt_state_dict(original_dict: dict, new_items=None):
    if new_items is None:
        new_items = adapt_to_my_ViT
    for key in new_items.keys():
        original_dict[new_items[key]] = original_dict.pop(key)
    return original_dict
