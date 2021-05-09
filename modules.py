import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProjectionAndFlattenedPatches(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_size,), stride=(patch_size, ))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.n = img_size**2 / patch_size**2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n+1, embed_dim))

    def forward(self, x):
        b, c, h, w = x.shape
        # Conv allow to convert image x (B, C, Hi, Wi) -> (B, D, Ho, Wo), in this manner we obtain
        # flattened 2D patches x1,x2,...,xn if we consider that (B, D, Ho, Wo) can be flattened as (B, D, N), whit
        # N = Ho*Wo
        x = self.conv(x)
        # now flatten: (B, D, Ho, Wo) -> (B, D, N)
        x = torch.flatten(x, start_dim=2)
        # to reproduce the order of the paper: (B, N, D) is the shape for each batch of flattened patches
        x = torch.transpose(x, 1, 2)

        x = torch.cat((self.cls_token.repeat(b, 1, 1), x), dim=1)
        return x


class ScaledDotProductAttentions(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(q, k, v):
        attentions = q * k.transpose() / q.shape[0]**(1/2)
        attentions = F.softmax(attentions)*v
        return attentions


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim=768):
        super().__init__()
        self.n_heads = n_heads
        self.d_h = embed_dim/self.n_heads

        self.linear1 = nn.Linear(embed_dim, 3 * self.d_h)
        self.attentions = ScaledDotProductAttentions()
        self.linear2 = nn.Linear(self.n_heads*embed_dim, self.d_h)

    def forward(self, x):
        b, n, d = x.shape
        x = self.linear(x).reshape(b, n, 3, -1).expand(-1, -1, -1, -1, self.n_heads)
        q, k, v = x[:, :, 0, :, :], x[:, :, 1, :, :], x[:, :, 2, :, :]
        self_attentions = self.attentions(q, k, v)
        multihead_self_attention = torch.cat(self_attentions, dim=3)
        multihead_self_attention = self.linear2(multihead_self_attention)


class FeedForward(nn.Module):
    def __init__(self, embed_dim=768, dim_inner_layer=2048):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, dim_inner_layer)
        self.linear2 = nn.Linear(dim_inner_layer, embed_dim)

    def forward(self, x):
        x = self.linear2(F.relu(self.linear1(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm()
        self.msa = MultiHeadAttention(n_heads)
        self.norm2 = nn.LayerNorm()
        self.mlp = FeedForward()
        self.norm3 = nn.LayerNorm()

    def forward(self, x):
        x = self.msa(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        x = self.norm3(x)
        return x