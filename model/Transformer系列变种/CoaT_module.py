#CoaT: Co-Scale Conv-Attentional Image Transformers
# https://arxiv.org/abs/2104.06399

import torch
import math
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class SepConv2d(nn.Module):
    def __init__(self, nin, nout):
        super(SepConv2d, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class ConvAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.in_depthwiseconv = SepConv2d(dim, dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attn_depthwiseconv = SepConv2d(dim, dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, d, h = *x.shape, self.heads

        cls = x[:, :1]
        image_token = x[:, 1:]
        H = W = int(math.sqrt(n - 1))

        image_token = rearrange(image_token, 'b (l w) d -> b d l w', l=H, w=W)
        image_token = self.in_depthwiseconv(image_token)
        image_token = rearrange(image_token, ' b d h w -> b (h w) d')
        x = x + torch.cat((cls, image_token), dim=1)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        k = k.transpose(2, 3)
        k = k.softmax(dim=-1)
        context = einsum('b h i j, b h j a -> b h i a', k, v)
        attn = einsum('b h i j, b h j j -> b h i j', q, context)

        cls = v[:, :, :1]
        value_token = v[:, :, 1:]
        value_token = rearrange(value_token, 'b h (l w) d -> b (h d) l w', l=H, w=W)
        value_token = self.attn_depthwiseconv(value_token)
        value_token = rearrange(value_token, ' b (h d) l w -> b h (l w) d', h=h)
        v = torch.cat((cls, value_token), dim=2)

        out = einsum('b h i j, b h i j -> b h i j', q, v)
        out = out + attn

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
