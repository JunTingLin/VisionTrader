import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # Flash Attention via PyTorch SDPA (fused kernel, no O(n²) attn matrix in VRAM).
        # dropout_p=0.0 to match the original code, which applied no dropout on the
        # attention weights (softmax was used directly). Output-projection dropout in
        # to_out is preserved unchanged.
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                                       dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



class TE(nn.Module):
    def __init__(self, *, image_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        num_pixels = image_size[0] * image_size[1]
        patch_dim = channels
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(patch_dim, dim),
        )

        self.h = image_size[0]
        self.w = image_size[1]
        self.pos_embedding = nn.Parameter(torch.randn(1, num_pixels, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.h, w=self.w)
        return x


class TE_1D(nn.Module):
    def __init__(self, *, window_len=13, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        # ori input : (b, l, c)

        num_pixels = window_len
        patch_dim = channels
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(num_pixels, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x
