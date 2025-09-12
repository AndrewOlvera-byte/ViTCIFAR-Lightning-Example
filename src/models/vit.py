import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt
from src.registry import register_model

class PatchEmbed(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_chans=3, embed_dim = 384):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.grid = image_size // patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # B, C, H', W'
        x = x.flatten(2).transpose(1, 2) # B, N, C  (N = H'*W')
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self .fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
    
    def forward(self, x):
        #Pre-LN ViT
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

@register_model("vit")
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        embed_dim=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=10,
        dropout=0.0,
        attn_dropout=0.0,
        grad_ckpt: bool = False,
    ):
        super().__init__()
        self.grad_ckpt = bool(grad_ckpt)
        self.patch_embed = PatchEmbed(image_size, patch_size, 3, embed_dim)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_vit)

    @staticmethod
    def _init_vit(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.patch_embed(x)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        if self.grad_ckpt and self.training:
            for blk in self.blocks:
                try:
                    x = ckpt(blk, x, use_reentrant=False)
                except TypeError:
                    x = ckpt(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)