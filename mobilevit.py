import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ConvBlock(nn.Module):
    """Replacement for Conv2dNormActivation compatible with torchvision 0.12.0"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class MobileViTBlock(nn.Module):
    """MobileViT block with patch-based self-attention"""
    def __init__(self, in_channels, out_channels, kernel_size=3, patch_size=8):
        super().__init__()
        self.patch_size = patch_size

        # Local representation
        self.conv1 = ConvBlock(in_channels, in_channels, kernel_size)

        # Project to out_channels for patch embedding
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # Compute embedding dimension = patch_size*patch_size*out_channels
        self.embedding_dim = patch_size * patch_size * out_channels

        # Normalize over that embedding dimension
        self.norm = nn.LayerNorm(self.embedding_dim)

        # MLP-style “attention” over each patch embedding
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.SiLU(),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # 1) pad so h,w divisible by patch_size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            # pad format: (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            h, w = x.shape[-2:]

        # 2) local conv
        local_feat = self.conv1(x)

        # 3) project to out_channels
        global_feat = self.conv2(local_feat)

        # 4) unfold into (num_patches, embedding_dim)
        patches = rearrange(
            global_feat,
            'b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)',
            p1=self.patch_size, p2=self.patch_size
        )

        # 5) norm + lightweight attention
        patches = self.norm(patches)                      # → [b, num_patches, embedding_dim]
        attn_weights = self.attention(patches)            # → same shape
        attended    = patches * attn_weights              # → same shape

        # 6) fold back to spatial
        nh = h // self.patch_size
        nw = w // self.patch_size
        out = rearrange(
            attended,
            'b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)',
            nh=nh, nw=nw, p1=self.patch_size, p2=self.patch_size, c=global_feat.shape[1]
        )

        return out

class MobileViT(nn.Module):
    """Efficient MobileViT backbone with depth estimation head"""
    def __init__(self):
        super().__init__()
        # stem
        self.stem = nn.Sequential(
            ConvBlock(3, 16, 3, stride=2),
            ConvBlock(16, 32, 3, stride=1)
        )
        # MobileViT blocks
        self.block1 = MobileViTBlock(32,  64, patch_size=8)
        self.block2 = MobileViTBlock(64, 128, patch_size=8)
        self.block3 = MobileViTBlock(128,256, patch_size=8)
        # depth head
        self.depth_head = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBlock(128, 64, 3),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
        # freeze early layers
        for p in self.stem.parameters():  p.requires_grad = False
        for p in self.block1.parameters(): p.requires_grad = False

    def forward(self, x):
        # only cast to half when on CUDA
        if x.is_cuda and x.dtype != torch.float16:
            x = x.half()
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        depth = self.depth_head(x)
        return depth.float()

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)