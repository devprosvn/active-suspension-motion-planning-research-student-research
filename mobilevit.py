import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.conv1 = ConvBlock(in_channels, in_channels, kernel_size)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # embedding dim = patch_area * out_channels
        self.embedding_dim = patch_size * patch_size * out_channels
        self.norm = nn.LayerNorm(self.embedding_dim)
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.SiLU(),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape

        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            h, w = x.shape[-2:]

        local_feat = self.conv1(x)
        global_feat = self.conv2(local_feat)

        patches = rearrange(global_feat, 'b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)',
                            p1=self.patch_size, p2=self.patch_size)

        patches = self.norm(patches)
        attended = patches * self.attention(patches)

        nh, nw = h // self.patch_size, w // self.patch_size
        out = rearrange(attended, 'b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)',
                        nh=nh, nw=nw, p1=self.patch_size, p2=self.patch_size, c=global_feat.shape[1])
        return out

class MobileViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(3, 16, 3, stride=2),
            ConvBlock(16, 32, 3, stride=1)
        )
        self.block1 = MobileViTBlock(32, 64, patch_size=8)
        self.block2 = MobileViTBlock(64, 128, patch_size=8)
        self.block3 = MobileViTBlock(128, 256, patch_size=8)

        self.depth_head = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBlock(128, 64, 3),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # freeze stem and block1 if needed
        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.block1.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.is_cuda and x.dtype != torch.float16:
            x = x.half()

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        depth = self.depth_head(x)

        # Final resize to match sensor output (e.g., 360x640)
        depth = F.interpolate(depth, size=(360, 640), mode='bilinear', align_corners=False)
        return depth.float()

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)