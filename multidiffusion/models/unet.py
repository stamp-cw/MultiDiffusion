"""
UNet model implementation for both PyTorch and PaddlePaddle
"""

import math
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import paddle
import paddle.nn as pnn
import paddle.nn.functional as pF

class TimeEmbedding(nn.Module):
    """Time embedding layer for diffusion models"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.emb = math.log(10000) / (half_dim - 1)
        self.register_buffer('freq', torch.exp(-self.emb * torch.arange(half_dim)))

    def forward(self, t):
        emb = t[:, None] * self.freq[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class PaddleTimeEmbedding(pnn.Layer):
    """Time embedding layer for PaddlePaddle"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.emb = math.log(10000) / (half_dim - 1)
        self.register_buffer('freq', paddle.exp(-self.emb * paddle.arange(half_dim)))

    def forward(self, t):
        emb = t[:, None] * self.freq[None, :]
        return paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=-1)

class ResBlock(nn.Module):
    """Residual block with time embedding"""
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_channels, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = F.relu(self.conv1(x))
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        h = F.relu(self.dropout(self.conv2(h)))
        return h + self.skip(x)

class PaddleResBlock(pnn.Layer):
    """Residual block with time embedding for PaddlePaddle"""
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = pnn.Conv2D(in_channels, out_channels, 3, padding=1)
        self.time_mlp = pnn.Linear(time_channels, out_channels)
        self.conv2 = pnn.Conv2D(out_channels, out_channels, 3, padding=1)
        self.dropout = pnn.Dropout(dropout)
        self.skip = pnn.Conv2D(in_channels, out_channels, 1) if in_channels != out_channels else pnn.Identity()

    def forward(self, x, time_emb):
        h = pF.relu(self.conv1(x))
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        h = pF.relu(self.dropout(self.conv2(h)))
        return h + self.skip(x)

class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, C, -1)
        k = k.view(B, C, -1)
        v = v.view(B, C, -1)

        attn = torch.einsum('bci,bcj->bij', q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=2)
        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.view(B, C, H, W)
        return self.proj(out) + x

class PaddleAttentionBlock(pnn.Layer):
    """Self-attention block for PaddlePaddle"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = pnn.GroupNorm(8, channels)
        self.qkv = pnn.Conv2D(channels, channels * 3, 1)
        self.proj = pnn.Conv2D(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = paddle.chunk(qkv, 3, axis=1)
        q = q.reshape([B, C, -1])
        k = k.reshape([B, C, -1])
        v = v.reshape([B, C, -1])

        attn = paddle.einsum('bci,bcj->bij', q, k) * (C ** -0.5)
        attn = pF.softmax(attn, axis=2)
        out = paddle.einsum('bij,bcj->bci', attn, v)
        out = out.reshape([B, C, H, W])
        return self.proj(out) + x

class BaseUNet(ABC):
    """Abstract base class for UNet implementation"""
    @abstractmethod
    def forward(self, x, t):
        pass

class UNet(nn.Module, BaseUNet):
    """PyTorch implementation of UNet for diffusion models"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Time embedding
        time_dim = config.time_embedding_dim
        self.time_embed = TimeEmbedding(time_dim)
        self.time_embed_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.ReLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # Initial projection
        channels = [config.base_channels * m for m in [1] + config.channel_multipliers]
        self.init_conv = nn.Conv2d(config.channels, channels[0], 3, padding=1)
        
        # Downsampling
        self.downs = nn.ModuleList([])
        for i in range(len(channels) - 1):
            self.downs.append(nn.ModuleList([
                ResBlock(channels[i], channels[i], time_dim, config.dropout),
                ResBlock(channels[i], channels[i], time_dim, config.dropout),
                AttentionBlock(channels[i]) if config.image_size // (2 ** i) in config.attention_resolutions else nn.Identity(),
                nn.Conv2d(channels[i], channels[i + 1], 4, 2, 1)
            ]))
        
        # Middle
        mid_channels = channels[-1]
        self.mid = nn.ModuleList([
            ResBlock(mid_channels, mid_channels, time_dim, config.dropout),
            AttentionBlock(mid_channels),
            ResBlock(mid_channels, mid_channels, time_dim, config.dropout)
        ])
        
        # Upsampling
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(channels) - 1)):
            self.ups.append(nn.ModuleList([
                ResBlock(channels[i + 1] + channels[i], channels[i], time_dim, config.dropout),
                ResBlock(channels[i] + channels[i], channels[i], time_dim, config.dropout),
                AttentionBlock(channels[i]) if config.image_size // (2 ** i) in config.attention_resolutions else nn.Identity(),
                nn.ConvTranspose2d(channels[i], channels[i - 1] if i > 0 else channels[i], 4, 2, 1)
            ]))
            
        # Output
        self.final = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], config.channels, 3, padding=1)
        )

    def forward(self, x, t):
        # Time embedding
        t = self.time_embed(t)
        t = self.time_embed_mlp(t)
        
        # Initial convolution
        h = self.init_conv(x)
        hs = [h]
        
        # Downsampling
        for down in self.downs:
            h = down[0](h, t)
            h = down[1](h, t)
            h = down[2](h)
            h = down[3](h)
            hs.append(h)
            
        # Middle
        h = self.mid[0](h, t)
        h = self.mid[1](h)
        h = self.mid[2](h, t)
        
        # Upsampling
        for up in self.ups:
            h = torch.cat([h, hs.pop()], dim=1)
            h = up[0](h, t)
            h = torch.cat([h, hs.pop()], dim=1)
            h = up[1](h, t)
            h = up[2](h)
            h = up[3](h)
            
        # Output
        return self.final(h)

class PaddleUNet(pnn.Layer, BaseUNet):
    """PaddlePaddle implementation of UNet for diffusion models"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Time embedding
        time_dim = config.time_embedding_dim
        self.time_embed = PaddleTimeEmbedding(time_dim)
        self.time_embed_mlp = pnn.Sequential(
            pnn.Linear(time_dim, time_dim * 4),
            pnn.ReLU(),
            pnn.Linear(time_dim * 4, time_dim)
        )

        # Initial projection
        channels = [config.base_channels * m for m in [1] + config.channel_multipliers]
        self.init_conv = pnn.Conv2D(config.channels, channels[0], 3, padding=1)
        
        # Downsampling
        self.downs = pnn.LayerList([])
        for i in range(len(channels) - 1):
            self.downs.append(pnn.LayerList([
                PaddleResBlock(channels[i], channels[i], time_dim, config.dropout),
                PaddleResBlock(channels[i], channels[i], time_dim, config.dropout),
                PaddleAttentionBlock(channels[i]) if config.image_size // (2 ** i) in config.attention_resolutions else pnn.Identity(),
                pnn.Conv2D(channels[i], channels[i + 1], 4, 2, 1)
            ]))
        
        # Middle
        mid_channels = channels[-1]
        self.mid = pnn.LayerList([
            PaddleResBlock(mid_channels, mid_channels, time_dim, config.dropout),
            PaddleAttentionBlock(mid_channels),
            PaddleResBlock(mid_channels, mid_channels, time_dim, config.dropout)
        ])
        
        # Upsampling
        self.ups = pnn.LayerList([])
        for i in reversed(range(len(channels) - 1)):
            self.ups.append(pnn.LayerList([
                PaddleResBlock(channels[i + 1] + channels[i], channels[i], time_dim, config.dropout),
                PaddleResBlock(channels[i] + channels[i], channels[i], time_dim, config.dropout),
                PaddleAttentionBlock(channels[i]) if config.image_size // (2 ** i) in config.attention_resolutions else pnn.Identity(),
                pnn.Conv2DTranspose(channels[i], channels[i - 1] if i > 0 else channels[i], 4, 2, 1)
            ]))
            
        # Output
        self.final = pnn.Sequential(
            pnn.GroupNorm(8, channels[0]),
            pnn.ReLU(),
            pnn.Conv2D(channels[0], config.channels, 3, padding=1)
        )

    def forward(self, x, t):
        # Time embedding
        t = self.time_embed(t)
        t = self.time_embed_mlp(t)
        
        # Initial convolution
        h = self.init_conv(x)
        hs = [h]
        
        # Downsampling
        for down in self.downs:
            h = down[0](h, t)
            h = down[1](h, t)
            h = down[2](h)
            h = down[3](h)
            hs.append(h)
            
        # Middle
        h = self.mid[0](h, t)
        h = self.mid[1](h)
        h = self.mid[2](h, t)
        
        # Upsampling
        for up in self.ups:
            h = paddle.concat([h, hs.pop()], axis=1)
            h = up[0](h, t)
            h = paddle.concat([h, hs.pop()], axis=1)
            h = up[1](h, t)
            h = up[2](h)
            h = up[3](h)
            
        # Output
        return self.final(h) 