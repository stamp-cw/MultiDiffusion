"""
Diffusion model implementation with various noise schedules
"""

import math
import numpy as np
import torch
import torch.nn as nn
import paddle
import paddle.nn as pnn
from abc import ABC, abstractmethod
from .unet import UNet, PaddleUNet

class NoiseSchedule(ABC):
    """Abstract base class for noise schedules"""
    @abstractmethod
    def get_alpha_bars(self, timesteps):
        pass

class LinearSchedule(NoiseSchedule):
    """Linear noise schedule"""
    def __init__(self, beta_start=1e-4, beta_end=0.02):
        self.beta_start = beta_start
        self.beta_end = beta_end
        
    def get_alpha_bars(self, timesteps):
        betas = torch.linspace(self.beta_start, self.beta_end, timesteps)
        alphas = 1. - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return alpha_bars

class CosineSchedule(NoiseSchedule):
    """Cosine noise schedule from improved DDPM paper"""
    def __init__(self, s=0.008):
        self.s = s
        
    def get_alpha_bars(self, timesteps):
        steps = torch.linspace(0, timesteps, timesteps + 1)
        f_t = torch.cos((steps / timesteps + self.s) / (1. + self.s) * math.pi * 0.5) ** 2
        f_t = f_t / f_t[0]
        alpha_bars = f_t[1:]
        return alpha_bars

class CustomSchedule(NoiseSchedule):
    """Custom noise schedule loaded from file"""
    def __init__(self, alpha_bars):
        self.saved_alpha_bars = torch.tensor(alpha_bars)
        
    def get_alpha_bars(self, timesteps):
        if len(self.saved_alpha_bars) != timesteps:
            raise ValueError(f"Saved schedule has {len(self.saved_alpha_bars)} steps but {timesteps} were requested")
        return self.saved_alpha_bars

class DiffusionModel(nn.Module):
    """PyTorch implementation of diffusion model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = UNet(config)
        
        # Set up noise schedule
        if config.diffusion.beta_schedule == "linear":
            self.noise_schedule = LinearSchedule(
                config.diffusion.beta_start,
                config.diffusion.beta_end
            )
        elif config.diffusion.beta_schedule == "cosine":
            self.noise_schedule = CosineSchedule()
        else:
            raise ValueError(f"Unknown beta schedule: {config.diffusion.beta_schedule}")
            
        self.timesteps = config.diffusion.timesteps
        self.register_buffer('alpha_bars', self.noise_schedule.get_alpha_bars(self.timesteps))
        
    def forward(self, x, t):
        """Forward pass predicting noise"""
        return self.model(x, t)
        
    def p_sample(self, x_t, t):
        """Single step of reverse diffusion sampling"""
        t_tensor = torch.tensor([t], device=x_t.device)
        eps_theta = self.model(x_t, t_tensor)
        
        alpha_bar = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t-1] if t > 0 else torch.tensor(1.)
        
        # Compute parameters of posterior q(x_{t-1} | x_t, x_0)
        beta = 1 - alpha_bar / alpha_bar_prev
        variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
        
        # Get mean of posterior
        x_prev_mean = torch.sqrt(alpha_bar_prev) * (
            x_t / torch.sqrt(alpha_bar) - 
            eps_theta * torch.sqrt(1 - alpha_bar) / torch.sqrt(alpha_bar)
        )
        
        if t == 0:
            return x_prev_mean
        else:
            noise = torch.randn_like(x_t)
            return x_prev_mean + torch.sqrt(variance) * noise
            
    @torch.no_grad()
    def sample(self, batch_size, device):
        """Generate samples"""
        x = torch.randn(batch_size, self.config.channels,
                       self.config.image_size, self.config.image_size,
                       device=device)
                       
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t)
        return x
        
    def get_loss(self, x_0, noise=None):
        """Compute training loss"""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_0.device)
        
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Get noisy samples
        x_t = torch.sqrt(self.alpha_bars[t])[:, None, None, None] * x_0 + \
              torch.sqrt(1 - self.alpha_bars[t])[:, None, None, None] * noise
              
        # Predict noise and compute loss
        noise_pred = self.model(x_t, t)
        return nn.MSELoss()(noise_pred, noise)

class PaddleDiffusionModel(pnn.Layer):
    """PaddlePaddle implementation of diffusion model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = PaddleUNet(config)
        
        # Set up noise schedule
        if config.diffusion.beta_schedule == "linear":
            betas = paddle.linspace(config.diffusion.beta_start, 
                                  config.diffusion.beta_end,
                                  config.diffusion.timesteps)
        elif config.diffusion.beta_schedule == "cosine":
            steps = paddle.linspace(0, config.diffusion.timesteps, 
                                  config.diffusion.timesteps + 1)
            f_t = paddle.cos((steps / config.diffusion.timesteps + 0.008) / 1.008 * math.pi * 0.5) ** 2
            f_t = f_t / f_t[0]
            betas = 1 - f_t[1:] / f_t[:-1]
        else:
            raise ValueError(f"Unknown beta schedule: {config.diffusion.beta_schedule}")
            
        self.timesteps = config.diffusion.timesteps
        alphas = 1. - betas
        alpha_bars = paddle.cumprod(alphas, 0)
        self.register_buffer('alpha_bars', alpha_bars)
        
    def forward(self, x, t):
        """Forward pass predicting noise"""
        return self.model(x, t)
        
    def p_sample(self, x_t, t):
        """Single step of reverse diffusion sampling"""
        t_tensor = paddle.to_tensor([t], dtype='int64')
        eps_theta = self.model(x_t, t_tensor)
        
        alpha_bar = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t-1] if t > 0 else paddle.to_tensor(1.)
        
        # Compute parameters of posterior q(x_{t-1} | x_t, x_0)
        beta = 1 - alpha_bar / alpha_bar_prev
        variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
        
        # Get mean of posterior
        x_prev_mean = paddle.sqrt(alpha_bar_prev) * (
            x_t / paddle.sqrt(alpha_bar) - 
            eps_theta * paddle.sqrt(1 - alpha_bar) / paddle.sqrt(alpha_bar)
        )
        
        if t == 0:
            return x_prev_mean
        else:
            noise = paddle.randn(x_t.shape)
            return x_prev_mean + paddle.sqrt(variance) * noise
            
    @paddle.no_grad()
    def sample(self, batch_size):
        """Generate samples"""
        x = paddle.randn([batch_size, self.config.channels,
                         self.config.image_size, self.config.image_size])
                       
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t)
        return x
        
    def get_loss(self, x_0, noise=None):
        """Compute training loss"""
        batch_size = x_0.shape[0]
        t = paddle.randint(0, self.timesteps, [batch_size])
        
        if noise is None:
            noise = paddle.randn(x_0.shape)
            
        # Get noisy samples
        x_t = paddle.sqrt(self.alpha_bars[t])[:, None, None, None] * x_0 + \
              paddle.sqrt(1 - self.alpha_bars[t])[:, None, None, None] * noise
              
        # Predict noise and compute loss
        noise_pred = self.model(x_t, t)
        return pnn.MSELoss()(noise_pred, noise) 