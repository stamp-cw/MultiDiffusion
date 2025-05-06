"""
Visualization utilities for both PyTorch and PaddlePaddle
"""

import os
import math
import numpy as np
import torch
import paddle
import matplotlib.pyplot as plt
from torchvision.utils import make_grid as torch_make_grid
from paddle.vision.transforms import to_tensor

def denormalize(tensor):
    """Convert normalized image tensor to uint8 numpy array"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().float() * 0.5 + 0.5
        tensor = tensor.clamp(0, 1) * 255
        return tensor.byte().numpy()
    elif isinstance(tensor, paddle.Tensor):
        tensor = tensor.numpy() * 0.5 + 0.5
        tensor = np.clip(tensor, 0, 1) * 255
        return tensor.astype(np.uint8)
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")

def make_grid(tensors, nrow=8):
    """Create image grid from batch of tensors"""
    if isinstance(tensors, torch.Tensor):
        return torch_make_grid(tensors, nrow=nrow, normalize=True, range=(-1, 1))
    elif isinstance(tensors, paddle.Tensor):
        # Convert to numpy for consistent handling
        tensors = tensors.numpy()
        b, c, h, w = tensors.shape
        
        # Compute grid size
        nrow = min(nrow, b)
        ncol = math.ceil(b / nrow)
        
        # Create empty grid
        grid = np.zeros((c, h * nrow, w * ncol))
        
        # Fill grid with images
        for idx in range(b):
            i = idx % nrow
            j = idx // nrow
            grid[:, i * h:(i + 1) * h, j * w:(j + 1) * w] = tensors[idx]
            
        return to_tensor(grid)
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensors)}")

def save_grid(tensors, path, nrow=8, title=None):
    """Save batch of images as grid"""
    # Create output directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(10, 10))
    if title:
        plt.title(title)
    
    # Convert to grid and save
    grid = make_grid(tensors, nrow=nrow)
    if isinstance(grid, torch.Tensor):
        grid = grid.permute(1, 2, 0).cpu().numpy()
    else:
        grid = grid.transpose(1, 2, 0).numpy()
    
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def create_gif(image_dir, output_path, fps=10):
    """Create GIF from directory of images"""
    try:
        import imageio
    except ImportError:
        raise ImportError("Please install imageio with 'pip install imageio'")
        
    # Get all PNG files
    images = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".png"):
            file_path = os.path.join(image_dir, filename)
            images.append(imageio.imread(file_path))
            
    # Save as GIF
    imageio.mimsave(output_path, images, fps=fps)

def plot_loss(losses, path, title="Training Loss"):
    """Plot loss curve"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(path)
    plt.close() 