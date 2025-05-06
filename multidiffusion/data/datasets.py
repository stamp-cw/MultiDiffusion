"""
Dataset loaders for common image datasets
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
import paddle
import paddle.vision as vision
import paddle.vision.transforms as T
from torch.utils.data import DataLoader
from paddle.io import DataLoader as PaddleDataLoader
from abc import ABC, abstractmethod

class BaseDataset(ABC):
    """Abstract base class for dataset implementations"""
    @abstractmethod
    def get_dataloader(self, batch_size, num_workers, pin_memory=True):
        pass
        
    @abstractmethod
    def get_data_shape(self):
        pass

class TorchDataset(BaseDataset):
    """PyTorch dataset implementation"""
    def __init__(self, name, root="./data", train=True, download=True):
        self.name = name
        self.root = root
        self.train = train
        
        # Create transforms
        if name in ["mnist", "cifar10", "cifar100"]:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) if name == "mnist" else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
        # Load dataset
        if name == "mnist":
            self.dataset = torchvision.datasets.MNIST(root, train=train, transform=self.transform, download=download)
        elif name == "cifar10":
            self.dataset = torchvision.datasets.CIFAR10(root, train=train, transform=self.transform, download=download)
        elif name == "cifar100":
            self.dataset = torchvision.datasets.CIFAR100(root, train=train, transform=self.transform, download=download)
        elif name == "celeba":
            if not os.path.exists(os.path.join(root, "celeba")):
                raise ValueError("CelebA dataset not found. Please download it manually.")
            self.dataset = torchvision.datasets.CelebA(root, split="train" if train else "test", transform=self.transform, download=False)
        elif name == "lsun":
            if not os.path.exists(os.path.join(root, "lsun")):
                raise ValueError("LSUN dataset not found. Please download it manually.")
            self.dataset = torchvision.datasets.LSUN(root, classes=["bedroom_train" if train else "bedroom_val"], transform=self.transform)
        else:
            raise ValueError(f"Unknown dataset: {name}")
            
    def get_dataloader(self, batch_size, num_workers, pin_memory=True):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True if self.train else False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
    def get_data_shape(self):
        if self.name == "mnist":
            return (1, 28, 28)
        elif self.name in ["cifar10", "cifar100"]:
            return (3, 32, 32)
        else:
            return (3, 64, 64)

class PaddleDataset(BaseDataset):
    """PaddlePaddle dataset implementation"""
    def __init__(self, name, root="./data", train=True):
        self.name = name
        self.root = root
        self.train = train
        
        # Create transforms
        if name in ["mnist", "cifar10", "cifar100"]:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5] if name == "mnist" else [0.5, 0.5, 0.5],
                          std=[0.5] if name == "mnist" else [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = T.Compose([
                T.Resize(64),
                T.CenterCrop(64),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
        # Load dataset
        if name == "mnist":
            self.dataset = vision.datasets.MNIST(root, mode="train" if train else "test", transform=self.transform, download=True)
        elif name == "cifar10":
            self.dataset = vision.datasets.Cifar10(root, mode="train" if train else "test", transform=self.transform, download=True)
        elif name == "cifar100":
            self.dataset = vision.datasets.Cifar100(root, mode="train" if train else "test", transform=self.transform, download=True)
        else:
            raise ValueError(f"Dataset {name} not supported in PaddlePaddle yet")
            
    def get_dataloader(self, batch_size, num_workers, pin_memory=True):
        return PaddleDataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True if self.train else False,
            num_workers=num_workers
        )
        
    def get_data_shape(self):
        if self.name == "mnist":
            return (1, 28, 28)
        elif self.name in ["cifar10", "cifar100"]:
            return (3, 32, 32)
        else:
            return (3, 64, 64)

def get_dataset(name, framework="pytorch", **kwargs):
    """Factory function to get dataset loader"""
    if framework == "pytorch":
        return TorchDataset(name, **kwargs)
    elif framework == "paddle":
        return PaddleDataset(name, **kwargs)
    else:
        raise ValueError(f"Unknown framework: {framework}") 