"""
Basic tests for MultiDiffusion
"""

import os
import torch
import paddle
import pytest
from types import SimpleNamespace
import numpy as np
from multidiffusion.models import DiffusionModel, PaddleDiffusionModel
from multidiffusion.data import get_dataset
from multidiffusion.trainers import get_trainer

@pytest.fixture
def config():
    """Create test config"""
    return SimpleNamespace(
        model=SimpleNamespace(
            name="ddpm",
            backbone="unet",
            image_size=32,
            channels=3,
            time_embedding_dim=128,
            base_channels=64,
            channel_multipliers=[1, 2, 2],
            attention_resolutions=[16],
            num_res_blocks=2,
            dropout=0.1
        ),
        diffusion=SimpleNamespace(
            timesteps=100,
            beta_schedule="linear",
            beta_start=1e-4,
            beta_end=0.02
        ),
        training=SimpleNamespace(
            framework="pytorch",
            batch_size=4,
            num_epochs=1,
            learning_rate=1e-4,
            weight_decay=0.0,
            ema_decay=0.9999,
            gradient_clip=1.0,
            warmup_steps=100
        ),
        data=SimpleNamespace(
            name="cifar10",
            root="./data",
            num_workers=0,
            pin_memory=True
        ),
        evaluation=SimpleNamespace(
            eval_every=100,
            save_every=1000,
            sample_size=4,
            fid_samples=100,
            lpips_samples=100
        ),
        logging=SimpleNamespace(
            project="test",
            log_dir="test_runs",
            checkpoint_dir="test_checkpoints",
            sample_dir="test_samples",
            log_every=10
        ),
        distributed=SimpleNamespace(
            enabled=False,
            backend="nccl",
            world_size=1,
            rank=0
        )
    )

def test_pytorch_model(config):
    """Test PyTorch model"""
    model = DiffusionModel(config)
    assert isinstance(model, DiffusionModel)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, config.model.channels,
                   config.model.image_size, config.model.image_size)
    t = torch.randint(0, config.diffusion.timesteps, (batch_size,))
    
    output = model(x, t)
    assert output.shape == x.shape
    
    # Test sampling
    samples = model.sample(batch_size, device="cpu")
    assert samples.shape == (batch_size, config.model.channels,
                           config.model.image_size, config.model.image_size)
    
    # Test loss computation
    loss = model.get_loss(x)
    assert isinstance(loss.item(), float)

def test_paddle_model(config):
    """Test PaddlePaddle model"""
    config.training.framework = "paddle"
    model = PaddleDiffusionModel(config)
    assert isinstance(model, PaddleDiffusionModel)
    
    # Test forward pass
    batch_size = 2
    x = paddle.randn([batch_size, config.model.channels,
                     config.model.image_size, config.model.image_size])
    t = paddle.randint(0, config.diffusion.timesteps, [batch_size])
    
    output = model(x, t)
    assert output.shape == x.shape
    
    # Test sampling
    samples = model.sample(batch_size)
    assert samples.shape == [batch_size, config.model.channels,
                           config.model.image_size, config.model.image_size]
    
    # Test loss computation
    loss = model.get_loss(x)
    assert isinstance(loss.item(), float)

def test_dataset_loading(config):
    """Test dataset loading"""
    # Test PyTorch dataset
    torch_dataset = get_dataset(config.data.name, framework="pytorch",
                              root=config.data.root)
    loader = torch_dataset.get_dataloader(config.training.batch_size,
                                        config.data.num_workers,
                                        config.data.pin_memory)
    batch = next(iter(loader))
    assert len(batch) == 2  # (images, labels)
    assert batch[0].shape[0] == config.training.batch_size
    
    # Test PaddlePaddle dataset
    paddle_dataset = get_dataset(config.data.name, framework="paddle",
                               root=config.data.root)
    loader = paddle_dataset.get_dataloader(config.training.batch_size,
                                         config.data.num_workers)
    batch = next(iter(loader))
    assert len(batch) == 2  # (images, labels)
    assert batch[0].shape[0] == config.training.batch_size

def test_trainer(config):
    """Test trainer functionality"""
    # Create output directories
    os.makedirs(config.logging.log_dir, exist_ok=True)
    os.makedirs(config.logging.checkpoint_dir, exist_ok=True)
    os.makedirs(config.logging.sample_dir, exist_ok=True)
    
    # Test PyTorch trainer
    config.training.framework = "pytorch"
    torch_trainer = get_trainer(config)
    assert hasattr(torch_trainer, "train")
    assert hasattr(torch_trainer, "evaluate")
    assert hasattr(torch_trainer, "save_checkpoint")
    assert hasattr(torch_trainer, "load_checkpoint")
    
    # Test PaddlePaddle trainer
    config.training.framework = "paddle"
    paddle_trainer = get_trainer(config)
    assert hasattr(paddle_trainer, "train")
    assert hasattr(paddle_trainer, "evaluate")
    assert hasattr(paddle_trainer, "save_checkpoint")
    assert hasattr(paddle_trainer, "load_checkpoint")
    
    # Clean up
    for dir_path in [config.logging.log_dir, config.logging.checkpoint_dir,
                    config.logging.sample_dir]:
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, file))
            os.rmdir(dir_path) 