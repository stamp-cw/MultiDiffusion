"""
Trainer implementations for both PyTorch and PaddlePaddle
"""

import os
import time
import torch
import paddle
from abc import ABC, abstractmethod
from torch.optim import Adam
from paddle.optimizer import Adam as PaddleAdam
from torch.optim.lr_scheduler import LambdaLR
from paddle.optimizer.lr import LRScheduler
from torch.cuda.amp import autocast, GradScaler
from paddle.amp import auto_cast, GradScaler as PaddleGradScaler
from ..models import DiffusionModel, PaddleDiffusionModel
from ..data import get_dataset
from ..utils.logging import get_logger
from ..utils.visualization import save_grid

logger = get_logger(__name__)

class BaseTrainer(ABC):
    """Abstract base class for trainers"""
    @abstractmethod
    def train(self):
        pass
        
    @abstractmethod
    def evaluate(self):
        pass
        
    @abstractmethod
    def save_checkpoint(self):
        pass
        
    @abstractmethod
    def load_checkpoint(self):
        pass

class TorchTrainer(BaseTrainer):
    """PyTorch trainer implementation"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup data
        self.dataset = get_dataset(
            config.data.name,
            framework="pytorch",
            root=os.path.join(config.data.root, config.data.name)
        )
        self.train_loader = self.dataset.get_dataloader(
            config.training.batch_size,
            config.data.num_workers,
            config.data.pin_memory
        )
        
        # Create model
        self.model = DiffusionModel(config).to(self.device)
        
        # Setup training
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Learning rate schedule
        def warmup_schedule(step):
            if step < config.training.warmup_steps:
                return step / config.training.warmup_steps
            return 1.0
            
        self.scheduler = LambdaLR(self.optimizer, warmup_schedule)
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Initialize tracking
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
    def train(self):
        """Training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.epoch, self.config.training.num_epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch in self.train_loader:
                # Get batch
                x = batch[0].to(self.device)
                
                # Forward pass with mixed precision
                with autocast():
                    loss = self.model.get_loss(x)
                    
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                
                # Logging
                epoch_loss += loss.item()
                
                if self.step % self.config.logging.log_every == 0:
                    logger.info(f"Step {self.step} | Loss: {loss.item():.4f}")
                    
                # Evaluation
                if self.step % self.config.evaluation.eval_every == 0:
                    self.evaluate()
                    
                # Save checkpoint
                if self.step % self.config.evaluation.save_every == 0:
                    self.save_checkpoint()
                    
                self.step += 1
                
            # Epoch end logging
            epoch_loss /= len(self.train_loader)
            logger.info(f"Epoch {epoch} | Average Loss: {epoch_loss:.4f}")
            
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_checkpoint(is_best=True)
                
            self.epoch += 1
            
    @torch.no_grad()
    def evaluate(self):
        """Generate and save samples"""
        self.model.eval()
        
        # Generate samples
        samples = self.model.sample(
            self.config.evaluation.sample_size,
            self.device
        )
        
        # Save samples
        save_grid(
            samples,
            os.path.join(
                self.config.logging.sample_dir,
                f"samples_step_{self.step}.png"
            )
        )
        
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        path = os.path.join(
            self.config.logging.checkpoint_dir,
            f"checkpoint_step_{self.step}.pt"
        )
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(
                self.config.logging.checkpoint_dir,
                "best_model.pt"
            )
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

class PaddleTrainer(BaseTrainer):
    """PaddlePaddle trainer implementation"""
    def __init__(self, config):
        self.config = config
        
        # Setup data
        self.dataset = get_dataset(
            config.data.name,
            framework="paddle",
            root=os.path.join(config.data.root, config.data.name)
        )
        self.train_loader = self.dataset.get_dataloader(
            config.training.batch_size,
            config.data.num_workers
        )
        
        # Create model
        self.model = PaddleDiffusionModel(config)
        
        # Setup training
        self.optimizer = PaddleAdam(
            parameters=self.model.parameters(),
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Learning rate schedule
        class WarmupScheduler(LRScheduler):
            def __init__(self, warmup_steps, learning_rate):
                self.warmup_steps = warmup_steps
                super().__init__(learning_rate)
                
            def get_lr(self):
                if self.step_num < self.warmup_steps:
                    return self.base_lr * (self.step_num / self.warmup_steps)
                return self.base_lr
                
        self.scheduler = WarmupScheduler(
            config.training.warmup_steps,
            config.training.learning_rate
        )
        
        # Mixed precision training
        self.scaler = PaddleGradScaler()
        
        # Initialize tracking
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
    def train(self):
        """Training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.epoch, self.config.training.num_epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch in self.train_loader:
                # Get batch
                x = batch[0]
                
                # Forward pass with mixed precision
                with auto_cast():
                    loss = self.model.get_loss(x)
                    
                # Backward pass
                scaled_loss = self.scaler.scale(loss)
                scaled_loss.backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                paddle.nn.ClipGradByNorm(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                
                # Logging
                epoch_loss += loss.item()
                
                if self.step % self.config.logging.log_every == 0:
                    logger.info(f"Step {self.step} | Loss: {loss.item():.4f}")
                    
                # Evaluation
                if self.step % self.config.evaluation.eval_every == 0:
                    self.evaluate()
                    
                # Save checkpoint
                if self.step % self.config.evaluation.save_every == 0:
                    self.save_checkpoint()
                    
                self.step += 1
                
            # Epoch end logging
            epoch_loss /= len(self.train_loader)
            logger.info(f"Epoch {epoch} | Average Loss: {epoch_loss:.4f}")
            
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_checkpoint(is_best=True)
                
            self.epoch += 1
            
    @paddle.no_grad()
    def evaluate(self):
        """Generate and save samples"""
        self.model.eval()
        
        # Generate samples
        samples = self.model.sample(self.config.evaluation.sample_size)
        
        # Save samples
        save_grid(
            samples,
            os.path.join(
                self.config.logging.sample_dir,
                f"samples_step_{self.step}.png"
            )
        )
        
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        path = os.path.join(
            self.config.logging.checkpoint_dir,
            f"checkpoint_step_{self.step}.pdparams"
        )
        paddle.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(
                self.config.logging.checkpoint_dir,
                "best_model.pdparams"
            )
            paddle.save(checkpoint, best_path)
            
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = paddle.load(path)
        
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        self.model.set_state_dict(checkpoint['model_state_dict'])
        self.optimizer.set_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.set_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

def get_trainer(config):
    """Factory function to get trainer"""
    if config.training.framework == "pytorch":
        return TorchTrainer(config)
    elif config.training.framework == "paddle":
        return PaddleTrainer(config)
    else:
        raise ValueError(f"Unknown framework: {config.training.framework}") 