"""
Training script for diffusion models
"""

import os
import argparse
import yaml
from types import SimpleNamespace
from multidiffusion.trainers import get_trainer
from multidiffusion.utils.logging import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                      help="Path to checkpoint to resume from")
    parser.add_argument("--framework", type=str, choices=["pytorch", "paddle"],
                      default="pytorch", help="Deep learning framework to use")
    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)

def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override framework if specified
    if args.framework:
        config.training.framework = args.framework
    
    # Create output directories
    os.makedirs(config.logging.log_dir, exist_ok=True)
    os.makedirs(config.logging.checkpoint_dir, exist_ok=True)
    os.makedirs(config.logging.sample_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Using framework: {config.training.framework}")
    
    # Create trainer
    trainer = get_trainer(config)
    
    # Load checkpoint if specified
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 