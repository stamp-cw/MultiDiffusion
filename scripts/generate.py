"""
Generation script for trained diffusion models
"""

import os
import argparse
import yaml
from types import SimpleNamespace
import torch
import paddle
from multidiffusion.trainers import get_trainer
from multidiffusion.utils.visualization import save_grid, create_gif
from multidiffusion.utils.logging import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from diffusion model")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--framework", type=str, choices=["pytorch", "paddle"],
                      default="pytorch", help="Deep learning framework to use")
    parser.add_argument("--num_samples", type=int, default=64,
                      help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for generation")
    parser.add_argument("--output_dir", type=str, default="samples",
                      help="Output directory for generated samples")
    parser.add_argument("--save_grid", action="store_true",
                      help="Save samples as a grid")
    parser.add_argument("--create_gif", action="store_true",
                      help="Create GIF of the generation process")
    parser.add_argument("--seed", type=int, default=None,
                      help="Random seed for reproducibility")
    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)

def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        paddle.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Override framework if specified
    if args.framework:
        config.training.framework = args.framework
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Using framework: {config.training.framework}")
    
    # Create trainer and load checkpoint
    trainer = get_trainer(config)
    trainer.load_checkpoint(args.checkpoint)
    trainer.model.eval()
    
    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    
    all_samples = []
    for i in range(0, args.num_samples, args.batch_size):
        batch_size = min(args.batch_size, args.num_samples - i)
        
        if config.training.framework == "pytorch":
            with torch.no_grad():
                samples = trainer.model.sample(batch_size, trainer.device)
        else:
            with paddle.no_grad():
                samples = trainer.model.sample(batch_size)
                
        all_samples.append(samples)
        
        # Save individual samples
        for j, sample in enumerate(samples):
            idx = i + j
            if config.training.framework == "pytorch":
                sample = sample.cpu()
            save_grid(
                sample.unsqueeze(0),
                os.path.join(args.output_dir, f"sample_{idx:04d}.png")
            )
            
    # Concatenate all samples
    if config.training.framework == "pytorch":
        all_samples = torch.cat(all_samples)
    else:
        all_samples = paddle.concat(all_samples)
        
    # Save grid if requested
    if args.save_grid:
        save_grid(
            all_samples,
            os.path.join(args.output_dir, "grid.png"),
            title=f"Generated Samples"
        )
        
    # Create GIF if requested
    if args.create_gif:
        create_gif(
            args.output_dir,
            os.path.join(args.output_dir, "generation.gif")
        )
        
    logger.info("Generation complete!")

if __name__ == "__main__":
    main() 