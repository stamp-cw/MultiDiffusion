"""
Gradio demo for diffusion model
"""

import os
import argparse
import yaml
from types import SimpleNamespace
import torch
import gradio as gr
from multidiffusion.trainers import get_trainer
from multidiffusion.utils.visualization import save_grid
from multidiffusion.utils.logging import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Launch Gradio demo")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--share", action="store_true",
                      help="Create public link")
    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)

class DiffusionDemo:
    def __init__(self, config_path, checkpoint_path):
        # Load config
        self.config = load_config(config_path)
        
        # Setup logging
        self.logger = setup_logging(self.config)
        
        # Create trainer and load checkpoint
        self.trainer = get_trainer(self.config)
        self.trainer.load_checkpoint(checkpoint_path)
        self.trainer.model.eval()
        
        # Create temporary directory for samples
        os.makedirs("temp", exist_ok=True)
        
    def generate(self, num_samples, batch_size, seed):
        if seed is not None:
            torch.manual_seed(seed)
            
        all_samples = []
        for i in range(0, num_samples, batch_size):
            batch_size = min(batch_size, num_samples - i)
            
            with torch.no_grad():
                samples = self.trainer.model.sample(batch_size, self.trainer.device)
            all_samples.append(samples)
            
        all_samples = torch.cat(all_samples)
        
        # Save grid
        output_path = os.path.join("temp", "grid.png")
        save_grid(all_samples, output_path, title="Generated Samples")
        
        return output_path

def main():
    # Parse arguments
    args = parse_args()
    
    # Create demo
    demo = DiffusionDemo(args.config, args.checkpoint)
    
    # Create interface
    interface = gr.Interface(
        fn=demo.generate,
        inputs=[
            gr.Slider(minimum=1, maximum=64, step=1, value=16,
                     label="Number of Samples"),
            gr.Slider(minimum=1, maximum=16, step=1, value=4,
                     label="Batch Size"),
            gr.Number(label="Random Seed (optional)", precision=0)
        ],
        outputs=gr.Image(type="filepath", label="Generated Samples"),
        title="Diffusion Model Demo",
        description="Generate images using a trained diffusion model",
        article="""
        This demo uses a diffusion model trained on various datasets to generate images.
        You can control:
        - Number of samples to generate
        - Batch size for generation
        - Random seed for reproducibility
        """,
        examples=[
            [16, 4, 42],
            [32, 8, 123],
            [64, 16, 456]
        ]
    )
    
    # Launch interface
    interface.launch(share=args.share)

if __name__ == "__main__":
    main() 