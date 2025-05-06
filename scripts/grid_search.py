"""
Grid search script for hyperparameter optimization
"""

import os
import itertools
import argparse
import yaml
import copy
from types import SimpleNamespace
from datetime import datetime
import pandas as pd
from multidiffusion.trainers import get_trainer
from multidiffusion.utils.logging import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter grid search")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to base config file")
    parser.add_argument("--grid_config", type=str, required=True,
                      help="Path to grid search config file")
    parser.add_argument("--output_dir", type=str, default="grid_search_results",
                      help="Directory to save results")
    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)

def update_nested_dict(d, key_path, value):
    """Update nested dictionary using dot notation key path"""
    keys = key_path.split('.')
    current = d
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value
    return d

def run_single_trial(base_config, param_dict, trial_dir):
    """Run a single trial with given parameters"""
    # Create copy of base config
    config_dict = copy.deepcopy(vars(base_config))
    
    # Update config with trial parameters
    for param_path, value in param_dict.items():
        config_dict = update_nested_dict(config_dict, param_path, value)
    
    # Convert back to namespace
    config = SimpleNamespace(**config_dict)
    
    # Update paths for this trial
    config.logging.log_dir = os.path.join(trial_dir, "logs")
    config.logging.checkpoint_dir = os.path.join(trial_dir, "checkpoints")
    config.logging.sample_dir = os.path.join(trial_dir, "samples")
    
    # Create directories
    os.makedirs(config.logging.log_dir, exist_ok=True)
    os.makedirs(config.logging.checkpoint_dir, exist_ok=True)
    os.makedirs(config.logging.sample_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Starting trial with parameters: {param_dict}")
    
    # Create and train model
    trainer = get_trainer(config)
    trainer.train()
    
    # Return best loss achieved
    return trainer.best_loss

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configs
    base_config = load_config(args.config)
    with open(args.grid_config, "r") as f:
        grid_config = yaml.safe_load(f)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate parameter combinations
    param_names = list(grid_config.keys())
    param_values = [grid_config[name] for name in param_names]
    param_combinations = list(itertools.product(*param_values))
    
    # Store results
    results = []
    
    # Run trials
    for i, values in enumerate(param_combinations):
        param_dict = dict(zip(param_names, values))
        trial_dir = os.path.join(output_dir, f"trial_{i:03d}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Save trial config
        trial_config = {
            "parameters": param_dict,
            "base_config": args.config,
            "trial_id": i
        }
        with open(os.path.join(trial_dir, "trial_config.yaml"), "w") as f:
            yaml.dump(trial_config, f)
        
        # Run trial
        try:
            best_loss = run_single_trial(base_config, param_dict, trial_dir)
            status = "completed"
        except Exception as e:
            best_loss = None
            status = f"failed: {str(e)}"
        
        # Store results
        result = {
            "trial_id": i,
            "status": status,
            "best_loss": best_loss,
            **param_dict
        }
        results.append(result)
        
        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
        
        # Save best parameters so far
        if status == "completed":
            completed_trials = df[df["status"] == "completed"]
            if not completed_trials.empty:
                best_trial = completed_trials.loc[completed_trials["best_loss"].idxmin()]
                best_params = {
                    "trial_id": int(best_trial["trial_id"]),
                    "best_loss": float(best_trial["best_loss"]),
                    "parameters": {k: v for k, v in best_trial.items() 
                                 if k not in ["trial_id", "status", "best_loss"]}
                }
                with open(os.path.join(output_dir, "best_parameters.yaml"), "w") as f:
                    yaml.dump(best_params, f)
    
    print(f"\nGrid search completed. Results saved to {output_dir}")
    
    # Print best results
    completed_trials = df[df["status"] == "completed"]
    if not completed_trials.empty:
        best_trial = completed_trials.loc[completed_trials["best_loss"].idxmin()]
        print("\nBest parameters found:")
        for param, value in best_trial.items():
            if param not in ["trial_id", "status", "best_loss"]:
                print(f"{param}: {value}")
        print(f"Best loss: {best_trial['best_loss']:.4f}")

if __name__ == "__main__":
    main() 