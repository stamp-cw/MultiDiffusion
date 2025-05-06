"""
Logging utilities
"""

import os
import sys
from loguru import logger

def setup_logging(config):
    """Setup logging configuration"""
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler
    os.makedirs(config.logging.log_dir, exist_ok=True)
    logger.add(
        os.path.join(config.logging.log_dir, "train.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="1 day"
    )
    
    return logger

def get_logger(name):
    """Get logger instance"""
    return logger.bind(name=name) 