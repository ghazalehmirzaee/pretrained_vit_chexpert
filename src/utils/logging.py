import logging
import sys
import os
from datetime import datetime
import json
import torch
import platform
from typing import Dict, Any


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        config: Configuration dictionary

    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    log_dir = config['paths']['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger('chexpert')
    return logger


def log_system_info() -> None:
    """Log system information"""
    logger = logging.getLogger('chexpert')

    # System info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")

    # PyTorch info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


def log_config(config: Dict[str, Any]) -> None:
    """
    Log configuration parameters

    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger('chexpert')
    logger.info("Configuration:")
    logger.info(json.dumps(config, indent=2))


def log_epoch_metrics(epoch: int, metrics: Dict[str, float], mode: str = 'train') -> None:
    """
    Log metrics for an epoch

    Args:
        epoch: Current epoch number
        metrics: Dictionary of metrics
        mode: Either 'train' or 'val'
    """
    logger = logging.getLogger('chexpert')

    # Format metrics string
    metrics_str = ' | '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Epoch {epoch} ({mode}): {metrics_str}")


def log_batch_metrics(epoch: int, batch: int, metrics: Dict[str, float],
                      total_batches: int) -> None:
    """
    Log metrics for a batch

    Args:
        epoch: Current epoch number
        batch: Current batch number
        metrics: Dictionary of metrics
        total_batches: Total number of batches
    """
    logger = logging.getLogger('chexpert')

    # Format metrics string
    metrics_str = ' | '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Epoch {epoch} [{batch}/{total_batches}]: {metrics_str}")


def log_training_start(config: Dict[str, Any]) -> None:
    """
    Log training start information

    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger('chexpert')
    logger.info("=" * 50)
    logger.info("Starting training with configuration:")
    log_config(config)
    logger.info("=" * 50)

