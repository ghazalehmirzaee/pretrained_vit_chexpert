import os
import yaml
import torch
import wandb
import argparse
import json
from pathlib import Path
import numpy as np
from datetime import datetime

from src.data.dataset import CheXpertDataset
from src.data.transforms import CheXpertTransforms
from src.models.vit import VisionTransformer
from src.training.trainer import Trainer
from src.utils.logging import setup_logging, log_system_info, log_training_start
from src.utils.metrics import MetricCalculator
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CheXpert model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config: dict):
    """Initialize Weights & Biases"""
    run = wandb.init(
        project=config['wandb']['project'],
        name=config['wandb']['name'],
        entity=config['wandb']['entity'],
        config=config,
        reinit=True
    )
    return run


def create_dataloaders(config: dict):
    """Create training and validation dataloaders"""
    # Create transforms
    train_transform = CheXpertTransforms.get_transforms('train', config['data']['image_size'])
    val_transform = CheXpertTransforms.get_transforms('val', config['data']['image_size'])

    # Create datasets
    train_dataset = CheXpertDataset(
        csv_path=config['data']['train_csv'],
        image_root_dir=config['data']['train_dir'],
        uncertainty_strategy=config['data']['uncertainty_strategy'],
        transform=train_transform
    )

    val_dataset = CheXpertDataset(
        csv_path=config['data']['valid_csv'],
        image_root_dir=config['data']['val_dir'],
        uncertainty_strategy=config['data']['uncertainty_strategy'],
        transform=val_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader


def create_model(config: dict, device: torch.device):
    """Create and initialize the model"""
    model = VisionTransformer(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        in_chans=config['model']['in_chans'],
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        mlp_ratio=config['model']['mlp_ratio'],
        drop_rate=config['model']['drop_rate']
    )

    # Load pre-trained weights if specified
    if os.path.exists(config['model']['pretrained_path']):
        model.load_state_dict(torch.load(config['model']['pretrained_path']))

    return model.to(device)


def main():
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load disease classes
    with open('configs/disease_classes.json', 'r') as f:
        disease_config = json.load(f)

    # Setup directories
    for dir_name in ['save_dir', 'log_dir']:
        Path(config['paths'][dir_name]).mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(config)
    log_system_info()
    log_training_start(config)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    try:
        # Initialize wandb
        run = setup_wandb(config)

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(config)

        # Create model
        model = create_model(config, device)

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )

        # Resume from checkpoint if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)

        # Train model
        trainer.train()

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

    finally:
        # Cleanup
        wandb.finish()
        logger.info("Training script completed")


if __name__ == '__main__':
    main()

