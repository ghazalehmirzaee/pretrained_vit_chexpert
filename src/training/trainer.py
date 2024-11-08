import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import numpy as np
import logging
import os
from datetime import datetime
from src.utils.metrics import MetricCalculator
from src.training.losses import CheXpertLoss
import torch.cuda.amp as amp
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for CheXpert model"""

    def __init__(self, model: nn.Module, train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader, config: dict, device: torch.device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.current_epoch = 0

        # Setup loss function
        self.criterion = CheXpertLoss(
            num_classes=config['model']['num_classes'],
            uncertainty_strategy=config['data']['uncertainty_strategy']
        ).to(device)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            [
                {'params': model.parameters()},
                {'params': self.criterion.parameters(),
                 'lr': config['training']['learning_rate'] * 0.1}
            ],
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
            eps=config['optimizer']['eps']
        )

        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()

        # Setup metric calculator
        with open('configs/disease_classes.json', 'r') as f:
            import json
            disease_config = json.load(f)
            self.disease_names = [d['name'] for d in disease_config['diseases']]
        self.metric_calculator = MetricCalculator(self.disease_names)

        # Setup mixed precision training
        self.scaler = amp.GradScaler()

        # Setup tracking variables
        self.best_val_auc = 0
        self.best_epoch = 0
        self.patience_counter = 0

        logger.info("Trainer initialized")

    def _setup_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Setup learning rate scheduler with warmup"""
        warmup_steps = len(self.train_loader) * self.config['training']['warmup_epochs']
        total_steps = len(self.train_loader) * self.config['training']['epochs']

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_auc': self.best_val_auc,
            'metrics': metrics
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(
            self.config['paths']['save_dir'],
            f'checkpoint_epoch_{self.current_epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(
                self.config['paths']['save_dir'],
                'best_model.pt'
            )
            torch.save(checkpoint, best_path)
            logger.info(f"Saved new best model with AUC: {metrics['mean_competition_auc']:.4f}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_auc = checkpoint['best_val_auc']

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_predictions = []
        epoch_targets = []
        epoch_losses = {'total': [], 'wbce': [], 'focal': [], 'asl': []}

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            with amp.autocast():
                outputs = self.model(images)
                loss, loss_components = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config['training']['max_grad_norm'] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update learning rate
            self.scheduler.step()

            # Record batch results
            with torch.no_grad():
                preds = torch.sigmoid(outputs)
                epoch_predictions.append(preds.cpu().numpy())
                epoch_targets.append(targets.cpu().numpy())
                epoch_losses['total'].append(loss.item())
                for k, v in loss_components.items():
                    epoch_losses[k].append(v.item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            # Log to wandb
            if batch_idx % 100 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    **{f'train/batch_loss_{k}': v[-1] for k, v in epoch_losses.items()}
                })

        # Calculate epoch metrics
        predictions = np.vstack(epoch_predictions)
        targets = np.vstack(epoch_targets)
        metrics = self.metric_calculator.calculate_metrics(targets, predictions)

        # Add losses to metrics
        metrics.update({
            f'loss_{k}': np.mean(v) for k, v in epoch_losses.items()
        })

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_predictions = []
        val_targets = []
        val_losses = []

        pbar = tqdm(self.val_loader, desc='Validation')
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            with amp.autocast():
                outputs = self.model(images)
                loss, _ = self.criterion(outputs, targets)

            # Record batch results
            preds = torch.sigmoid(outputs)
            val_predictions.append(preds.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
            val_losses.append(loss.item())

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Calculate validation metrics
        predictions = np.vstack(val_predictions)
        targets = np.vstack(val_targets)
        metrics = self.metric_calculator.calculate_metrics(targets, predictions)
        metrics['loss'] = np.mean(val_losses)

        return metrics

    def train(self) -> Dict[str, float]:
        """Main training loop"""
        logger.info("Starting training...")
        early_stopping_patience = self.config['training']['early_stopping_patience']

        try:
            for epoch in range(self.current_epoch, self.config['training']['epochs']):
                self.current_epoch = epoch

                # Train epoch
                train_metrics = self.train_epoch()

                # Validate
                val_metrics = self.validate()

                # Log metrics
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss_total'],
                    'train/mean_auc': train_metrics['mean_auc'],
                    'train/mean_competition_auc': train_metrics['mean_competition_auc'],
                    'val/loss': val_metrics['loss'],
                    'val/mean_auc': val_metrics['mean_auc'],
                    'val/mean_competition_auc': val_metrics['mean_competition_auc'],
                })

                # Log disease-specific metrics
                for disease in self.disease_names:
                    if f"{disease}_auc" in train_metrics and f"{disease}_auc" in val_metrics:
                        wandb.log({
                            f'train/auc_{disease}': train_metrics[f"{disease}_auc"],
                            f'val/auc_{disease}': val_metrics[f"{disease}_auc"]
                        })

                # Print epoch summary
                logger.info(
                    f"Epoch {epoch + 1}/{self.config['training']['epochs']} - "
                    f"Train Loss: {train_metrics['loss_total']:.4f}, "
                    f"Train AUC: {train_metrics['mean_competition_auc']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val AUC: {val_metrics['mean_competition_auc']:.4f}"
                )

                # Save checkpoint if best model
                is_best = val_metrics['mean_competition_auc'] > self.best_val_auc
                if is_best:
                    self.best_val_auc = val_metrics['mean_competition_auc']
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self.save_checkpoint(val_metrics, is_best=True)
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {early_stopping_patience} "
                        "epochs without improvement"
                    )
                    break

                # Save regular checkpoint
                if (epoch + 1) % self.config['training']['save_freq'] == 0:
                    self.save_checkpoint(val_metrics)

            # Training completed
            logger.info(f"Training completed! Best validation AUC: {self.best_val_auc:.4f} "
                        f"at epoch {self.best_epoch}")

            return {
                'best_val_auc': float(self.best_val_auc),
                'best_epoch': self.best_epoch,
                'final_train_loss': float(train_metrics['loss_total']),
                'final_train_auc': float(train_metrics['mean_competition_auc']),
                'final_val_loss': float(val_metrics['loss']),
                'final_val_auc': float(val_metrics['mean_competition_auc'])
            }

        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise