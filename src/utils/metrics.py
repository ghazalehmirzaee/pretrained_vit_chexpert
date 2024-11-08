import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class MetricCalculator:
    """Calculate and track metrics for CheXpert"""

    def __init__(self, disease_names: List[str]):
        """
        Initialize metric calculator

        Args:
            disease_names: List of disease names
        """
        self.disease_names = disease_names
        self.competition_tasks = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion"
        ]

    def calculate_metrics(self,
                          targets: np.ndarray,
                          predictions: np.ndarray,
                          threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate metrics for all diseases

        Args:
            targets: Ground truth labels
            predictions: Model predictions
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Handle multi-class case
        if len(targets.shape) == 3:  # (N, C, 3) shape
            targets = targets[:, :, 1]  # Use positive class

        # Calculate metrics for each disease
        for i, disease in enumerate(self.disease_names):
            disease_targets = targets[:, i]
            disease_preds = predictions[:, i]

            # Skip if all targets are same class
            if len(np.unique(disease_targets)) < 2:
                continue

            try:
                # ROC AUC
                auc = roc_auc_score(disease_targets, disease_preds)
                metrics[f"{disease}_auc"] = auc

                # Average Precision
                ap = average_precision_score(disease_targets, disease_preds)
                metrics[f"{disease}_ap"] = ap

                # Binary metrics
                binary_preds = (disease_preds > threshold).astype(float)
                tp = np.sum((binary_preds == 1) & (disease_targets == 1))
                fp = np.sum((binary_preds == 1) & (disease_targets == 0))
                fn = np.sum((binary_preds == 0) & (disease_targets == 1))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                metrics[f"{disease}_precision"] = precision
                metrics[f"{disease}_recall"] = recall
                metrics[f"{disease}_f1"] = f1

            except Exception as e:
                logger.warning(f"Error calculating metrics for {disease}: {str(e)}")

        # Calculate mean metrics for competition tasks
        competition_aucs = [metrics[f"{d}_auc"] for d in self.competition_tasks
                            if f"{d}_auc" in metrics]
        if competition_aucs:
            metrics["mean_competition_auc"] = np.mean(competition_aucs)

        # Calculate mean metrics across all diseases
        all_aucs = [v for k, v in metrics.items() if k.endswith("_auc")]
        if all_aucs:
            metrics["mean_auc"] = np.mean(all_aucs)

        return metrics

    def plot_curves(self,
                    targets: np.ndarray,
                    predictions: np.ndarray,
                    save_dir: str = None):
        """
        Plot ROC and PR curves

        Args:
            targets: Ground truth labels
            predictions: Model predictions
            save_dir: Directory to save plots
        """
        from sklearn.metrics import roc_curve
        import matplotlib.pyplot as plt

        for i, disease in enumerate(self.disease_names):
            disease_targets = targets[:, i]
            disease_preds = predictions[:, i]

            if len(np.unique(disease_targets)) < 2:
                continue

            # Plot ROC curve
            fpr, tpr, _ = roc_curve(disease_targets, disease_preds)
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {disease}')

            if save_dir:
                plt.savefig(f"{save_dir}/roc_{disease}.png")
            plt.close()

            # Plot PR curve
            precision, recall, _ = precision_recall_curve(disease_targets, disease_preds)
            plt.figure(figsize=(8, 8))
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {disease}')

            if save_dir:
                plt.savefig(f"{save_dir}/pr_{disease}.png")
            plt.close()

