import torch
import numpy as np
from typing import Union, Dict, List
from enum import Enum


class UncertaintyStrategy(Enum):
    """Enumeration of uncertainty handling strategies"""
    U_IGNORE = "U-Ignore"
    U_ZEROS = "U-Zeros"
    U_ONES = "U-Ones"
    U_MULTICLASS = "U-MultiClass"
    U_SELFTRAINED = "U-SelfTrained"


class UncertaintyHandler:
    """Handler for different uncertainty strategies in CheXpert"""

    def __init__(self, strategy: str):
        """
        Initialize uncertainty handler

        Args:
            strategy: One of U-Ignore, U-Zeros, U-Ones, U-MultiClass, U-SelfTrained
        """
        self.strategy = UncertaintyStrategy(strategy)

    def process_labels(self,
                       labels: np.ndarray,
                       self_trained_probs: Dict[str, np.ndarray] = None) -> np.ndarray:
        """
        Process labels according to uncertainty strategy

        Args:
            labels: Original labels with -1 for uncertain
            self_trained_probs: Predictions from pretrained model for U-SelfTrained

        Returns:
            Processed labels
        """
        if self.strategy == UncertaintyStrategy.U_ZEROS:
            return self._process_u_zeros(labels)
        elif self.strategy == UncertaintyStrategy.U_ONES:
            return self._process_u_ones(labels)
        elif self.strategy == UncertaintyStrategy.U_MULTICLASS:
            return self._process_u_multiclass(labels)
        elif self.strategy == UncertaintyStrategy.U_SELFTRAINED:
            assert self_trained_probs is not None, "Need model predictions for U-SelfTrained"
            return self._process_u_selftrained(labels, self_trained_probs)
        else:  # U_IGNORE
            return labels

    @staticmethod
    def _process_u_zeros(labels: np.ndarray) -> np.ndarray:
        """Convert uncertain labels to zeros"""
        processed = labels.copy()
        processed[processed == -1] = 0
        return processed

    @staticmethod
    def _process_u_ones(labels: np.ndarray) -> np.ndarray:
        """Convert uncertain labels to ones"""
        processed = labels.copy()
        processed[processed == -1] = 1
        return processed

    @staticmethod
    def _process_u_multiclass(labels: np.ndarray) -> np.ndarray:
        """Convert to one-hot encoding with 3 classes"""
        num_samples, num_classes = labels.shape
        processed = np.zeros((num_samples, num_classes, 3))

        for i in range(num_samples):
            for j in range(num_classes):
                if labels[i, j] == 1:
                    processed[i, j, 1] = 1  # Positive
                elif labels[i, j] == 0:
                    processed[i, j, 0] = 1  # Negative
                elif labels[i, j] == -1:
                    processed[i, j, 2] = 1  # Uncertain

        return processed

    @staticmethod
    def _process_u_selftrained(labels: np.ndarray,
                               predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Replace uncertain labels with model predictions"""
        processed = labels.copy()
        uncertain_mask = (processed == -1)

        if predictions is not None:
            for disease_idx in range(labels.shape[1]):
                disease_preds = predictions[disease_idx]
                disease_uncertain = uncertain_mask[:, disease_idx]
                processed[disease_uncertain, disease_idx] = \
                    (disease_preds[disease_uncertain] > 0.5).astype(float)

        return processed

    def get_loss_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Get mask for loss computation

        Args:
            labels: Tensor of labels

        Returns:
            Boolean mask indicating which labels to use in loss
        """
        if self.strategy == UncertaintyStrategy.U_IGNORE:
            return labels != -1
        else:
            return torch.ones_like(labels, dtype=torch.bool)

