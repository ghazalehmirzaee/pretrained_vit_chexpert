import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import os
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class CheXpertDataset(Dataset):
    """CheXpert dataset with uncertainty handling"""

    def __init__(self,
                 csv_path: str,
                 image_root_dir: str,
                 uncertainty_strategy: str = "U-Zeros",
                 transform: Optional[Callable] = None):
        """
        Args:
            csv_path: Path to CheXpert CSV file
            image_root_dir: Root directory containing images
            uncertainty_strategy: How to handle uncertainty labels
                Options: U-Ignore, U-Zeros, U-Ones, U-MultiClass, U-SelfTrained
            transform: Optional transforms to be applied on images
        """
        self.df = pd.read_csv(csv_path)
        self.image_root_dir = image_root_dir
        self.transform = transform
        self.uncertainty_strategy = uncertainty_strategy

        # Get disease names (excluding Path and demographic columns)
        self.disease_names = self.df.columns[5:].tolist()

        # Convert labels based on uncertainty strategy
        self.labels = self._process_labels()

        logger.info(f"Loaded {len(self.df)} images with {len(self.disease_names)} diseases")
        logger.info(f"Using uncertainty strategy: {uncertainty_strategy}")
        self._log_class_distribution()

    def _process_labels(self):
        """Process labels according to uncertainty strategy"""
        labels = self.df.iloc[:, 5:].values  # Get all disease labels

        if self.uncertainty_strategy == "U-Zeros":
            # Convert -1 (uncertain) to 0, and keep 0 and 1 as is
            labels[labels == -1] = 0
        elif self.uncertainty_strategy == "U-Ones":
            # Convert -1 (uncertain) to 1, and keep 0 and 1 as is
            labels[labels == -1] = 1
        elif self.uncertainty_strategy == "U-MultiClass":
            # Convert to one-hot encoding with 3 classes per disease
            new_labels = np.zeros((len(labels), len(self.disease_names), 3))
            for i in range(len(labels)):
                for j in range(len(self.disease_names)):
                    if labels[i, j] == 1:
                        new_labels[i, j, 1] = 1  # Positive class
                    elif labels[i, j] == 0:
                        new_labels[i, j, 0] = 1  # Negative class
                    elif labels[i, j] == -1:
                        new_labels[i, j, 2] = 1  # Uncertain class
            labels = new_labels
        elif self.uncertainty_strategy == "U-Ignore":
            # Keep -1 values, they will be masked in loss computation
            pass

        return labels

    def _log_class_distribution(self):
        """Log the distribution of classes for each disease"""
        for i, disease in enumerate(self.disease_names):
            if self.uncertainty_strategy == "U-MultiClass":
                neg = np.sum(self.labels[:, i, 0])
                pos = np.sum(self.labels[:, i, 1])
                unc = np.sum(self.labels[:, i, 2])
                total = len(self.labels)
                logger.info(f"{disease}: Positive: {pos / total:.2%}, "
                            f"Negative: {neg / total:.2%}, "
                            f"Uncertain: {unc / total:.2%}")
            else:
                values, counts = np.unique(self.labels[:, i], return_counts=True)
                dist = dict(zip(values, counts))
                logger.info(f"{disease}: {dist}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image path
        image_path = os.path.join(self.image_root_dir, self.df.iloc[idx]['Path'])

        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            # Get labels for this image
            if self.uncertainty_strategy == "U-MultiClass":
                labels = torch.FloatTensor(self.labels[idx])
            else:
                labels = torch.FloatTensor(self.labels[idx])

            return image, labels

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise

