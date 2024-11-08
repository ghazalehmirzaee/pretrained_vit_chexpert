import torch
from torchvision import transforms
from typing import Tuple


class CheXpertTransforms:
    """Transformations for CheXpert images"""

    @staticmethod
    def get_transforms(mode: str = 'train', image_size: int = 320) -> transforms.Compose:
        """
        Get transformation pipeline

        Args:
            mode: Either 'train' or 'val'
            image_size: Target image size

        Returns:
            Composition of transforms
        """
        # Normalization parameters from ImageNet
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if mode == 'train':
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=(-5, 5),
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                ),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2
                ),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ])

    @staticmethod
    def get_inverse_transform() -> transforms.Compose:
        """Get inverse normalization transform for visualization"""
        return transforms.Compose([
            transforms.Normalize(
                mean=[0., 0., 0.],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(
                mean=[-0.485, -0.456, -0.406],
                std=[1., 1., 1.]
            ),
        ])

