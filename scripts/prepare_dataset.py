import os
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import logging
import argparse
from typing import Tuple


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('prepare_dataset')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Prepare CheXpert dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to CheXpert-v1.0 directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed dataset')
    return parser.parse_args()


def process_csv(csv_path: str, output_path: str) -> pd.DataFrame:
    """
    Process CheXpert CSV file

    Args:
        csv_path: Path to original CSV file
        output_path: Path to save processed CSV

    Returns:
        Processed DataFrame
    """
    logger = logging.getLogger('prepare_dataset')
    logger.info(f"Processing {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Fill missing values with 0
    df.fillna(0, inplace=True)

    # Get disease columns (columns after 'Path')
    disease_cols = df.columns[df.columns.get_loc('Path') + 1:].tolist()

    # Convert uncertain labels (-1) to NaN
    for col in disease_cols:
        df[col] = df[col].replace(-1, np.nan)

    # Save processed CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed CSV to {output_path}")

    return df


def organize_images(df: pd.DataFrame, source_dir: str, target_dir: str) -> None:
    """
    Organize images into a cleaner directory structure

    Args:
        df: DataFrame with image paths
        source_dir: Source directory containing original images
        target_dir: Target directory for organized images
    """
    logger = logging.getLogger('prepare_dataset')
    logger.info(f"Organizing images into {target_dir}")

    # Create target directory
    os.makedirs(target_dir, exist_ok=True)

    # Copy images
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            logger.info(f"Processed {idx} images")

        # Get source and target paths
        source_path = os.path.join(source_dir, row['Path'])
        target_path = os.path.join(target_dir, os.path.basename(row['Path']))

        # Create target directory if needed
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Copy image
        shutil.copy2(source_path, target_path)

    logger.info("Finished organizing images")


def main():
    """Main function"""
    # Setup
    logger = setup_logging()
    args = parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Process train set
    train_df = process_csv(
        os.path.join(args.data_dir, 'train.csv'),
        os.path.join(args.output_dir, 'train.csv')
    )

    # Process validation set
    val_df = process_csv(
        os.path.join(args.data_dir, 'valid.csv'),
        os.path.join(args.output_dir, 'valid.csv')
    )

    # Organize images
    organize_images(
        train_df,
        args.data_dir,
        os.path.join(args.output_dir, 'train')
    )
    organize_images(
        val_df,
        args.data_dir,
        os.path.join(args.output_dir, 'valid')
    )

    logger.info("Dataset preparation completed!")


if __name__ == '__main__':
    main()

