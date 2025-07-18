"""
Data generator for training tumor detection models.
Handles data loading, preprocessing, and augmentation.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, List, Optional, Dict, Any, Union
import os
import cv2
from pathlib import Path
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data_processing.medical_image_loader import MedicalImageLoader

logger = logging.getLogger(__name__)


class TumorDataGenerator(keras.utils.Sequence):
    """
    Data generator for tumor detection training with augmentation.
    """
    
    def __init__(self, image_paths: List[str], mask_paths: List[str],
                 batch_size: int = 8, input_size: Tuple[int, int] = (256, 256),
                 channels: int = 1, shuffle: bool = True,
                 augmentation: bool = True, normalize: bool = True):
        """
        Initialize the data generator.
        
        Args:
            image_paths: List of paths to image files
            mask_paths: List of paths to mask files
            batch_size: Batch size for training
            input_size: Target input size (height, width)
            channels: Number of input channels
            shuffle: Whether to shuffle data
            augmentation: Whether to apply data augmentation
            normalize: Whether to normalize images
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.input_size = input_size
        self.channels = channels
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.normalize = normalize
        
        self.image_loader = MedicalImageLoader()
        self.indexes = np.arange(len(self.image_paths))
        
        # Initialize augmentation pipeline
        if self.augmentation:
            self.augmentation_pipeline = self._create_augmentation_pipeline()
        
        # Shuffle if requested
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        logger.info(f"Data generator initialized with {len(self.image_paths)} samples")
    
    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one batch of data.
        
        Args:
            index: Batch index
            
        Returns:
            Tuple of (images, masks)
        """
        # Get batch indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Initialize batch arrays
        batch_images = []
        batch_masks = []
        
        # Load and process each sample in the batch
        for idx in batch_indexes:
            try:
                # Load image
                image_data = self.image_loader.load_image(self.image_paths[idx])
                image = image_data['image']
                
                # Load mask
                mask_data = self.image_loader.load_image(self.mask_paths[idx])
                mask = mask_data['image']
                
                # Preprocess image and mask
                image, mask = self._preprocess_sample(image, mask)
                
                # --- FORCE 3 CHANNELS ---
                if image.ndim == 2:
                    image = np.stack([image]*3, axis=-1)
                elif image.shape[2] == 1:
                    image = np.concatenate([image]*3, axis=-1)
                print(f"Image shape before batch: {image.shape}")
                batch_images.append(image)
                batch_masks.append(mask)
                
            except Exception as e:
                logger.warning(f"Error loading sample {idx}: {e}")
                # Add a zero-filled sample as fallback
                batch_images.append(np.zeros((*self.input_size, 3)))
                batch_masks.append(np.zeros((*self.input_size, 1)))
        
        # Convert to numpy arrays
        batch_images = np.array(batch_images, dtype=np.float32)
        batch_masks = np.array(batch_masks, dtype=np.float32)
        print(f"Batch images shape: {batch_images.shape}")
        
        return batch_images, batch_masks
    
    def _preprocess_sample(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a single image-mask pair.
        Args:
            image: Input image
            mask: Input mask
        Returns:
            Tuple of (processed_image, processed_mask)
        """
        # --- DO NOT convert 3-channel images to grayscale ---
        # Only ensure mask is 2D
        if len(mask.shape) == 3 and mask.shape[2] > 1:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        # Resize to target size
        image = cv2.resize(image, (self.input_size[1], self.input_size[0]), 
                          interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.input_size[1], self.input_size[0]), 
                         interpolation=cv2.INTER_NEAREST)
        # Normalize image
        if self.normalize:
            image = self.image_loader.normalize_image(image)
        # Binarize mask
        mask = (mask > 0.5).astype(np.float32)
        # Apply augmentation if enabled
        if self.augmentation:
            image, mask = self._apply_augmentation(image, mask)
        # Only add channel dimension if image is 2D
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
        return image, mask
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """
        Create data augmentation pipeline using Albumentations.
        
        Returns:
            Augmentation pipeline
        """
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.GaussNoise(p=0.2),
                A.GaussNoise(p=0.2),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.PiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ])
    
    def _apply_augmentation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation to image and mask.
        
        Args:
            image: Input image
            mask: Input mask
            
        Returns:
            Tuple of (augmented_image, augmented_mask)
        """
        # Convert to uint8 for augmentation
        image_uint8 = (image * 255).astype(np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Apply augmentation
        augmented = self.augmentation_pipeline(image=image_uint8, mask=mask_uint8)
        
        # Convert back to float32
        augmented_image = augmented['image'].astype(np.float32) / 255.0
        augmented_mask = (augmented['mask'] > 127).astype(np.float32)
        
        return augmented_image, augmented_mask
    
    def on_epoch_end(self):
        """Called at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)


class TumorDataManager:
    """
    Data manager for organizing and preparing training data.
    """
    
    def __init__(self, data_dir: str, config: Dict[str, Any]):
        """
        Initialize the data manager.
        
        Args:
            data_dir: Root directory containing data
            config: Configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.image_loader = MedicalImageLoader()
        
        # Extract configuration parameters
        self.input_size = tuple(config['data']['input_size'])
        self.batch_size = config['data']['batch_size']
        self.channels = config['data']['channels']
        self.validation_split = config['data']['validation_split']
        self.test_split = config['data']['test_split']
        
        # Data paths
        self.raw_data_path = Path(config['data']['raw_data_path'])
        self.processed_data_path = Path(config['data']['processed_data_path'])
        self.masks_path = Path(config['data']['masks_path'])
        
        # Supported formats
        self.supported_formats = config['data']['supported_formats']
        
        logger.info("Tumor data manager initialized")
    
    def discover_data(self):
        print("Looking for images in:", self.raw_data_path)
        print("Looking for masks in:", self.masks_path)
        print("Supported formats:", self.supported_formats)
        print("Found images:", os.listdir(self.raw_data_path))
        print("Found masks:", os.listdir(self.masks_path))
        images = [f for f in os.listdir(self.raw_data_path) if f.endswith('.tif') and not f.endswith('_mask.tif')]
        masks = [f for f in os.listdir(self.masks_path) if f.endswith('_mask.tif')]

        image_paths = []
        mask_paths = []

        for img in images:
            base = img[:-4]  # remove '.tif'
            mask_name = base + '_mask.tif'
            if mask_name in masks:
                image_paths.append(os.path.join(self.raw_data_path, img))
                mask_paths.append(os.path.join(self.masks_path, mask_name))

        print(f"Found {len(image_paths)} image/mask pairs")
        return {'images': image_paths, 'masks': mask_paths}
    
    def split_data(self, image_paths: List[str], mask_paths: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """
        Split data into train/validation/test sets.
        
        Args:
            image_paths: List of image paths
            mask_paths: List of mask paths
            
        Returns:
            Dictionary with train/val/test splits
        """
        # Ensure we have matching pairs
        if len(image_paths) != len(mask_paths):
            raise ValueError("Number of images and masks must match")
        
        # Create pairs and shuffle
        pairs = list(zip(image_paths, mask_paths))
        np.random.shuffle(pairs)
        
        # Calculate split indices
        total_samples = len(pairs)
        test_size = int(total_samples * self.test_split)
        val_size = int(total_samples * self.validation_split)
        train_size = total_samples - test_size - val_size
        
        # Split data
        train_pairs = pairs[:train_size]
        val_pairs = pairs[train_size:train_size + val_size]
        test_pairs = pairs[train_size + val_size:]
        
        # Organize splits
        splits = {
            'train': {
                'images': [pair[0] for pair in train_pairs],
                'masks': [pair[1] for pair in train_pairs]
            },
            'validation': {
                'images': [pair[0] for pair in val_pairs],
                'masks': [pair[1] for pair in val_pairs]
            },
            'test': {
                'images': [pair[0] for pair in test_pairs],
                'masks': [pair[1] for pair in test_pairs]
            }
        }
        
        logger.info(f"Data split: {len(train_pairs)} train, {len(val_pairs)} validation, {len(test_pairs)} test")
        
        return splits
    
    def create_generators(self, splits: Dict[str, Dict[str, List[str]]]) -> Dict[str, TumorDataGenerator]:
        """
        Create data generators for each split.
        
        Args:
            splits: Data splits dictionary
            
        Returns:
            Dictionary of data generators
        """
        generators = {}
        
        for split_name, split_data in splits.items():
            # Determine augmentation and shuffle settings
            augmentation = split_name == 'train'
            shuffle = split_name == 'train'
            
            generator = TumorDataGenerator(
                image_paths=split_data['images'],
                mask_paths=split_data['masks'],
                batch_size=self.batch_size,
                input_size=self.input_size,
                channels=self.channels,
                shuffle=shuffle,
                augmentation=augmentation,
                normalize=True
            )
            
            generators[split_name] = generator
        
        logger.info(f"Created data generators for {list(generators.keys())}")
        return generators
    
    def get_data_info(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'total_samples': len(image_paths),
            'formats': {},
            'sizes': [],
            'modalities': []
        }
        
        for path in image_paths[:100]:  # Sample first 100 for efficiency
            try:
                file_info = self.image_loader.get_image_info(path)
                
                # Count formats
                ext = file_info['extension']
                info['formats'][ext] = info['formats'].get(ext, 0) + 1
                
                # Collect sizes
                if 'size' in file_info:
                    info['sizes'].append(file_info['size'])
                
                # Collect modalities
                if 'modality' in file_info:
                    info['modalities'].append(file_info['modality'])
                    
            except Exception as e:
                logger.warning(f"Could not get info for {path}: {e}")
        
        # Calculate statistics
        if info['sizes']:
            info['size_stats'] = {
                'min': min(info['sizes']),
                'max': max(info['sizes']),
                'mean': np.mean(info['sizes'])
            }
        
        if info['modalities']:
            info['modality_counts'] = {}
            for modality in info['modalities']:
                info['modality_counts'][modality] = info['modalities'].count(modality)
        
        return info


def create_data_generators(config_path: str = "config/training_config.yaml") -> Dict[str, TumorDataGenerator]:
    """
    Convenience function to create data generators from configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary of data generators
    """
    from utils.config_loader import load_config
    
    # Load configuration
    config_loader = load_config(config_path)
    config = config_loader.get_config()
    
    # Create data manager
    data_manager = TumorDataManager(config['data']['raw_data_path'], config)
    
    # Discover and split data
    data_files = data_manager.discover_data()
    splits = data_manager.split_data(data_files['images'], data_files['masks'])
    
    # Create generators
    generators = data_manager.create_generators(splits)
    
    return generators


if __name__ == "__main__":
    # Example usage
    print("Creating sample data generator...")
    
    # Create dummy data paths (replace with actual paths)
    dummy_image_paths = ["data/raw/sample1.dcm", "data/raw/sample2.dcm"]
    dummy_mask_paths = ["data/masks/sample1.png", "data/masks/sample2.png"]
    
    # Create generator
    generator = TumorDataGenerator(
        image_paths=dummy_image_paths,
        mask_paths=dummy_mask_paths,
        batch_size=2,
        input_size=(256, 256),
        channels=1
    )
    
    print(f"Generator created with {len(generator)} batches")
    print("Data generator ready for training!") 