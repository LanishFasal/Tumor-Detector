"""
Inference module for tumor detection.
Handles prediction on new medical images using trained models.
"""

import os
import sys
import argparse
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.config_loader import load_config
from src.data_processing.medical_image_loader import MedicalImageLoader
from src.models.unet_model import create_unet_model
from src.visualization.training_plots import TrainingVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TumorPredictor:
    """
    Class for making tumor predictions on medical images.
    """
    
    def __init__(self, model_path: str, config_path: str = "config/training_config.yaml"):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
            config_path: Path to configuration file
        """
        self.model_path = Path(model_path)
        self.config_loader = load_config(config_path)
        self.config = self.config_loader.get_config()
        
        # Initialize components
        self.model = None
        self.image_loader = MedicalImageLoader()
        
        # Load model
        self._load_model()
        
        logger.info("Tumor predictor initialized")
    
    def _load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Create model architecture
        input_shape = tuple(self.config['data']['input_size']) + (self.config['data']['channels'],)
        num_classes = self.config['model']['output_channels']
        
        self.model = create_unet_model(
            input_shape=input_shape,
            num_classes=num_classes,
            model_type=self.config['model']['architecture']
        )
        
        # Build model
        self.model.build_model()
        
        # Load weights
        self.model.load_model(str(self.model_path))
        
        logger.info(f"Model loaded from {self.model_path}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for prediction.
        Args:
            image: Input image
        Returns:
            Preprocessed image
        """
        # Get preprocessing parameters from config
        input_size = tuple(self.config['data']['input_size'])
        normalization = self.config['data']['normalization']
        # --- DO NOT convert 3-channel images to grayscale ---
        # Resize to target size
        image = cv2.resize(image, (input_size[1], input_size[0]), 
                          interpolation=cv2.INTER_LINEAR)
        # Normalize image
        image = self.image_loader.normalize_image(image, method=normalization)
        # Ensure image is 3-channel
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)
        elif image.shape[2] == 1:
            image = np.concatenate([image]*3, axis=-1)
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image.astype(np.float32)
    
    def predict(self, image_path: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Make prediction on a single image.
        
        Args:
            image_path: Path to the image file
            threshold: Threshold for binary segmentation
            
        Returns:
            Dictionary with prediction results
        """
        # Load image
        image_data = self.image_loader.load_image(image_path)
        original_image = image_data['image']
        
        # Preprocess image
        processed_image = self.preprocess_image(original_image)
        
        # Make prediction
        prediction = self.model.model.predict(processed_image, verbose=0)
        
        # Post-process prediction
        prediction = prediction[0]  # Remove batch dimension
        binary_prediction = (prediction > threshold).astype(np.float32)
        
        # Calculate tumor area
        tumor_area = np.sum(binary_prediction)
        total_area = binary_prediction.shape[0] * binary_prediction.shape[1]
        tumor_percentage = (tumor_area / total_area) * 100
        
        # Create result dictionary
        result = {
            'image_path': image_path,
            'original_image': original_image,
            'processed_image': processed_image[0],  # Remove batch dimension
            'prediction': prediction,
            'binary_prediction': binary_prediction,
            'tumor_area': tumor_area,
            'total_area': total_area,
            'tumor_percentage': tumor_percentage,
            'metadata': image_data.get('metadata', {}),
            'prediction_time': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction completed for {image_path}")
        logger.info(f"Tumor area: {tumor_area} pixels ({tumor_percentage:.2f}%)")
        
        return result
    
    def predict_batch(self, image_paths: List[str], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Make predictions on multiple images.
        
        Args:
            image_paths: List of image paths
            threshold: Threshold for binary segmentation
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                result = self.predict(image_path, threshold)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'prediction_time': datetime.now().isoformat()
                })
        
        return results
    
    def create_overlay(self, original_image: np.ndarray, prediction: np.ndarray, 
                      alpha: float = 0.6) -> np.ndarray:
        """
        Create overlay of prediction on original image.
        Args:
            original_image: Original image
            prediction: Prediction mask
            alpha: Transparency factor
        Returns:
            Overlay image
        """
        # Ensure images are the same size
        if original_image.shape[:2] != prediction.shape[:2]:
            prediction = cv2.resize(prediction, (original_image.shape[1], original_image.shape[0]))
        # Normalize original image to [0, 1]
        if original_image.max() > 1:
            original_image = original_image.astype(np.float32) / 255.0
        # Create RGB image if grayscale
        if len(original_image.shape) == 2:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = original_image
        # Ensure prediction is 2D
        if prediction.ndim == 3 and prediction.shape[2] == 1:
            prediction = np.squeeze(prediction, axis=-1)
        # Create colored mask
        mask_rgb = np.zeros_like(original_rgb)
        mask_rgb[:, :, 0] = prediction  # Red channel for tumor
        # Create overlay
        overlay = original_rgb * (1 - alpha) + mask_rgb * alpha
        return np.clip(overlay, 0, 1)
    
    def save_prediction_results(self, result: Dict[str, Any], output_dir: Path) -> None:
        """
        Save prediction results and visualizations.
        
        Args:
            result: Prediction result dictionary
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get base filename
        image_path = Path(result['image_path'])
        base_name = image_path.stem
        
        # Save original image
        original_path = output_dir / f"{base_name}_original.png"
        plt.imsave(original_path, result['original_image'], cmap='gray')
        
        # Save prediction mask
        prediction_path = output_dir / f"{base_name}_prediction.png"
        plt.imsave(prediction_path, result['prediction'], cmap='Reds')
        
        # Save binary prediction
        binary_path = output_dir / f"{base_name}_binary.png"
        plt.imsave(binary_path, result['binary_prediction'], cmap='Reds')
        
        # Save overlay
        overlay = self.create_overlay(result['original_image'], result['prediction'])
        overlay_path = output_dir / f"{base_name}_overlay.png"
        plt.imsave(overlay_path, overlay)
        
        # Save metadata
        metadata = {
            'image_path': result['image_path'],
            'tumor_area': int(result['tumor_area']),
            'total_area': int(result['total_area']),
            'tumor_percentage': float(result['tumor_percentage']),
            'prediction_time': result['prediction_time'],
            'metadata': result.get('metadata', {})
        }
        
        import yaml
        metadata_path = output_dir / f"{base_name}_metadata.yaml"
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        logger.info(f"Results saved to {output_dir}")
    
    def create_prediction_report(self, results: List[Dict[str, Any]], 
                               output_dir: Path) -> None:
        """
        Create comprehensive prediction report.
        
        Args:
            results: List of prediction results
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter successful predictions
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            logger.warning("No successful predictions to report")
            return
        
        # Calculate statistics
        tumor_percentages = [r['tumor_percentage'] for r in successful_results]
        tumor_areas = [r['tumor_area'] for r in successful_results]
        
        stats = {
            'total_images': len(results),
            'successful_predictions': len(successful_results),
            'failed_predictions': len(results) - len(successful_results),
            'tumor_percentage_stats': {
                'mean': np.mean(tumor_percentages),
                'std': np.std(tumor_percentages),
                'min': np.min(tumor_percentages),
                'max': np.max(tumor_percentages),
                'median': np.median(tumor_percentages)
            },
            'tumor_area_stats': {
                'mean': np.mean(tumor_areas),
                'std': np.std(tumor_areas),
                'min': np.min(tumor_areas),
                'max': np.max(tumor_areas),
                'median': np.median(tumor_areas)
            }
        }
        
        # Save statistics
        import yaml
        stats_path = output_dir / 'prediction_statistics.yaml'
        with open(stats_path, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False)
        
        # Create visualization
        self._create_prediction_visualization(successful_results, output_dir)
        
        logger.info(f"Prediction report saved to {output_dir}")
    
    def _create_prediction_visualization(self, results: List[Dict[str, Any]], 
                                       output_dir: Path) -> None:
        """
        Create visualization of prediction results.
        
        Args:
            results: List of successful prediction results
            output_dir: Output directory
        """
        # Create figure with subplots
        num_samples = min(6, len(results))
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        fig.suptitle('Tumor Detection Results', fontsize=16, fontweight='bold')
        
        for i in range(num_samples):
            result = results[i]
            
            # Original image
            axes[i, 0].imshow(result['original_image'], cmap='gray')
            axes[i, 0].set_title(f'Sample {i+1} - Original')
            axes[i, 0].axis('off')
            
            # Prediction
            axes[i, 1].imshow(result['prediction'], cmap='Reds', alpha=0.7)
            axes[i, 1].set_title(f'Sample {i+1} - Prediction')
            axes[i, 1].axis('off')
            
            # Overlay
            overlay = self.create_overlay(result['original_image'], result['prediction'])
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f'Sample {i+1} - Overlay\nTumor: {result["tumor_percentage"]:.1f}%')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = output_dir / 'prediction_visualization.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Prediction visualization saved to {viz_path}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Make tumor predictions")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/predictions",
        help="Output directory for results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction threshold"
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = TumorPredictor(args.model, args.config)
    
    # Determine input images
    image_paths = []
    
    if args.image:
        image_paths = [args.image]
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            sys.exit(1)
        
        # Find all supported image files
        supported_formats = ['.dcm', '.nii', '.nii.gz', '.png', '.jpg', '.tiff']
        for format_ext in supported_formats:
            image_paths.extend(list(input_dir.rglob(f"*{format_ext}")))
        
        image_paths = [str(p) for p in image_paths]
    else:
        logger.error("Either --image or --input_dir must be specified")
        sys.exit(1)
    
    if not image_paths:
        logger.error("No images found")
        sys.exit(1)
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Make predictions
    results = predictor.predict_batch(image_paths, args.threshold)
    
    # Save results
    output_dir = Path(args.output_dir)
    for result in results:
        if 'error' not in result:
            predictor.save_prediction_results(result, output_dir)
    
    # Create report
    predictor.create_prediction_report(results, output_dir)
    
    logger.info("Inference completed successfully!")


if __name__ == "__main__":
    main() 