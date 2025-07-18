"""
Training visualization module for tumor detection.
Handles plotting training curves, metrics, and model evaluation results.
"""

import seaborn as sns
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """
    Visualization class for training results and model evaluation.
    """
    
    def __init__(self, history):
        """
        Initialize the visualizer with training history.
        
        Args:
            history: Keras training history object
        """
        self.history = history
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Setup matplotlib and seaborn plotting styles."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Set figure size and DPI
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 12
    
    def plot_training_curves(self, save_path: Optional[Path] = None) -> None:
        """
        Plot training and validation curves.
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
        
        # Loss curves
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in self.history.history:
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        if 'binary_accuracy' in self.history.history:
            axes[0, 1].plot(self.history.history['binary_accuracy'], label='Training Accuracy', linewidth=2)
            if 'val_binary_accuracy' in self.history.history:
                axes[0, 1].plot(self.history.history['val_binary_accuracy'], label='Validation Accuracy', linewidth=2)
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Dice coefficient curves
        if 'dice_coefficient' in self.history.history:
            axes[1, 0].plot(self.history.history['dice_coefficient'], label='Training Dice', linewidth=2)
            if 'val_dice_coefficient' in self.history.history:
                axes[1, 0].plot(self.history.history['val_dice_coefficient'], label='Validation Dice', linewidth=2)
            axes[1, 0].set_title('Dice Coefficient')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Dice Coefficient')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # IoU curves
        if 'iou_metric' in self.history.history:
            axes[1, 1].plot(self.history.history['iou_metric'], label='Training IoU', linewidth=2)
            if 'val_iou_metric' in self.history.history:
                axes[1, 1].plot(self.history.history['val_iou_metric'], label='Validation IoU', linewidth=2)
            axes[1, 1].set_title('Intersection over Union (IoU)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('IoU')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, save_path: Optional[Path] = None) -> None:
        """
        Plot comparison of different metrics.
        
        Args:
            save_path: Path to save the plot
        """
        # Get final training and validation metrics
        metrics_data = {}
        
        for key in self.history.history.keys():
            if not key.startswith('val_'):
                train_metric = self.history.history[key][-1]
                val_key = f'val_{key}'
                val_metric = self.history.history[val_key][-1] if val_key in self.history.history else None
                
                metrics_data[key] = {
                    'Training': train_metric,
                    'Validation': val_metric
                }
        
        if not metrics_data:
            logger.warning("No metrics found for comparison")
            return
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = list(metrics_data.keys())
        x = np.arange(len(metrics))
        width = 0.35
        
        train_values = [metrics_data[m]['Training'] for m in metrics]
        val_values = [metrics_data[m]['Validation'] for m in metrics if metrics_data[m]['Validation'] is not None]
        
        bars1 = ax.bar(x - width/2, train_values, width, label='Training', alpha=0.8)
        
        if val_values:
            bars2 = ax.bar(x + width/2, val_values, width, label='Validation', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Final Training vs Validation Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        if val_values:
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to {save_path}")
        
        plt.show()
    
    def plot_learning_rate(self, save_path: Optional[Path] = None) -> None:
        """
        Plot learning rate schedule.
        
        Args:
            save_path: Path to save the plot
        """
        if 'lr' not in self.history.history:
            logger.warning("Learning rate history not available")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.history.history['lr'], linewidth=2, color='red')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning rate plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: Optional[Path] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        from sklearn.metrics import confusion_matrix
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
        
        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Background', 'Tumor'],
                   yticklabels=['Background', 'Tumor'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_samples(self, images: np.ndarray, masks: np.ndarray, 
                              predictions: np.ndarray, num_samples: int = 6,
                              save_path: Optional[Path] = None) -> None:
        """
        Plot sample predictions with ground truth.
        
        Args:
            images: Input images
            masks: Ground truth masks
            predictions: Model predictions
            num_samples: Number of samples to plot
            save_path: Path to save the plot
        """
        num_samples = min(num_samples, len(images))
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        fig.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
        
        for i in range(num_samples):
            # Original image
            axes[i, 0].imshow(images[i].squeeze(), cmap='gray')
            axes[i, 0].set_title(f'Sample {i+1} - Original Image')
            axes[i, 0].axis('off')
            
            # Ground truth mask
            axes[i, 1].imshow(masks[i].squeeze(), cmap='Reds', alpha=0.7)
            axes[i, 1].set_title(f'Sample {i+1} - Ground Truth')
            axes[i, 1].axis('off')
            
            # Prediction
            axes[i, 2].imshow(predictions[i].squeeze(), cmap='Reds', alpha=0.7)
            axes[i, 2].set_title(f'Sample {i+1} - Prediction')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction samples saved to {save_path}")
        
        plt.show()
    
    def create_training_summary(self, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Create a comprehensive training summary.
        
        Args:
            save_path: Path to save the summary
            
        Returns:
            Dictionary with training summary
        """
        summary = {
            'total_epochs': len(self.history.history['loss']),
            'final_metrics': {},
            'best_metrics': {},
            'training_time': None  # Would need to be calculated separately
        }
        
        # Final metrics
        for key in self.history.history.keys():
            if not key.startswith('val_'):
                summary['final_metrics'][key] = self.history.history[key][-1]
                
                # Best metrics
                if key in ['dice_coefficient', 'iou_metric', 'binary_accuracy']:
                    summary['best_metrics'][key] = max(self.history.history[key])
                else:
                    summary['best_metrics'][key] = min(self.history.history[key])
        
        # Validation metrics
        for key in self.history.history.keys():
            if key.startswith('val_'):
                original_key = key[4:]  # Remove 'val_' prefix
                summary['final_metrics'][key] = self.history.history[key][-1]
                
                # Best validation metrics
                if original_key in ['dice_coefficient', 'iou_metric', 'binary_accuracy']:
                    summary['best_metrics'][key] = max(self.history.history[key])
                else:
                    summary['best_metrics'][key] = min(self.history.history[key])
        
        # Save summary
        if save_path:
            import yaml
            with open(save_path, 'w') as f:
                yaml.dump(summary, f, default_flow_style=False)
            logger.info(f"Training summary saved to {save_path}")
        
        return summary


class ModelEvaluator:
    """
    Class for evaluating model performance and creating evaluation plots.
    """
    
    def __init__(self, model, test_generator):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained model
            test_generator: Test data generator
        """
        self.model = model
        self.test_generator = test_generator
        self.predictions = None
        self.true_labels = None
    
    def generate_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions on test set.
        
        Returns:
            Tuple of (predictions, true_labels)
        """
        logger.info("Generating predictions on test set...")
        
        predictions = []
        true_labels = []
        
        for i in range(len(self.test_generator)):
            batch_images, batch_masks = self.test_generator[i]
            batch_predictions = self.model.predict(batch_images, verbose=0)
            
            predictions.extend(batch_predictions)
            true_labels.extend(batch_masks)
        
        self.predictions = np.array(predictions)
        self.true_labels = np.array(true_labels)
        
        logger.info(f"Generated predictions for {len(self.predictions)} samples")
        
        return self.predictions, self.true_labels
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Returns:
            Dictionary with metric values
        """
        if self.predictions is None or self.true_labels is None:
            raise ValueError("Predictions must be generated first")
        
        from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
        
        # Binarize predictions
        pred_binary = (self.predictions > 0.5).astype(np.float32)
        true_binary = (self.true_labels > 0.5).astype(np.float32)
        
        # Flatten arrays
        pred_flat = pred_binary.flatten()
        true_flat = true_binary.flatten()
        
        # Calculate metrics
        metrics = {
            'precision': precision_score(true_flat, pred_flat, zero_division=0),
            'recall': recall_score(true_flat, pred_flat, zero_division=0),
            'f1_score': f1_score(true_flat, pred_flat, zero_division=0),
            'jaccard_score': jaccard_score(true_flat, pred_flat, zero_division=0),
            'dice_coefficient': self._calculate_dice_coefficient(pred_binary, true_binary)
        }
        
        return metrics
    
    def _calculate_dice_coefficient(self, pred: np.ndarray, true: np.ndarray) -> float:
        """
        Calculate Dice coefficient.
        
        Args:
            pred: Predictions
            true: Ground truth
            
        Returns:
            Dice coefficient
        """
        intersection = np.sum(pred * true)
        union = np.sum(pred) + np.sum(true)
        
        if union == 0:
            return 0.0
        
        return (2.0 * intersection) / union
    
    def create_evaluation_report(self, save_dir: Path) -> None:
        """
        Create comprehensive evaluation report.
        
        Args:
            save_dir: Directory to save evaluation results
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate predictions
        self.generate_predictions()
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Save metrics
        import yaml
        with open(save_dir / 'evaluation_metrics.yaml', 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
        
        # Create visualizer
        visualizer = TrainingVisualizer(None)  # No history needed for evaluation
        
        # Plot confusion matrix
        pred_binary = (self.predictions > 0.5).astype(np.float32)
        true_binary = (self.true_labels > 0.5).astype(np.float32)
        visualizer.plot_confusion_matrix(
            true_binary, pred_binary,
            save_path=save_dir / 'confusion_matrix.png'
        )
        
        # Plot sample predictions
        num_samples = min(6, len(self.predictions))
        sample_indices = np.random.choice(len(self.predictions), num_samples, replace=False)
        
        sample_images = []
        sample_masks = []
        sample_predictions = []
        
        for i in range(len(self.test_generator)):
            if len(sample_images) >= num_samples:
                break
            batch_images, batch_masks = self.test_generator[i]
            batch_predictions = self.model.predict(batch_images, verbose=0)
            
            for j in range(len(batch_images)):
                if len(sample_images) < num_samples:
                    sample_images.append(batch_images[j])
                    sample_masks.append(batch_masks[j])
                    sample_predictions.append(batch_predictions[j])
        
        visualizer.plot_prediction_samples(
            np.array(sample_images),
            np.array(sample_masks),
            np.array(sample_predictions),
            save_path=save_dir / 'sample_predictions.png'
        )
        
        logger.info(f"Evaluation report saved to {save_dir}")


if __name__ == "__main__":
    # Example usage
    print("Training visualizer module loaded successfully!")
    print("Use this module to create training plots and evaluation reports.") 