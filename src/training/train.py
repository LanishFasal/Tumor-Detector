"""
Training script for tumor detection model.
Handles model training, validation, and evaluation.
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.config_loader import load_config
from src.models.unet_model import create_unet_model
from src.training.data_generator import TumorDataManager, create_data_generators
from src.visualization.training_plots import TrainingVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TumorDetectionTrainer:
    """
    Trainer class for tumor detection models.
    """
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_loader = load_config(config_path)
        self.config = self.config_loader.get_config()
        
        # Extract configuration
        self.data_config = self.config['data']
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.output_config = self.config['output']
        
        # Initialize components
        self.model = None
        self.data_manager = None
        self.generators = None
        self.callbacks = []
        self.history = None
        
        # Create output directories
        self._create_output_directories()
        
        logger.info("Tumor detection trainer initialized")
    
    def _create_output_directories(self):
        """Create necessary output directories."""
        directories = [
            self.output_config['model_save_path'],
            self.output_config['results_save_path'],
            self.output_config['plots_save_path']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self):
        """Prepare training data and create generators."""
        logger.info("Preparing training data...")
        
        # Create data manager
        self.data_manager = TumorDataManager(
            self.data_config['raw_data_path'],
            self.config
        )
        
        # Discover and split data
        data_files = self.data_manager.discover_data()
        
        if not data_files['images'] or not data_files['masks']:
            raise ValueError("No images or masks found in data directories")
        
        splits = self.data_manager.split_data(
            data_files['images'],
            data_files['masks']
        )
        
        # Create generators
        self.generators = self.data_manager.create_generators(splits)
        
        # Log data information
        data_info = self.data_manager.get_data_info(data_files['images'])
        logger.info(f"Data info: {data_info}")
        
        logger.info("Data preparation completed")
    
    def build_model(self, resume_from_checkpoint=False):
        """Build and compile the model. Optionally resume from checkpoint."""
        logger.info("Building model...")
        # Create model
        self.model = create_unet_model(
            input_shape=tuple(self.data_config['input_size']) + (self.data_config['channels'],),
            num_classes=self.model_config['output_channels'],
            model_type=self.model_config['architecture'],
            filters=self.model_config['unet']['filters'],
            dropout_rate=self.model_config['unet']['dropout_rate'],
            batch_norm=self.model_config['unet']['batch_normalization']
        )
        # Build model
        unet_model = self.model.build_model()
        # Compile model
        self.model.compile_model(
            learning_rate=self.training_config['initial_learning_rate']
        )
        # Resume from checkpoint if requested
        if resume_from_checkpoint:
            checkpoint_path = os.path.join(self.output_config['model_save_path'], 'best_model.h5')
            if os.path.exists(checkpoint_path):
                self.model.model.load_weights(checkpoint_path)
                logger.info(f"Resumed model weights from {checkpoint_path}")
            else:
                logger.warning(f"No checkpoint found at {checkpoint_path}, starting from scratch.")
        # Log model summary
        model_summary = self.model.get_model_summary()
        logger.info(f"Model summary:\n{model_summary}")
        # Save model summary to file
        summary_path = Path(self.output_config['results_save_path']) / 'model_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(model_summary)
        logger.info("Model built and compiled successfully")
    
    def setup_callbacks(self):
        """Setup training callbacks."""
        logger.info("Setting up callbacks...")
        
        # Model checkpoint
        if self.training_config['checkpointing']['enabled']:
            checkpoint_path = Path(self.output_config['model_save_path']) / 'best_model.h5'
            checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=self.training_config['checkpointing']['monitor'],
                mode=self.training_config['checkpointing']['mode'],
                save_best_only=self.training_config['checkpointing']['save_best_only'],
                verbose=1
            )
            self.callbacks.append(checkpoint)
        
        # Early stopping
        if self.training_config['early_stopping']['enabled']:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor=self.training_config['early_stopping']['monitor'],
                patience=self.training_config['early_stopping']['patience'],
                restore_best_weights=self.training_config['early_stopping']['restore_best_weights'],
                verbose=1
            )
            self.callbacks.append(early_stopping)
        
        # Learning rate scheduler
        if self.training_config['learning_rate_schedule'] == 'reduce_on_plateau':
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['lr_scheduler']['factor'],
                patience=self.training_config['lr_scheduler']['patience'],
                min_lr=self.training_config['lr_scheduler']['min_lr'],
                verbose=1
            )
            self.callbacks.append(lr_scheduler)
        
        # TensorBoard logging
        if self.config['logging']['tensorboard']:
            log_dir = Path(self.output_config['results_save_path']) / 'tensorboard_logs'
            tensorboard = keras.callbacks.TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
            self.callbacks.append(tensorboard)
        
        # CSV logger
        csv_logger = keras.callbacks.CSVLogger(
            filename=Path(self.output_config['results_save_path']) / 'training_log.csv',
            separator=',',
            append=False
        )
        self.callbacks.append(csv_logger)
        
        logger.info(f"Setup {len(self.callbacks)} callbacks")
    
    def train(self):
        """Train the model."""
        logger.info("Starting model training...")
        
        # Prepare data if not already done
        if self.generators is None:
            self.prepare_data()
        
        # Build model if not already done
        if self.model is None:
            self.build_model()
        
        # Setup callbacks
        self.setup_callbacks()
        
        # Training parameters
        epochs = self.training_config['epochs']
        train_generator = self.generators['train']
        val_generator = self.generators['validation']
        
        # Start training
        start_time = datetime.now()
        logger.info(f"Training started at {start_time}")
        
        try:
            self.history = self.model.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=self.callbacks,
                verbose=1
            )
            
            end_time = datetime.now()
            training_duration = end_time - start_time
            logger.info(f"Training completed in {training_duration}")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Training failed: {e}")
            traceback.print_exc()
            return False
        
        return True
    
    def evaluate(self):
        """Evaluate the trained model."""
        logger.info("Evaluating model...")
        
        if self.model is None or self.generators is None:
            raise ValueError("Model and data must be prepared before evaluation")
        
        # Evaluate on test set
        test_generator = self.generators['test']
        evaluation_results = self.model.model.evaluate(
            test_generator,
            verbose=1
        )
        
        # Create results dictionary
        metrics = self.model.model.metrics_names
        results = dict(zip(metrics, evaluation_results))
        
        # Log results
        logger.info("Test set evaluation results:")
        for metric, value in results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Save results
        results_path = Path(self.output_config['results_save_path']) / 'evaluation_results.yaml'
        with open(results_path, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        return results
    
    def save_model(self, model_name: str = "final_model"):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_path = Path(self.output_config['model_save_path']) / f"{model_name}.h5"
        self.model.save_model(str(model_path))
        
        logger.info(f"Model saved to {model_path}")
    
    def create_training_report(self):
        """Create comprehensive training report."""
        if self.history is None:
            logger.warning("No training history available for report")
            return
        
        logger.info("Creating training report...")
        
        # Create visualizer
        visualizer = TrainingVisualizer(self.history)
        
        # Generate plots
        plots_dir = Path(self.output_config['plots_save_path'])
        
        # Training curves
        visualizer.plot_training_curves(save_path=plots_dir / 'training_curves.png')
        
        # Metrics comparison
        visualizer.plot_metrics_comparison(save_path=plots_dir / 'metrics_comparison.png')
        
        # Learning rate schedule
        if 'lr' in self.history.history:
            visualizer.plot_learning_rate(save_path=plots_dir / 'learning_rate.png')
        
        logger.info("Training report created successfully")
    
    def run_full_training_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("Starting full training pipeline...")
        
        try:
            # Prepare data
            self.prepare_data()
            
            # Build model
            self.build_model()
            
            # Train model
            success = self.train()
            
            if success:
                # Evaluate model
                evaluation_results = self.evaluate()
                
                # Save final model
                self.save_model()
                
                # Create training report
                self.create_training_report()
                
                logger.info("Full training pipeline completed successfully!")
                return True
            else:
                logger.error("Training failed")
                return False
                
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            traceback.print_exc()
            return False


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train tumor detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "data_only", "model_only", "train_only"],
        default="full",
        help="Training mode"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = TumorDetectionTrainer(args.config)
    
    # Run based on mode
    if args.mode == "full":
        trainer.prepare_data()
        trainer.build_model(resume_from_checkpoint=args.resume)
        success = trainer.train()
        if success:
            evaluation_results = trainer.evaluate()
            trainer.save_model()
            trainer.create_training_report()
    elif args.mode == "data_only":
        trainer.prepare_data()
        success = True
    elif args.mode == "model_only":
        trainer.prepare_data()
        trainer.build_model(resume_from_checkpoint=args.resume)
        success = True
    elif args.mode == "train_only":
        trainer.prepare_data()
        trainer.build_model(resume_from_checkpoint=args.resume)
        success = trainer.train()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        success = False
    
    if success:
        logger.info("Operation completed successfully")
        sys.exit(0)
    else:
        logger.error("Operation failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 