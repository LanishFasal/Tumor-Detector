"""
U-Net model implementation for medical image segmentation.
Includes ResNet backbone and custom loss functions for tumor detection.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class UNetModel:
    """
    U-Net model for medical image segmentation with ResNet50 backbone.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 1),
                 num_classes: int = 1, filters: List[int] = None,
                 dropout_rate: float = 0.2, batch_norm: bool = True):
        """
        Initialize U-Net model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes (1 for binary segmentation)
            filters: List of filter sizes for each level
            dropout_rate: Dropout rate for regularization
            batch_norm: Whether to use batch normalization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters = filters or [64, 128, 256, 512]
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.model = None
        
    def build_model(self) -> Model:
        """
        Build the U-Net model architecture.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder (ResNet50 backbone)
        encoder_outputs = self._build_encoder(inputs)
        
        # Decoder
        decoder_output = self._build_decoder(encoder_outputs)
        
        # Output layer
        if self.num_classes == 1:
            outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder_output)
        else:
            outputs = layers.Conv2D(self.num_classes, (1, 1), activation='softmax')(decoder_output)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        logger.info(f"U-Net model built successfully with input shape: {self.input_shape}")
        return self.model
    
    def _build_encoder(self, inputs: layers.Input) -> List[layers.Layer]:
        """
        Build the encoder part using ResNet50 backbone.
        
        Args:
            inputs: Input layer
            
        Returns:
            List of encoder outputs for skip connections
        """
        # Load ResNet50 without top layers
        resnet = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling=None
        )
        
        # Get intermediate outputs for skip connections
        encoder_outputs = []
        
        # Extract features from different ResNet layers
        layer_names = [
            'conv1_relu',      # 64 filters
            'conv2_block3_out', # 256 filters
            'conv3_block4_out', # 512 filters
            'conv4_block6_out'  # 1024 filters
        ]
        
        for layer_name in layer_names:
            layer = resnet.get_layer(layer_name)
            encoder_outputs.append(layer.output)
        
        logger.info(f"Encoder built with {len(encoder_outputs)} skip connection levels")
        return encoder_outputs
    
    def _build_decoder(self, encoder_outputs: List[layers.Layer]) -> layers.Layer:
        """
        Build the decoder part with skip connections.
        
        Args:
            encoder_outputs: List of encoder outputs for skip connections
            
        Returns:
            Final decoder output
        """
        # Start with the deepest encoder output
        x = encoder_outputs[-1]
        
        # Decoder blocks (reverse order of encoder outputs)
        for i in range(len(encoder_outputs) - 1, 0, -1):
            # Upsample
            x = self._upsample_block(x, self.filters[i-1])
            
            # Concatenate with skip connection
            skip_connection = encoder_outputs[i-1]
            
            # Ensure spatial dimensions match
            if x.shape[1:3] != skip_connection.shape[1:3]:
                x = layers.Resizing(
                    skip_connection.shape[1],
                    skip_connection.shape[2]
                )(x)
            
            x = layers.Concatenate()([x, skip_connection])
            
            # Apply convolution blocks
            x = self._conv_block(x, self.filters[i-1])
        
        # Final upsampling to original size
        if x.shape[1:3] != self.input_shape[:2]:
            x = layers.Resizing(self.input_shape[0], self.input_shape[1])(x)
        
        logger.info(f"Decoder built successfully")
        return x
    
    def _upsample_block(self, x: layers.Layer, filters: int) -> layers.Layer:
        """
        Create an upsampling block.
        
        Args:
            x: Input tensor
            filters: Number of filters
            
        Returns:
            Upsampled tensor
        """
        x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        
        if self.batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        if self.dropout_rate > 0:
            x = layers.Dropout(self.dropout_rate)(x)
        
        return x
    
    def _conv_block(self, x: layers.Layer, filters: int) -> layers.Layer:
        """
        Create a convolution block.
        
        Args:
            x: Input tensor
            filters: Number of filters
            
        Returns:
            Convolved tensor
        """
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        if self.batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        if self.batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        if self.dropout_rate > 0:
            x = layers.Dropout(self.dropout_rate)(x)
        
        return x
    
    def compile_model(self, learning_rate: float = 0.001,
                     loss_weights: Optional[Dict[str, float]] = None) -> None:
        """
        Compile the model with custom loss functions and metrics.
        
        Args:
            learning_rate: Learning rate for optimizer
            loss_weights: Weights for different loss components
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        # Custom loss function
        loss_fn = self._combined_loss()
        
        # Optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Metrics
        metrics = [
            self._dice_coefficient,
            self._iou_metric,
            'binary_accuracy' if self.num_classes == 1 else 'categorical_accuracy'
        ]
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )
        
        logger.info("Model compiled successfully")
    
    def _combined_loss(self):
        """
        Create combined loss function (Dice Loss + Binary Crossentropy).
        
        Returns:
            Combined loss function
        """
        def combined_loss(y_true, y_pred):
            # Dice Loss
            dice_loss = 1 - self._dice_coefficient(y_true, y_pred)
            
            # Binary Crossentropy
            bce_loss = keras.losses.binary_crossentropy(y_true, y_pred)
            
            # Combine losses (equal weight)
            return 0.5 * dice_loss + 0.5 * bce_loss
        
        return combined_loss
    
    def _dice_coefficient(self, y_true, y_pred, smooth: float = 1e-6):
        """
        Calculate Dice coefficient.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Dice coefficient
        """
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + 
                                             tf.keras.backend.sum(y_pred_f) + smooth)
    
    def _iou_metric(self, y_true, y_pred, smooth: float = 1e-6):
        """
        Calculate Intersection over Union (IoU).
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            smooth: Smoothing factor
            
        Returns:
            IoU value
        """
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
        
        return (intersection + smooth) / (union + smooth)
    
    def get_model_summary(self) -> str:
        """
        Get model summary as string.
        
        Returns:
            Model summary string
        """
        if self.model is None:
            raise ValueError("Model must be built before getting summary")
        
        # Capture model summary
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be built before saving")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load the model from file.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(
            filepath,
            custom_objects={
                'dice_coefficient': self._dice_coefficient,
                'iou_metric': self._iou_metric,
                'combined_loss': self._combined_loss()
            }
        )
        logger.info(f"Model loaded from {filepath}")


class AttentionUNet(UNetModel):
    """
    Attention U-Net variant with attention gates for better feature selection.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 1),
                 num_classes: int = 1, filters: List[int] = None,
                 dropout_rate: float = 0.2, batch_norm: bool = True):
        """
        Initialize Attention U-Net model.
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
            filters: List of filter sizes
            dropout_rate: Dropout rate
            batch_norm: Whether to use batch normalization
        """
        super().__init__(input_shape, num_classes, filters, dropout_rate, batch_norm)
    
    def _attention_gate(self, x: layers.Layer, g: layers.Layer, filters: int) -> layers.Layer:
        """
        Create attention gate for feature selection.
        
        Args:
            x: Input feature map
            g: Gating signal
            filters: Number of filters
            
        Returns:
            Attention-weighted feature map
        """
        # Transform gating signal
        g_conv = layers.Conv2D(filters, (1, 1), padding='same')(g)
        g_bn = layers.BatchNormalization()(g_conv)
        g_relu = layers.ReLU()(g_bn)
        
        # Transform input feature map
        x_conv = layers.Conv2D(filters, (1, 1), padding='same')(x)
        x_bn = layers.BatchNormalization()(x_conv)
        x_relu = layers.ReLU()(x_bn)
        
        # Add transformed features
        combined = layers.Add()([g_relu, x_relu])
        combined_relu = layers.ReLU()(combined)
        
        # Generate attention weights
        attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(combined_relu)
        
        # Apply attention to input
        attended = layers.Multiply()([x, attention])
        
        return attended
    
    def _build_decoder(self, encoder_outputs: List[layers.Layer]) -> layers.Layer:
        """
        Build decoder with attention gates.
        
        Args:
            encoder_outputs: List of encoder outputs
            
        Returns:
            Final decoder output
        """
        x = encoder_outputs[-1]
        
        for i in range(len(encoder_outputs) - 1, 0, -1):
            # Upsample
            x = self._upsample_block(x, self.filters[i-1])
            
            # Apply attention gate
            skip_connection = encoder_outputs[i-1]
            attended_skip = self._attention_gate(skip_connection, x, self.filters[i-1])
            
            # Ensure spatial dimensions match
            if x.shape[1:3] != attended_skip.shape[1:3]:
                x = layers.Resizing(
                    attended_skip.shape[1],
                    attended_skip.shape[2]
                )(x)
            
            # Concatenate
            x = layers.Concatenate()([x, attended_skip])
            
            # Apply convolution blocks
            x = self._conv_block(x, self.filters[i-1])
        
        # Final upsampling
        if x.shape[1:3] != self.input_shape[:2]:
            x = layers.Resizing(self.input_shape[0], self.input_shape[1])(x)
        
        return x


def create_unet_model(input_shape: Tuple[int, int, int] = (256, 256, 1),
                     num_classes: int = 1, model_type: str = 'unet',
                     **kwargs) -> UNetModel:
    """
    Factory function to create U-Net models.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        model_type: Type of model ('unet' or 'attention_unet')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized U-Net model
    """
    if model_type == 'unet':
        return UNetModel(input_shape, num_classes, **kwargs)
    elif model_type == 'attention_unet':
        return AttentionUNet(input_shape, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    print("Creating U-Net model...")
    
    # Create model
    model = create_unet_model(
        input_shape=(256, 256, 1),
        num_classes=1,
        model_type='unet'
    )
    
    # Build model
    unet_model = model.build_model()
    
    # Compile model
    model.compile_model(learning_rate=0.001)
    
    # Print summary
    print("\nModel Summary:")
    print(model.get_model_summary())
    
    print("\nU-Net model created successfully!") 