"""
Medical image loader for handling DICOM, NIfTI, and common image formats.
Supports various medical imaging modalities used in tumor detection.
"""

import os
import numpy as np
import cv2
from typing import Union, Tuple, Optional, Dict, Any
import logging
from pathlib import Path

# Medical imaging libraries
try:
    import pydicom
    from pydicom.dataset import FileDataset
except ImportError:
    pydicom = None
    logging.warning("PyDicom not available. DICOM support disabled.")

try:
    import nibabel as nib
except ImportError:
    nib = None
    logging.warning("Nibabel not available. NIfTI support disabled.")

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None
    logging.warning("SimpleITK not available. Some medical formats may not be supported.")

from PIL import Image

logger = logging.getLogger(__name__)


class MedicalImageLoader:
    """
    Medical image loader for various formats including DICOM, NIfTI, and common image formats.
    """
    
    def __init__(self):
        """Initialize the medical image loader."""
        self.supported_formats = {
            '.dcm': self._load_dicom,
            '.nii': self._load_nifti,
            '.nii.gz': self._load_nifti,
            '.png': self._load_common_image,
            '.jpg': self._load_common_image,
            '.jpeg': self._load_common_image,
            '.tiff': self._load_common_image,
            '.tif': self._load_common_image,
            '.bmp': self._load_common_image
        }
    
    def load_image(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a medical image from file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary containing:
                - 'image': numpy array of the image
                - 'metadata': dictionary of image metadata
                - 'format': detected format
                - 'shape': image shape
                - 'dtype': image data type
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Get file extension
        file_extension = file_path.suffix.lower()
        
        # Handle compressed NIfTI files
        if file_extension == '.gz' and file_path.stem.endswith('.nii'):
            file_extension = '.nii.gz'
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        try:
            # Load the image using the appropriate method
            loader_method = self.supported_formats[file_extension]
            result = loader_method(file_path)
            
            # Add common metadata
            result['format'] = file_extension
            result['file_path'] = str(file_path)
            result['shape'] = result['image'].shape
            result['dtype'] = result['image'].dtype
            
            logger.info(f"Successfully loaded {file_path} (shape: {result['shape']})")
            return result
            
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            raise
    
    def _load_dicom(self, file_path: Path) -> Dict[str, Any]:
        """
        Load DICOM image and extract metadata.
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            Dictionary with image data and metadata
        """
        if pydicom is None:
            raise ImportError("PyDicom is required for DICOM support")
        
        try:
            # Load DICOM file
            dcm = pydicom.dcmread(str(file_path))
            
            # Extract image data
            image = dcm.pixel_array.astype(np.float32)
            
            # Extract metadata
            metadata = {
                'patient_name': getattr(dcm, 'PatientName', 'Unknown'),
                'patient_id': getattr(dcm, 'PatientID', 'Unknown'),
                'study_date': getattr(dcm, 'StudyDate', 'Unknown'),
                'modality': getattr(dcm, 'Modality', 'Unknown'),
                'body_part': getattr(dcm, 'BodyPartExamined', 'Unknown'),
                'window_center': getattr(dcm, 'WindowCenter', None),
                'window_width': getattr(dcm, 'WindowWidth', None),
                'slice_thickness': getattr(dcm, 'SliceThickness', None),
                'pixel_spacing': getattr(dcm, 'PixelSpacing', None),
                'rows': getattr(dcm, 'Rows', image.shape[0]),
                'columns': getattr(dcm, 'Columns', image.shape[1]),
                'bits_allocated': getattr(dcm, 'BitsAllocated', 16),
                'samples_per_pixel': getattr(dcm, 'SamplesPerPixel', 1)
            }
            
            return {
                'image': image,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading DICOM file {file_path}: {e}")
            raise
    
    def _load_nifti(self, file_path: Path) -> Dict[str, Any]:
        """
        Load NIfTI image and extract metadata.
        
        Args:
            file_path: Path to NIfTI file
            
        Returns:
            Dictionary with image data and metadata
        """
        if nib is None:
            raise ImportError("Nibabel is required for NIfTI support")
        
        try:
            # Load NIfTI file
            nii_img = nib.load(str(file_path))
            
            # Extract image data
            image = nii_img.get_fdata().astype(np.float32)
            
            # Extract metadata
            header = nii_img.header
            metadata = {
                'data_type': str(header.get_data_dtype()),
                'dimensions': header.get_data_shape(),
                'voxel_size': header.get_zooms(),
                'affine': header.get_best_affine(),
                'units': header.get_xyzt_units(),
                'description': header.get('descrip', b'').decode('utf-8', errors='ignore'),
                'intent': header.get('intent_name', b'').decode('utf-8', errors='ignore')
            }
            
            return {
                'image': image,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading NIfTI file {file_path}: {e}")
            raise
    
    def _load_common_image(self, file_path: Path) -> Dict[str, Any]:
        """
        Load common image formats (PNG, JPG, TIFF, etc.).
        Args:
            file_path: Path to image file
        Returns:
            Dictionary with image data and metadata
        """
        try:
            # Load image using PIL
            pil_image = Image.open(file_path)
            # Convert to RGB (3 channels) regardless of original mode
            pil_image = pil_image.convert('RGB')
            # Convert to numpy array
            image = np.array(pil_image).astype(np.float32)
            # Extract metadata
            metadata = {
                'format': pil_image.format,
                'mode': pil_image.mode,
                'size': pil_image.size,
                'info': pil_image.info
            }
            return {
                'image': image,
                'metadata': metadata
            }
        except Exception as e:
            logger.error(f"Error loading image file {file_path}: {e}")
            raise
    
    def normalize_image(self, image: np.ndarray, method: str = 'min_max') -> np.ndarray:
        """
        Normalize image using specified method.
        
        Args:
            image: Input image array
            method: Normalization method ('min_max', 'z_score', 'histogram_equalization')
            
        Returns:
            Normalized image array
        """
        if method == 'min_max':
            # Min-max normalization to [0, 1]
            img_min = np.min(image)
            img_max = np.max(image)
            if img_max > img_min:
                return (image - img_min) / (img_max - img_min)
            else:
                return image
        
        elif method == 'z_score':
            # Z-score normalization
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                return (image - mean) / std
            else:
                return image - mean
        
        elif method == 'histogram_equalization':
            # Histogram equalization
            if len(image.shape) == 2:
                return cv2.equalizeHist(image.astype(np.uint8)).astype(np.float32)
            else:
                return image
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def apply_window_level(self, image: np.ndarray, window_center: float, 
                          window_width: float) -> np.ndarray:
        """
        Apply window/level adjustment (commonly used in medical imaging).
        
        Args:
            image: Input image array
            window_center: Window center value
            window_width: Window width value
            
        Returns:
            Windowed image array
        """
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        
        # Clip values to window range
        windowed = np.clip(image, window_min, window_max)
        
        # Normalize to [0, 1]
        return (windowed - window_min) / (window_max - window_min)
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                    method: str = 'bilinear') -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image array
            target_size: Target size (height, width)
            method: Interpolation method ('bilinear', 'bicubic', 'nearest')
            
        Returns:
            Resized image array
        """
        if method == 'bilinear':
            interpolation = cv2.INTER_LINEAR
        elif method == 'bicubic':
            interpolation = cv2.INTER_CUBIC
        elif method == 'nearest':
            interpolation = cv2.INTER_NEAREST
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        return cv2.resize(image, (target_size[1], target_size[0]), 
                        interpolation=interpolation)
    
    def get_image_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get basic information about an image file without loading the full image.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary with image information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        info = {
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'extension': file_path.suffix.lower()
        }
        
        # Try to get format-specific information
        try:
            if info['extension'] == '.dcm' and pydicom:
                dcm = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                info.update({
                    'modality': getattr(dcm, 'Modality', 'Unknown'),
                    'patient_name': getattr(dcm, 'PatientName', 'Unknown'),
                    'study_date': getattr(dcm, 'StudyDate', 'Unknown')
                })
            elif info['extension'] in ['.nii', '.nii.gz'] and nib:
                nii_img = nib.load(str(file_path))
                info.update({
                    'dimensions': nii_img.header.get_data_shape(),
                    'data_type': str(nii_img.header.get_data_dtype())
                })
            else:
                # For common image formats
                with Image.open(file_path) as img:
                    info.update({
                        'size': img.size,
                        'mode': img.mode,
                        'format': img.format
                    })
        except Exception as e:
            logger.warning(f"Could not extract detailed info for {file_path}: {e}")
        
        return info


# Convenience function
def load_medical_image(file_path: Union[str, Path], normalize: bool = True, 
                      target_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """
    Convenience function to load and optionally preprocess a medical image.
    
    Args:
        file_path: Path to the image file
        normalize: Whether to normalize the image
        target_size: Target size for resizing (height, width)
        
    Returns:
        Dictionary with processed image data and metadata
    """
    loader = MedicalImageLoader()
    result = loader.load_image(file_path)
    
    if normalize:
        result['image'] = loader.normalize_image(result['image'])
    
    if target_size is not None:
        result['image'] = loader.resize_image(result['image'], target_size)
    
    return result


if __name__ == "__main__":
    # Example usage
    loader = MedicalImageLoader()
    
    # Test with a sample image (you would need to provide actual file paths)
    print("Medical Image Loader initialized successfully!")
    print(f"Supported formats: {list(loader.supported_formats.keys())}") 