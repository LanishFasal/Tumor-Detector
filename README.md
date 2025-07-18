# Tumor Detection from MRI/CT Scans

A comprehensive deep learning system for automatic tumor detection and segmentation in medical MRI/CT scans using U-Net architecture.

## 🎯 Project Overview

This project implements an end-to-end tumor detection system that:
- Processes medical images (MRI/CT scans) using PyDicom
- Preprocesses and augments data for better model training
- Uses U-Net architecture for precise tumor segmentation
- Provides a web interface for easy interaction
- Generates detailed analysis reports

## 🚀 Features

- **Medical Image Processing**: Support for DICOM, NIfTI, and common image formats
- **Advanced Preprocessing**: Normalization, resizing, and data augmentation
- **U-Net Segmentation**: State-of-the-art architecture for medical image segmentation
- **Data Augmentation**: Albumentations for robust model training
- **Web Interface**: Flask-based UI for easy interaction
- **Visualization**: Comprehensive plotting and analysis tools
- **Model Evaluation**: Multiple metrics for performance assessment

## 📁 Project Structure

```
Tumor Detection/
├── data/
│   ├── raw/                 # Raw medical images
│   ├── processed/           # Preprocessed images
│   └── masks/              # Ground truth masks
├── models/
│   ├── saved_models/       # Trained model weights
│   └── model_architectures/ # Model definitions
├── src/
│   ├── data_processing/    # Data preprocessing modules
│   ├── models/            # Model architectures
│   ├── training/          # Training pipeline
│   ├── inference/         # Prediction and inference
│   └── visualization/     # Plotting and analysis
├── notebooks/             # Jupyter notebooks for exploration
├── utils/                # Utility functions
└── config/               # Configuration files
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd "Tumor Detection"
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

## 📊 Data Preparation

### Supported Formats
- **DICOM** (.dcm) - Standard medical imaging format
- **NIfTI** (.nii, .nii.gz) - Neuroimaging format
- **Common formats** (.png, .jpg, .tiff) - For processed images

### Data Organization
```
data/
├── raw/
│   ├── patient_001/
│   │   ├── scan_001.dcm
│   │   └── scan_002.dcm
│   └── patient_002/
│       └── scan_001.dcm
├── processed/
│   ├── images/
│   └── masks/
└── splits/
    ├── train/
    ├── val/
    └── test/
```

## 🎯 Usage

### 1. Data Preprocessing
```bash
python src/data_processing/preprocess.py --input_dir data/raw --output_dir data/processed
```

### 2. Model Training
```bash
python src/training/train.py --config config/training_config.yaml
```

### 3. Tumor Detection
```bash
python src/inference/predict.py --model_path models/saved_models/best_model.h5 --image_path data/test/image.dcm
```

## 🧠 Model Architecture

### U-Net Implementation
- **Encoder**: ResNet50 backbone with skip connections
- **Decoder**: Upsampling with concatenation of skip connections
- **Output**: Binary segmentation mask
- **Loss Function**: Dice Loss + Binary Crossentropy
- **Optimizer**: Adam with learning rate scheduling

### Key Features
- **Skip Connections**: Preserve fine-grained details
- **Batch Normalization**: Stable training
- **Dropout**: Prevent overfitting
- **Data Augmentation**: Improve generalization

## 📈 Performance Metrics

- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Jaccard index for segmentation quality
- **Precision/Recall**: Tumor detection accuracy
- **Hausdorff Distance**: Boundary accuracy measurement

## 🔧 Configuration

Edit `config/training_config.yaml` to customize:
- Model parameters
- Training hyperparameters
- Data augmentation settings
- Evaluation metrics

## 🌐 Web Interface

The Streamlit web app provides:
- **Image Upload**: Drag-and-drop medical images
- **Real-time Processing**: Instant tumor detection
- **Visualization**: Overlay results on original images
- **Analysis Reports**: Detailed segmentation metrics
- **Export Options**: Save results in various formats

## 📚 Key Skills Developed

1. **Medical Image Processing**
   - DICOM file handling with PyDicom
   - Image preprocessing and normalization
   - Multi-modal image support

2. **Deep Learning**
   - U-Net architecture implementation
   - Transfer learning with pre-trained models
   - Custom loss functions for medical segmentation

3. **Computer Vision**
   - Image segmentation techniques
   - Data augmentation strategies
   - Performance evaluation metrics

4. **Software Engineering**
   - Modular code architecture
   - Configuration management
   - Web application development

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This project is for educational and research purposes only. It should not be used for actual medical diagnosis without proper validation and regulatory approval.

## 🆘 Support

For questions or issues:
1. Check the documentation in the `docs/` folder
2. Review the example notebooks
3. Open an issue on GitHub

---

**Note**: This system requires significant computational resources for training. Consider using GPU acceleration for optimal performance.

# Tumor Detection Platform

## Deployment (Docker)

### Build the Docker image
```bash
docker build -t tumor-detection-app .
```

### Run the app (production, Gunicorn)
```bash
docker run -p 5000:5000 tumor-detection-app
```

The app will be available at http://localhost:5000

## Deployment (Manual, Gunicorn)

```bash
pip install -r requirements.txt
gunicorn -b 0.0.0.0:5000 backend_server.py
```

## Environment Variables
- `FLASK_SECRET_KEY`: Set a secure secret key for Flask sessions (default is set in backend_server.py, but override in production).
- `SENTRY_DSN`: (Optional) Set your Sentry DSN for error monitoring in production.

## Automated Tests

To run the test suite:
```bash
pytest
```

## Features
- AI-powered tumor detection
- Batch analysis
- PDF reporting
- User authentication
- GDPR/HIPAA compliance
- And more! 