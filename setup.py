#!/usr/bin/env python3
"""
Setup script for Tumor Detection project.
Handles dependency installation and project setup.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🏥 Tumor Detection System                 ║
    ║                                                              ║
    ║  AI-Powered Medical Image Analysis for Tumor Detection      ║
    ║  Built with U-Net, ResNet50, and Deep Learning              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def check_gpu():
    """Check for GPU availability."""
    print("\n🔍 Checking GPU availability...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"   - {gpu.name}")
            return True
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️  TensorFlow not installed yet. GPU check will be done after installation.")
        return False

def install_requirements():
    """Install project requirements."""
    print("\n📦 Installing dependencies...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        # Upgrade pip first
        print("   Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        print("   Installing project dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary project directories."""
    print("\n📁 Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/masks",
        "models/saved_models",
        "models/model_architectures",
        "results",
        "results/plots",
        "results/predictions",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    print("✅ All directories created!")

def download_sample_data():
    """Download sample data if requested."""
    print("\n📥 Sample Data")
    print("   To get started, you'll need medical images and corresponding masks.")
    print("   Place your data in the following directories:")
    print("   - Medical images: data/raw/")
    print("   - Ground truth masks: data/masks/")
    print("   ")
    print("   Supported formats:")
    print("   - Images: DICOM (.dcm), NIfTI (.nii, .nii.gz), PNG, JPG, TIFF")
    print("   - Masks: PNG, JPG (binary masks)")
    print("   ")
    print("   You can find sample datasets from:")
    print("   - Medical Segmentation Decathlon")
    print("   - BraTS Challenge")
    print("   - TCIA (The Cancer Imaging Archive)")

def test_installation():
    """Test the installation."""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        print("   Testing imports...")
        import tensorflow as tf
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt
        import streamlit as st
        import pydicom
        import nibabel as nib
        
        print("   ✅ All core dependencies imported successfully")
        
        # Test GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   ✅ GPU available: {len(gpus)} device(s)")
        else:
            print("   ⚠️  GPU not available (CPU will be used)")
        
        # Test project modules
        print("   Testing project modules...")
        from utils.config_loader import load_config
        from src.data_processing.medical_image_loader import MedicalImageLoader
        from src.models.unet_model import create_unet_model
        
        print("   ✅ Project modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Test error: {e}")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. 📁 Add your medical images to data/raw/")
    print("2. 📁 Add corresponding masks to data/masks/")
    print("3. 🔧 Configure settings in config/training_config.yaml")
    print("4. 🚀 Start training: python src/training/train.py")
    print("5. 🌐 Launch web app: streamlit run web_app/app.py")
    print("\n📚 Documentation:")
    print("- README.md: Project overview and usage")
    print("- notebooks/: Example notebooks for exploration")
    print("- config/: Configuration files")
    print("\n🆘 Need help? Check the README.md file for detailed instructions.")

def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check GPU
    check_gpu()
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed during dependency installation!")
        print("   Please check the error messages above and try again.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Download sample data info
    download_sample_data()
    
    # Test installation
    if not test_installation():
        print("\n❌ Setup failed during testing!")
        print("   Please check the error messages above and try again.")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 