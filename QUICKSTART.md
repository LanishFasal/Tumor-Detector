# 🚀 Quick Start Guide - Tumor Detection System

Get up and running with the tumor detection system in minutes!

## 📋 Prerequisites

- **Python 3.8+** installed
- **Git** for cloning the repository
- **Medical images** in DICOM, NIfTI, or common formats
- **Ground truth masks** (optional, for training)

## ⚡ Quick Setup (5 minutes)

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd "Tumor Detection"

# Run the setup script
python setup.py
```

### 2. Add Your Data
```bash
# Create data directories (if not already created)
mkdir -p data/raw data/masks

# Copy your medical images to data/raw/
# Copy your ground truth masks to data/masks/
```

### 3. Configure Settings
Edit `config/training_config.yaml` to match your data:
```yaml
data:
  input_size: [256, 256]  # Adjust based on your images
  batch_size: 4           # Reduce if you have limited memory
```

### 4. Train the Model
```bash
# Start training
python src/training/train.py --mode full
```

### 5. Launch Web App
```bash
# Start the interactive web interface
streamlit run web_app/app.py
```

## 🎯 Common Use Cases

### Case 1: I have medical images and want to detect tumors
1. Place your images in `data/raw/`
2. Run: `python src/inference/predict.py --input_dir data/raw --model models/saved_models/best_model.h5`
3. Check results in `results/predictions/`

### Case 2: I want to train on my own dataset
1. Place images in `data/raw/` and masks in `data/masks/`
2. Configure `config/training_config.yaml`
3. Run: `python src/training/train.py --mode full`
4. Monitor training progress and results

### Case 3: I want to use the web interface
1. Train or obtain a model
2. Run: `streamlit run web_app/app.py`
3. Open browser and upload images
4. View results interactively

## 🔧 Configuration Quick Reference

### Key Settings in `config/training_config.yaml`:

```yaml
# Data settings
data:
  input_size: [256, 256]    # Image size for training
  batch_size: 8             # Reduce if out of memory
  channels: 1               # 1 for grayscale, 3 for RGB

# Model settings  
model:
  architecture: "unet"      # Model type
  backbone: "resnet50"      # Encoder backbone

# Training settings
training:
  epochs: 100               # Number of training epochs
  initial_learning_rate: 0.001
```

## 🐛 Troubleshooting

### Common Issues:

**"CUDA out of memory"**
- Reduce `batch_size` in config
- Use smaller `input_size`
- Close other applications using GPU

**"No module named 'tensorflow'"**
- Run: `pip install -r requirements.txt`
- Or: `python setup.py`

**"No images found"**
- Check file paths in `data/raw/`
- Verify supported formats: `.dcm`, `.nii`, `.png`, `.jpg`

**"Model not found"**
- Train a model first: `python src/training/train.py`
- Or download a pre-trained model

### Getting Help:
- Check the full `README.md` for detailed documentation
- Review example notebooks in `notebooks/`
- Check logs in `logs/` directory

## 📊 Expected Results

### Training:
- Model saves to `models/saved_models/`
- Training plots in `results/plots/`
- Logs in `training.log`

### Inference:
- Predictions in `results/predictions/`
- Overlay images showing tumor regions
- Statistics in YAML format

### Web App:
- Interactive image upload and analysis
- Real-time visualization
- Analytics dashboard

## 🎓 Learning Path

1. **Start with data exploration**: `notebooks/01_data_exploration.ipynb`
2. **Train a simple model**: Use default settings
3. **Experiment with parameters**: Modify config file
4. **Analyze results**: Use web app and notebooks
5. **Optimize performance**: Adjust model architecture and training

## 📞 Support

- **Documentation**: `README.md`
- **Examples**: `notebooks/` directory
- **Configuration**: `config/` directory
- **Issues**: Check logs and error messages

---

**Happy tumor detecting! 🏥🔬** 