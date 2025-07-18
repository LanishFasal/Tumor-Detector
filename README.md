# 🧠 Tumor Detection System from MRI/CT Scans

An AI-powered medical imaging application using deep learning (U-Net + ResNet50) for tumor detection and segmentation from MRI/CT scans. It features a training pipeline, inference engine, preprocessing tools, and a user-friendly web interface.

## 📌 Key Features

- 🎯 Tumor Segmentation using U-Net with ResNet50 encoder

- 🖼️ Support for DICOM, NIfTI, PNG, JPG images

- ⚙️ Preprocessing + Data Augmentation using Albumentations

- 📈 Metrics & Visualizations for model performance

- 🌐 Flask/Streamlit Web App for real-time tumor detection

- 🔐 User Authentication with bcrypt & JSON user management

- 🐳 Docker + Gunicorn Deployment Ready

- ✅ Automated Installation & Setup (via setup.py)

## ⚙️ Setup Instructions

- ✅ Prerequisites

    Python ≥ 3.8

    pip ≥ 21.0

## 🔧 One-Step Setup

```bash

python setup.py
```

This script will:

- Check Python version & GPU

- Install dependencies

- Create necessary folders

- Test modules & environment


## 🧪 Run the Pipeline

1. Preprocess the Data
   
```bash
python src/data_processing/preprocess.py --input_dir data/raw --output_dir data/processed
```
3. Train the Model
   
```bash

python src/training/train.py --config config/training_config.yaml
```
4. Inference on Test Scan
   
```bash

python src/inference/predict.py --model_path models/saved_models/best_model.h5 --image_path data/raw/test_scan.dcm
```

## 🌐 Web Interface

Use the Streamlit app for interactive tumor detection:

```bash
streamlit run web_app/app.py
```

## 🔐 Authentication

User authentication is handled via users.json:
```python
json
Copy
Edit
{
  "user-id": {
    "username": "your_name",
    "password_hash": "hashed_password"
  }
}
```

## 📊 Evaluation Metrics

- Dice Coefficient (F1 score for segmentation)

- Intersection over Union (IoU)

- Precision, Recall

- Hausdorff Distance

## Screenshot

<img width="1887" height="934" alt="image" src="https://github.com/user-attachments/assets/5029b332-9119-42f5-89cd-f53cdca910ac" />


## 🐳 Docker Deployment

1. Build Docker Image
   
```bash

docker build -t tumor-detection-app .
```

2. Run App (Production with Gunicorn)
   
```bash

docker run -p 5000:5000 tumor-detection-app
```

Or run manually:

```bash

gunicorn -b 0.0.0.0:5000 backend_server.py
```

## 📦 Dependencies

Install them with:

```bash

pip install -r requirements.txt
```
Key packages:

- Flask, Flask-Login, Flask-Bcrypt

- scikit-learn, numpy, matplotlib

- reportlab, gunicorn

- sentry-sdk (optional for monitoring)

## ✅ Automated Testing

```bash

pytest
```
