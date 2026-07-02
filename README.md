# 🌾 Crop Yield Prediction Using 3D-CNN and BiLSTM

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B.svg)


A hybrid deep learning framework for **crop yield prediction** by integrating **Sentinel-2 multispectral satellite imagery** with **environmental and agricultural tabular data**. The proposed architecture combines **3D Convolutional Neural Networks (3D-CNN)** for spatial-spectral feature extraction and **Bidirectional Long Short-Term Memory (BiLSTM)** for temporal learning, followed by feature fusion for accurate yield prediction.

---

# 📖 Overview

Accurate crop yield prediction is essential for precision agriculture, food security, and sustainable farming. Traditional machine learning approaches struggle to capture both spatial information from satellite imagery and temporal crop growth patterns.

This project proposes a **multi-modal hybrid deep learning model** that combines:

- 🌍 Sentinel-2 multispectral satellite imagery
- 🌦 Environmental variables
- 🌱 Agricultural parameters
- 🧠 Deep feature fusion

The framework is accompanied by an interactive **Streamlit web application** for model training, visualization, prediction, and performance analysis.

---

# ✨ Features

- ✅ Hybrid **3D-CNN + BiLSTM** architecture
- ✅ Multi-modal data fusion
- ✅ Sentinel-2 multispectral image processing
- ✅ Environmental feature integration
- ✅ Crop yield prediction
- ✅ Interactive Streamlit dashboard
- ✅ Training visualization
- ✅ Performance evaluation
- ✅ Ablation study
- ✅ Per-crop performance analysis
- ✅ Monte-Carlo Dropout uncertainty estimation

---

# 🏗 Model Architecture

```
                Sentinel-2 Images
                       │
                 Data Preprocessing
                       │
                 3D Convolution
                       │
             Spatial Feature Extraction
                       │
                  BiLSTM Layers
                       │
         Environmental Tabular Features
                       │
                 Feature Fusion
                       │
                  Dense Network
                       │
               Crop Yield Prediction
```

---

# 🚀 Technology Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Deep Learning | PyTorch |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Plotly |
| Web Framework | Streamlit |
| Scientific Computing | SciPy |

---

# 📂 Project Structure

```text
Crop-Yield-Prediction-Using-3D-CNN-and-BiLSTM/
│
├── app.py
├── CropYieldPrediction_3DCNN_BiLSTM.ipynb
├── requirements.txt
├── results/
│   ├── training_history.png
│   ├── prediction_results.png
│   ├── per_crop_performance.png
│   ├── uncertainty_analysis.png
│   └── ...
├── checkpoints/
│
└── README.md
```

---
Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Streamlit Application

```bash
streamlit run app.py
```

The application provides:

- 📊 Project Overview
- 📈 Train & Evaluate
- 🌾 Predict Crop Yield
- 📉 Data Visualization
- 🔬 Ablation Study

---

# 🌾 Prediction Pipeline

```
Input Data
     │
     ▼
Data Preprocessing
     │
     ▼
3D-CNN Feature Extraction
     │
     ▼
BiLSTM Temporal Learning
     │
     ▼
Tabular Feature Processing
     │
     ▼
Feature Fusion
     │
     ▼
Dense Layers
     │
     ▼
Crop Yield Prediction
```

---

# 📊 Performance Evaluation

The proposed model is evaluated using standard regression metrics:

- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Coefficient of Determination (R²)

The hybrid architecture demonstrates improved prediction performance compared to conventional machine learning methods by effectively combining spatial, temporal, and environmental information.

---

# 📈 Streamlit Dashboard

The application includes:

### 🏠 Overview

- Project summary
- Model workflow
- Architecture visualization

---

### 📊 Train & Evaluate

- Model training
- Learning curves
- Validation metrics
- Test evaluation

---

### 🌾 Crop Yield Prediction

Users can configure:

- Crop type
- Rainfall
- Temperature
- Soil pH
- Nitrogen
- Latitude
- Longitude

The application predicts:

- Crop Yield
- Confidence Interval
- Prediction Quality
- NDVI Visualization

---

### 📉 Data Visualization

Visualizations include:

- Spectral signatures
- NDVI phenology
- False-color imagery
- Crop-wise comparison

---

### 🔬 Ablation Study

Performance comparison of different model variants:

- Full Hybrid Model
- 3D-CNN Only
- BiLSTM Only
- Tabular Only

using:

- RMSE
- MAE
- R² Score

---

# 📷 Results

The repository includes result visualizations for:

- Training history
- Prediction vs Ground Truth
- Residual analysis
- Per-crop performance
- NDVI curves
- Ablation study
- Uncertainty estimation
- Learning curves

---
