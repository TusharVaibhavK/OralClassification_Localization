# Oral Cancer Classification 🦷 🔬

An AI-powered system for classifying oral lesions as benign or malignant using deep learning models.

## 🧠 Models Overview

This project implements four powerful deep learning models for oral cancer classification:

### 1. UNet Model 🌐

The UNet architecture, originally designed for biomedical image segmentation, has been adapted for classification tasks:

- **Architecture**: Encoder-decoder structure with skip connections
- **Key Features**:
  - Preserves spatial information through skip connections
  - Efficient feature extraction with downsampling/upsampling paths
  - Global average pooling followed by classification head
- **Implementation**: Unet.py
- **Demo Application**: app.py

### 2. AttentionNet 👁️

An enhanced classification model with attention mechanisms:

- **Architecture**: Based on ResNet50 with added attention mechanisms
- **Key Features**:
  - Channel attention modules for focusing on important feature channels
  - Spatial attention for highlighting relevant regions in the image
  - Visualization of attention maps to show which areas influenced the model's decision
- **Implementation**: model.py
- **Demo Application**: app1.py
- **Training Script**: train.py

### 3. DeepLab Model 🖼️

A semantic segmentation model adapted for classification tasks:

- **Architecture**: DeepLabV3+ with atrous convolution for capturing multi-scale context
- **Key Features**:
  - Atrous Spatial Pyramid Pooling (ASPP) for extracting features at multiple scales
  - Fine-tuned for classification tasks
  - Robust to variations in lesion size and shape
- **Implementation**: deeplab.py
- **Demo Application**: deeplab_app.py
- **Training Script**: deeplab_train.py

### 4. Transformer Model 🔄

A cutting-edge model leveraging transformer-based architectures:

- **Architecture**: Vision Transformer (ViT) adapted for image classification
- **Key Features**:
  - Self-attention mechanism for capturing global dependencies
  - Patch-based image representation
  - High accuracy on complex datasets
- **Implementation**: transformer.py
- **Demo Application**: transformer_app.py
- **Training Script**: transformer_train.py

## 🚀 Project Setup

### Prerequisites

- Python 3.8+ 🐍
- PyTorch 1.8+ 🔥
- CUDA-capable GPU (recommended) ⚡

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TusharVaibhavK/OralCancerClassification.git
   cd OralCancerClassification
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install torch torchvision tqdm matplotlib scikit-learn seaborn streamlit pillow
   ```

## 💻 Usage

### Training Models

#### Train AttentionNet:
```bash
python -m AttentionNet.train
```

#### Train UNet:
```bash
python -m Unet.Unet
```

#### Train DeepLabV3+:
```bash
python -m DeepLabV3+.deeplab
```

#### Train Transformer:
```bash
python -m TransformerSegamentationNetwork.train_full_pipeline
```

### Running Applications

#### AttentionNet Web Interface:
```bash
streamlit run AttentionNet/app1.py
```

#### UNet Web Interface:
```bash
streamlit run Unet/app.py
```

#### DeepLabV3+ Web Interface:
```bash
streamlit run DeepLabV3+/main.py
```

#### Transformer-segmentation Network Web Interface:
```bash
streamlit run TransformerSegmentationNetwork/classification_ui.py
```

### Visualizing Attention Maps

```bash
python -m AttentionNet.visualize_attention --image_dir your_image_directory --num_images 5
```

## 📂 Project Structure

```
├── AttentionNet/               # AttentionNet implementation
│   ├── results/                # Results and evaluation metrics
│   │   ├── confusion_matrix.png
│   │   └── training_history.png
│   ├── app1.py                 # Streamlit web application
│   ├── model.py                # AttentionNet model architecture
│   ├── train.py                # Training script
│   └── visualize_attention.py  # Script for visualizing attention maps
├── DeepLabV3+/                 # DeepLab implementation
│   ├── results/                # Results and evaluation metrics
│   │   ├── deep_model.pth
│   │   ├── roc_curve.png
│   ├── deeplab.py              # DeepLab model architecture
│   ├── main.py                 # Streamlit web application
│   └── model_analysis.py       # Script for analyzing model performance
├── Oral Images Dataset/        # Dataset folder (placeholder for images)
├── TransformerSegmentationNetwork/  # Transformer implementation
│   ├── __pycache__/            # Python cache files
│   ├── results/                # Results and evaluation metrics
│   │   ├── best_model.pth
│   │   ├── confusion_matrix.png
│   │   ├── oral_cancer_model.pth
│   │   ├── oral_cancer_resnet50.pth
│   │   └── roc_curve.png
│   ├── classification_ui.py    # Streamlit web application
│   ├── curve.py                # Script for plotting curves
│   └── train_full_pipeline.py  # Training script for the full pipeline
├── Unet/                       # UNet implementation
│   ├── results/                # Results and evaluation metrics
│   │   ├── model.pth
│   │   ├── results.png
│   │   └── roc_curve.png
│   ├── app.py                  # Streamlit web application
│   ├── model_analysis.py       # Script for analyzing model performance
│   └── Unet.py                 # UNet model architecture and training
├── .gitignore                  # Git ignore file
├── model.pth                   # Placeholder for model weights
└── README.md                   # Project documentation

```

## 🔍 Results

Models are evaluated based on:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Visualizations of attention maps help explain which regions of the image contributed most to the classification decision.
