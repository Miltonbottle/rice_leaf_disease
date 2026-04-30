# 🌾 Rice Leaf Disease Classifier

A deep learning model that detects diseases in rice leaves using **MobileNetV4** with transfer learning. Trained on Google Colab with GPU acceleration.



##  Diseases Detected

| Class | Description |
|---|---|
|  Bacterial Leaf Blight | Water-soaked lesions turning yellow/white along leaf edges |
|  Brown Spot | Brown circular spots with yellow halo on leaves |
|  Leaf Smut | Small, slightly raised black spots on leaf surface |
|  Healthy | No disease detected (threshold-based) |

---

## 🏗️ Model Architecture

```
MobileNetV4 (pretrained on ImageNet-1k)
        │
        ├── Backbone (frozen in Phase 1)
        │     └── Last 2 blocks (unfrozen for fine-tuning)
        │
        └── Custom Classifier Head
              Linear(n_features → 512)
              BatchNorm1d + ReLU + Dropout(0.4)
              Linear(512 → 256)
              BatchNorm1d + ReLU + Dropout(0.3)
              Linear(256 → 3 classes)
```

**Base model:** `mobilenetv4_conv_small.e2400_r224_in1k` via [timm](https://github.com/huggingface/pytorch-image-models)

---

## 📊 Results

| Metric | Score |
|---|---|
| Best Validation Accuracy | **88.9%** |
| Training Strategy | 2-Phase Transfer Learning |
| Dataset Size | 120 images (40 per class) |
| Input Resolution | 224 × 224 |

---

## 🚀 Quick Start

### Run in Google Colab (Recommended)

1. Open the notebook in Colab
2. Go to `Runtime → Change runtime type → T4 GPU`
3. Run all cells in order

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/rice-leaf-disease-classifier
cd rice-leaf-disease-classifier

# Install dependencies
pip install torch torchvision timm kagglehub scikit-learn matplotlib seaborn pillow

# Predict on an image
python predict.py
```

---

## 📁 Project Structure

```
rice-leaf-disease-classifier/
│
├── rice_leaf_disease_colab.ipynb  # Main Colab training notebook
├── model_utils.py                 # Model architecture + transforms
├── predict.py                     # Local prediction script
├── test.py                        # Evaluation script
├── rice_model.pth                 # Trained model weights (after training)
└── README.md
```

---

## 🔄 Training Pipeline

### Data Augmentation
Training images are augmented with:
- Random crop, horizontal & vertical flip
- Random rotation (±30°)
- Color jitter (brightness, contrast, saturation, hue)
- Random perspective distortion
- Random erasing (cutout)

### Two-Phase Training Strategy

**Phase 1 — Head + Last 2 Blocks (15 epochs)**
- Backbone frozen, only classifier head and last 2 MobileNetV4 blocks trained
- Learning rate: `1e-3` with Cosine Annealing
- Early stopping: patience of 5 epochs

**Phase 2 — Full Fine-tuning (25 epochs)**
- All layers unfrozen
- Backbone LR: `1e-5` (10× slower to preserve pretrained features)
- Classifier LR: `1e-4`
- Early stopping: patience of 7 epochs

### Class Imbalance Handling
- `WeightedRandomSampler` oversamples minority classes during training
- `CrossEntropyLoss` with per-class weights
- `label_smoothing=0.1` prevents overconfident predictions

---

## 📦 Dependencies

```txt
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
kagglehub
scikit-learn
matplotlib
seaborn
Pillow
numpy
```

Install all at once:
```bash
pip install torch torchvision timm kagglehub scikit-learn matplotlib seaborn pillow numpy
```

---

## 🗃️ Dataset

**Rice Leaf Diseases** by [vbookshelf](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases) on Kaggle

- 120 images total, 40 per class
- 3 disease classes
- Downloaded automatically via `kagglehub` — no manual setup needed

```python
import kagglehub
kagglehub.dataset_download("vbookshelf/rice-leaf-diseases")
```

---

## 🔍 Predicting on New Images

### In Colab
Run Cell 6, upload any rice leaf image and get:
- Predicted disease class
- Confidence score
- Probability bar chart

### Locally with predict.py
```python
from predict import predict_leaf
predict_leaf("path/to/your/leaf.jpg")
```

```
Result for: leaf.jpg
Prediction: Brown spot
Confidence: 91.3%
  Bacterial leaf blight     ██                   8.2%
  Brown spot                ██████████████████   91.3%
  Leaf smut                 █                    0.5%
```

### Healthy Leaf Detection
If the model's top confidence is below **75%**, the leaf is classified as healthy:
```python
HEALTHY_THRESHOLD = 0.75
if confidence < HEALTHY_THRESHOLD:
    result = "✅ Healthy"
```

> **Note:** For more accurate healthy leaf detection, retrain with a 4th class containing healthy rice leaf images.

---

## 📈 Notebook Cells Overview

| Cell | Purpose |
|---|---|
| Cell 1 | Install `timm` and `kagglehub` |
| Cell 2 | Write `model_utils.py` to disk |
| Cell 3 | Download dataset from Kaggle |
| Cell 4 | Train model (2-phase strategy) |
| Cell 5 | Evaluate — classification report + confusion matrix |
| Cell 6 | Upload and predict a single image |
| Cell 7 | Download trained `rice_model.pth` |

---

## ⚙️ Configuration

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Train/Val split | 85% / 15% |
| Phase 1 epochs | 15 (early stop: 5) |
| Phase 2 epochs | 25 (early stop: 7) |
| Phase 1 LR | `1e-3` |
| Phase 2 backbone LR | `1e-5` |
| Phase 2 head LR | `1e-4` |
| Weight decay | `1e-4` |
| Label smoothing | `0.1` |
| Healthy threshold | `0.75` |

---

## 🛠️ Potential Improvements

- Add healthy rice leaf class (retrain with 4 classes for reliable healthy detection)
- Collect more images per class (200+ recommended for 95%+ accuracy)
- Try MobileNetV4 Medium or Large variants for higher accuracy
- Export to ONNX or TFLite for mobile deployment

---

## 📄 License

This project is licensed under the MIT License. The dataset is subject to its own [Kaggle license](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases).

---

## 🙏 Acknowledgements

- [timm](https://github.com/huggingface/pytorch-image-models) by Hugging Face for MobileNetV4
- [vbookshelf](https://www.kaggle.com/vbookshelf) for the Rice Leaf Diseases dataset on Kaggle
- Google Colab for free GPU access
