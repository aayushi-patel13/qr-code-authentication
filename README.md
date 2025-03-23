# qr-code-authentication

## Project Overview

This project tackles the problem of distinguishing original QR codes ("First Print") from reprinted or counterfeit versions ("Second Print"). This type of QR code authentication is vital for building anti-counterfeiting systems in sectors such as packaging, product authentication, and secure document management.

We implement and compare:

- A traditional machine learning pipeline using PCA and SVM.
- A deep learning approach using transfer learning with MobileNetV2.

---

## Dataset Description

The dataset consists of two classes:

- **First Print**: Original QR codes embedded with copy detection patterns.
- **Second Print**: Scanned and reprinted copies of the originals (counterfeits).

Each image shows subtle differences in microstructures, print quality, and CDP degradation.

---

## Project Components

### 1. Data Exploration & Feature Engineering

- Image quality stats: brightness, contrast, entropy
- Edge density, GLCM texture features
- Frequency analysis using FFT
- Local Binary Patterns (LBP)
- Histograms and edge comparisons

### 2. Traditional ML Model

- PCA used for dimensionality reduction
- SVM (RBF kernel) trained on top principal components
- Achieved **95% test accuracy**, confirmed with:
  - Confusion matrix
  - ROC curve (AUC = high)
  - Cross-validation scores

### 3. CNN Model (Transfer Learning)

- Used **MobileNetV2** pre-trained on ImageNet
- Frozen initial layers, fine-tuned deeper layers
- Data augmentation for robustness
- Achieved **100% test accuracy**
- Stress tested with blurred, noisy, and occluded QR codes

---

## Evaluation Metrics

We used the following metrics for performance comparison:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC Score

**CNN Model Confusion Matrix**:

```
[[20  0]
 [ 0 20]]
```

**SVM Model Confusion Matrix**:

```
[[18  3]
 [ 0 19]]
```

---

## How to Run the Code

### 1. Install Requirements

```
pip install -r requirements.txt
```

### 2. Run Models

- Traditional model: `python svm_pca_model.py`
- Deep learning model: `python cnn_model.py`
- EDA notebook: open `eda_analysis.ipynb`

Ensure you place your dataset in the `dataset/` folder:

```
dataset/
├── First Print/
└── Second Print/
```

---

## Deployment Considerations

- MobileNetV2 is lightweight, ideal for edge devices
- Fast inference and robust to occlusion and distortions
- SVM model is simpler and suitable for CPU-only environments

Security implications such as tamper-proof hash binding and end-to-end encryption can further secure this pipeline in real-world systems.

---

## Contributors

- Aayushi Patel [aayushiap8@gmail.com](mailto\:aayushiap8@gmail.com)

---

## License

This project is licensed under the MIT License.

