# **CIFAR-10 Image Classification with CNN**

---

## **1. Project Overview**

### **Description**

This project implements a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset**. It demonstrates a complete **end-to-end machine learning pipeline**, including data preprocessing, model design, training, evaluation, and visualization.

### **Objective**

The CIFAR-10 dataset is a standard benchmark in computer vision. This project aims to:

* Build a **robust CNN classifier** from scratch.
* Achieve **competitive accuracy** on a widely recognized benchmark.
* Implement **comprehensive evaluation metrics and visualizations**.
* Demonstrate **best practices** in ML project structure and reproducibility.

### **Key Features**

* **Custom CNN Architecture** built with TensorFlow/Keras.
* **Comprehensive Evaluation** using F1-score, Precision, Recall, and ROC-AUC.
* **Professional Visualizations**: Confusion matrix, learning curves, and sample predictions.
* **Robust Training Pipeline** with early stopping and model checkpointing.
* **Reproducible Results** using fixed random seeds and version pinning.
* **Production-Ready Code** with modular structure.

---

## **2. Dataset**

**CIFAR-10 Dataset**

* **Source**: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
* **Description**: 60,000 images of 10 classes (32×32 pixels, RGB).
* **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
* **Split**: 50,000 training images and 10,000 test images.
* **Format**: RGB images (32×32×3).

### **Data Preprocessing**

* **Normalization**: Pixel values scaled to `[0, 1]` (`/255.0`).
* **Label Reshaping**: Converted 2D labels to 1D.
* **Train-Validation Split**: 90% training / 10% validation.
* **Data Augmentation (Optional)**:

  * Random horizontal flips.
  * Random rotations (±15°).
  * Width/height shifts (10%).
  * Zoom variations (10%).

---

## **3. Model Architecture**

### **CNN Architecture Overview**

```
Input (32×32×3)
    ↓
Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.3)
    ↓
Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.3)
    ↓
Conv2D(128, 3×3) → BatchNorm → ReLU → Dropout(0.4)
    ↓
Flatten → Dense(128) → ReLU → Dropout(0.5)
    ↓
Dense(64) → ReLU → Dropout(0.5)
    ↓
Dense(10) → Softmax
```

### **Layer Summary**

| Layer Type   | Output Shape | Parameters | Activation |
| ------------ | ------------ | ---------- | ---------- |
| Conv2D       | (30, 30, 32) | 896        | ReLU       |
| MaxPooling2D | (15, 15, 32) | 0          | -          |
| Conv2D       | (13, 13, 64) | 18,496     | ReLU       |
| MaxPooling2D | (6, 6, 64)   | 0          | -          |
| Conv2D       | (4, 4, 128)  | 73,856     | ReLU       |
| Dense        | (128,)       | 262,272    | ReLU       |
| Dense        | (64,)        | 8,256      | ReLU       |
| Dense        | (10,)        | 650        | Softmax    |

**Total Parameters**: \~364,426

---

## **4. Training Setup**

### **Configuration**

* **Loss Function**: Sparse Categorical Cross-entropy.
* **Optimizer**: Adam (default learning rate = 0.001).
* **Metrics**: Accuracy, Precision, Recall, F1-score.
* **Batch Size**: 64.
* **Max Epochs**: 50.
* **Validation Split**: 10%.

### **Callbacks**

* **Early Stopping**: `patience=5` (monitors validation accuracy).
* **Model Checkpoint**: Saves best weights.
* **Restore Best Weights**: Ensures optimal final model.

---

## **5. Evaluation Results**

**Test Accuracy**: **0.7599**
**Training Time**: 1,228 seconds (\~20 minutes).
**Model Parameters**: 365,322

**Classification Report:**

```
              precision    recall  f1-score   support
    airplane       0.73      0.81      0.77      1000
  automobile       0.83      0.92      0.87      1000
        bird       0.63      0.65      0.64      1000
         cat       0.60      0.55      0.57      1000
        deer       0.76      0.72      0.74      1000
         dog       0.71      0.62      0.66      1000
        frog       0.81      0.86      0.83      1000
       horse       0.84      0.79      0.82      1000
        ship       0.79      0.90      0.84      1000
       truck       0.90      0.78      0.84      1000
    accuracy                           0.76     10000
```

**Macro AUC**: **0.9674**

[**Link to plots**](https://github.com/ArkChaudhary/Image-Classifier/tree/main/plots) (learning curves, confusion matrix, per-class metrics).

---

## **6. Technical Stack**

### **Core Libraries**

| Library      | Version  | Purpose                 |
| ------------ | -------- | ----------------------- |
| TensorFlow   | 2.10.1   | Deep learning framework |
| Keras        | Included | High-level neural API   |
| NumPy        | 1.21+    | Numerical computations  |
| Pandas       | 1.3+     | Data manipulation       |
| Matplotlib   | 3.5+     | Visualization           |
| Seaborn      | 0.11+    | Statistical plots       |
| scikit-learn | 1.0+     | Metrics & utilities     |

---

## **7. Reproducibility**

### **Setup Instructions**

```bash
git clone https://github.com/ArkChaudhary/Image-Classifier.git
cd Image-Classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Run Training**

```bash
python main.py  # Basic training
python main.py --epochs 50 --batch-size 64  # Custom training
```

**Reproducibility Measures**:

* Fixed seeds: `np.random.seed(42)` and `tf.random.set_seed(42)`.
* Deterministic operations.
* Version pinning in `requirements.txt`.

---

## **8. Results**

* **Performance Analysis**: Achieved \~76% accuracy and high AUC (\~0.96), which is strong for a custom CNN without transfer learning.
* **Challenges**: Model struggles with certain classes like *cat* and *bird* due to inter-class similarities.
* **Future Improvements**:

  * Add transfer learning with pretrained models (e.g., ResNet50).
  * More aggressive data augmentation.
  * Hyperparameter tuning with learning rate scheduling.

---

## **9. Project Structure**

```
Image-Classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── train.py
│   └── main.py
├── models/
│   ├── best_model.h5
│   └── final_model.h5
├── plots/
│   ├── learning_curves.png
│   ├── confusion_matrix.png
│   ├── per_class_metrics.png
│   └── sample_predictions.png
├── reports/
│   ├── classification_report.txt
│   ├── per_class_metrics.csv
│   └── roc_auc_scores.csv
```
