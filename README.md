# Explainable Medical Imaging for Pneumonia Detection from Chest X-Rays

## Abstract

Pneumonia is a common respiratory disease that can be identified through chest X-ray imaging. Deep learning models have demonstrated strong potential for automated medical image classification; however, their predictions can be difficult to interpret. This project investigates pneumonia classification from chest X-ray images using a baseline convolutional neural network (CNN) and a pretrained ResNet-18 transfer-learning model. The study also incorporates Grad-CAM-based explainability and systematic error analysis to investigate the visual regions associated with model predictions.

The PneumoniaMNIST dataset was used for the experiments. The training set contained 4,708 images, with 1,214 normal cases and 3,494 pneumonia cases, demonstrating a substantial class imbalance. The baseline CNN achieved a validation accuracy of 96.95% and an F1-score of 97.95%. The ResNet-18 transfer-learning model achieved a validation accuracy of 90.27%, F1-score of 93.66%, and ROC-AUC of 97.33%. On the unseen test set, ResNet-18 achieved 76.92% accuracy, 73.84% precision, 97.69% recall, 84.11% F1-score, and 93.36% ROC-AUC.

Error analysis identified 135 false-positive predictions and 9 false-negative predictions on the test set. Grad-CAM visualizations were generated for representative predictions and for false-positive and false-negative cases to investigate model attention. The findings demonstrate the importance of evaluating both predictive performance and model interpretability when developing explainable medical imaging systems.

> Note: This project is a research and educational prototype and is not intended for clinical diagnosis or medical decision-making.

---

# 1. Introduction

Medical imaging plays an important role in the detection and assessment of diseases. Chest X-rays are widely used for examining pulmonary abnormalities, including findings associated with pneumonia.

Recent advances in deep learning have enabled automated systems to learn visual patterns directly from medical images. Convolutional neural networks can extract hierarchical image features and use them for disease classification. However, high predictive performance alone does not guarantee that a model is making decisions for meaningful reasons.

A major challenge with deep learning in medical imaging is interpretability. A model may provide a prediction such as "pneumonia" without clearly explaining which regions of the image influenced that prediction.

Explainable Artificial Intelligence (XAI) techniques can help address this problem. In this project, Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize regions that contribute to model predictions.

The project therefore combines three components:

1. Pneumonia classification from chest X-rays.
2. Comparison of a baseline CNN with transfer learning using ResNet-18.
3. Explainability and error analysis using Grad-CAM.

---

# 2. Research Problem

Deep learning models can achieve strong performance in medical image classification, but their predictions may be difficult to interpret.

The central problem investigated in this project is:

> How effectively can deep learning models classify pneumonia from chest X-ray images, and can explainable AI techniques provide meaningful insight into the visual regions influencing model predictions?

---

# 3. Research Questions

### RQ1

How accurately can a CNN trained from scratch classify pneumonia from chest X-ray images?

### RQ2

How does a pretrained ResNet-18 transfer-learning model compare with the baseline CNN?

### RQ3

What types of prediction errors occur on unseen test data?

### RQ4

Can Grad-CAM be used to visualize the regions influencing model predictions?

### RQ5

What visual attention patterns are observed in false-positive and false-negative predictions?

---

# 4. Dataset

The project uses the PneumoniaMNIST dataset from the MedMNIST collection.

The dataset contains grayscale chest X-ray images for binary classification:

- Class 0: Normal
- Class 1: Pneumonia

The training split used in the project contained 4,708 images.

The class distribution was:

| Class | Images | Percentage |
|---|---:|---:|
| Normal | 1,214 | 25.79% |
| Pneumonia | 3,494 | 74.21% |
| Total | 4,708 | 100% |

The dataset is therefore imbalanced toward the pneumonia class.

Images were loaded at 224 × 224 resolution for the deep learning experiments.

---

# 5. Methodology

The experimental pipeline consisted of the following stages:

```text
PneumoniaMNIST Dataset
        |
        v
Data Exploration
        |
        v
Class Distribution Analysis
        |
        +----------------------+
        |                      |
        v                      v
Baseline CNN             ResNet-18
from Scratch             Transfer Learning
        |                      |
        +----------+-----------+
                   |
                   v
            Model Evaluation
                   |
                   v
             Error Analysis
                   |
                   v
               Grad-CAM
                   |
                   v
          Explainable Analysis