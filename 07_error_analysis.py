import torch
import torch.nn as nn
from torchvision import transforms, models

import medmnist
from medmnist import INFO

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# 1. Configuration
# ==========================================

DATA_FLAG = "pneumoniamnist"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = Path(
    "models/resnet18_transfer_learning.pth"
)

OUTPUT_DIR = Path(
    "results/figures/error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("Using device:", DEVICE)


# ==========================================
# 2. Dataset
# ==========================================

info = INFO[DATA_FLAG]

DataClass = getattr(
    medmnist,
    info["python_class"]
)


# ==========================================
# 3. Transform
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Lambda(
        lambda x: x.repeat(3, 1, 1)
    ),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 4. Load Test Dataset
# ==========================================

test_dataset = DataClass(
    split="test",
    transform=transform,
    download=True,
    size=224
)

print("Test samples:", len(test_dataset))


# ==========================================
# 5. Load ResNet-18
# ==========================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully!")


# ==========================================
# 6. Prediction
# ==========================================

true_labels = []
predicted_labels = []
probabilities = []

false_positive_indices = []
false_negative_indices = []


for index in range(len(test_dataset)):

    image, label = test_dataset[index]

    true_label = int(label[0])

    input_tensor = image.unsqueeze(
        0
    ).to(DEVICE)


    with torch.no_grad():

        output = model(
            input_tensor
        )

        probs = torch.softmax(
            output,
            dim=1
        )

        prediction = torch.argmax(
            output,
            dim=1
        ).item()

        pneumonia_probability = (
            probs[0, 1].item()
        )


    true_labels.append(
        true_label
    )

    predicted_labels.append(
        prediction
    )

    probabilities.append(
        pneumonia_probability
    )


    # ======================================
    # False Positive
    # Normal → Pneumonia
    # ======================================

    if (
        true_label == 0
        and prediction == 1
    ):

        false_positive_indices.append(
            index
        )


    # ======================================
    # False Negative
    # Pneumonia → Normal
    # ======================================

    if (
        true_label == 1
        and prediction == 0
    ):

        false_negative_indices.append(
            index
        )


# ==========================================
# 7. Metrics
# ==========================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

auc = roc_auc_score(
    true_labels,
    probabilities
)


# ==========================================
# 8. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)


print("\n==========================================")
print("TEST SET RESULTS")
print("==========================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {auc:.4f}"
)


print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 9. Error Counts
# ==========================================

print("\n==========================================")
print("ERROR ANALYSIS")
print("==========================================")

print(
    "False Positives:",
    len(false_positive_indices)
)

print(
    "False Negatives:",
    len(false_negative_indices)
)


# ==========================================
# 10. Save Confusion Matrix
# ==========================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "ResNet-18 Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Normal", "Pneumonia"]
)

plt.yticks(
    [0, 1],
    ["Normal", "Pneumonia"]
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


confusion_path = (
    OUTPUT_DIR /
    "resnet18_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 11. Save Error Indices
# ==========================================

np.save(
    OUTPUT_DIR /
    "false_positive_indices.npy",
    np.array(false_positive_indices)
)

np.save(
    OUTPUT_DIR /
    "false_negative_indices.npy",
    np.array(false_negative_indices)
)


# ==========================================
# 12. Save Error Summary
# ==========================================

summary_path = (
    OUTPUT_DIR /
    "error_summary.txt"
)

with open(
    summary_path,
    "w"
) as file:

    file.write(
        "ResNet-18 Error Analysis\n"
    )

    file.write(
        "========================\n\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1 Score: {f1:.4f}\n"
    )

    file.write(
        f"ROC-AUC: {auc:.4f}\n\n"
    )

    file.write(
        f"False Positives: "
        f"{len(false_positive_indices)}\n"
    )

    file.write(
        f"False Negatives: "
        f"{len(false_negative_indices)}\n"
    )


# ==========================================
# 13. Final Output
# ==========================================

print(
    "\nConfusion matrix saved at:",
    confusion_path
)

print(
    "Error summary saved at:",
    summary_path
)

print(
    "\nError analysis completed successfully!"
)