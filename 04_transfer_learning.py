import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
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

from pathlib import Path


# ==========================================
# 1. Configuration
# ==========================================

DATA_FLAG = "pneumoniamnist"

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
# 4. Load Data
# ==========================================

train_dataset = DataClass(
    split="train",
    transform=transform,
    download=True,
    size=224
)

val_dataset = DataClass(
    split="val",
    transform=transform,
    download=True,
    size=224
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ==========================================
# 5. DataLoaders
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# 6. Load Pretrained ResNet-18
# ==========================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)


# ==========================================
# 7. Freeze Feature Extractor
# ==========================================

for parameter in model.parameters():
    parameter.requires_grad = False


# ==========================================
# 8. Replace Final Layer
# ==========================================

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


model = model.to(DEVICE)

print("ResNet-18 model created successfully!")


# ==========================================
# 9. Loss & Optimizer
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 10. Training
# ==========================================

best_f1 = 0.0


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = (
            labels.squeeze()
            .long()
            .to(DEVICE)
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    average_loss = (
        running_loss /
        len(train_loader)
    )


    # ======================================
    # Validation
    # ======================================

    model.eval()

    predictions = []
    true_labels = []
    probabilities = []


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = (
                labels.squeeze()
                .long()
                .to(DEVICE)
            )

            outputs = model(images)

            probs = torch.softmax(
                outputs,
                dim=1
            )

            preds = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            true_labels.extend(
                labels.cpu().numpy()
            )

            probabilities.extend(
                probs[:, 1].cpu().numpy()
            )


    # ======================================
    # Metrics
    # ======================================

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        true_labels,
        probabilities
    )


    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Loss: {average_loss:.4f}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1 Score: {f1:.4f}"
    )

    print(
        f"ROC-AUC: {auc:.4f}"
    )


    # ======================================
    # Save Best Model
    # ======================================

    if f1 > best_f1:

        best_f1 = f1

        model_dir = Path("models")

        model_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = (
            model_dir /
            "resnet18_transfer_learning.pth"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        print(
            f"Best model saved: {model_path}"
        )


# ==========================================
# 11. Final Confusion Matrix
# ==========================================

cm = confusion_matrix(
    true_labels,
    predictions
)

print("\nFinal Confusion Matrix:")
print(cm)

print(
    f"\nBest Validation F1: {best_f1:.4f}"
)