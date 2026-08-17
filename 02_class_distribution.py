import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO
from collections import Counter
from pathlib import Path


# ==========================================
# 1. Dataset Configuration
# ==========================================

data_flag = "pneumoniamnist"

info = INFO[data_flag]
DataClass = getattr(medmnist, info["python_class"])


# ==========================================
# 2. Load Training Dataset
# ==========================================

dataset = DataClass(
    split="train",
    download=True,
    size=224
)

print("Dataset loaded successfully!")
print("Total training images:", len(dataset))


# ==========================================
# 3. Extract Labels
# ==========================================

labels = []

for i in range(len(dataset)):
    _, label = dataset[i]

    label_value = int(label[0])
    labels.append(label_value)


# ==========================================
# 4. Count Classes
# ==========================================

class_counts = Counter(labels)

normal_count = class_counts[0]
pneumonia_count = class_counts[1]

print("\nClass Distribution:")
print("Normal:", normal_count)
print("Pneumonia:", pneumonia_count)


# ==========================================
# 5. Calculate Percentages
# ==========================================

total = len(labels)

normal_percentage = (normal_count / total) * 100
pneumonia_percentage = (pneumonia_count / total) * 100

print("\nClass Percentages:")
print(f"Normal: {normal_percentage:.2f}%")
print(f"Pneumonia: {pneumonia_percentage:.2f}%")


# ==========================================
# 6. Create Results Folder
# ==========================================

output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)


# ==========================================
# 7. Create Bar Chart
# ==========================================

classes = ["Normal", "Pneumonia"]
counts = [normal_count, pneumonia_count]

plt.figure(figsize=(8, 6))

plt.bar(classes, counts)

plt.title("PneumoniaMNIST Training Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.tight_layout()


# ==========================================
# 8. Save Figure
# ==========================================

output_path = output_dir / "class_distribution.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print(
    f"\nClass distribution chart saved at: {output_path}"
)