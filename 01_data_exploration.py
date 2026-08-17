import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO
from pathlib import Path


# ==========================================
# 1. Dataset Configuration
# ==========================================

data_flag = "pneumoniamnist"

info = INFO[data_flag]

DataClass = getattr(medmnist, info["python_class"])

print("Dataset:", info["description"])
print("Dataset class:", info["python_class"])


# ==========================================
# 2. Load Training Dataset
# ==========================================

dataset = DataClass(
    split="train",
    download=True,
    size=224
)

print("Dataset loaded successfully!")
print("Number of training images:", len(dataset))


# ==========================================
# 3. Class Names
# ==========================================

class_names = {
    0: "Normal",
    1: "Pneumonia"
}


# ==========================================
# 4. Create Results Folder
# ==========================================

output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)


# ==========================================
# 5. Create Visualization
# ==========================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)


# ==========================================
# 6. Display Sample X-rays
# ==========================================

for i, ax in enumerate(axes.flat):

    image, label = dataset[i]

    # Convert label to integer
    label_value = int(label[0])

    # Display image
    ax.imshow(image, cmap="gray")

    # Display class
    ax.set_title(
        f"Class: {class_names[label_value]}"
    )

    ax.axis("off")


# ==========================================
# 7. Save Figure
# ==========================================

plt.tight_layout()

output_path = output_dir / "sample_xrays.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 8. Confirmation
# ==========================================

print(
    f"Sample X-ray visualization saved successfully at: {output_path}"
)