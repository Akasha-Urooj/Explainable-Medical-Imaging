import torch
import torch.nn as nn
from torchvision import transforms, models

import medmnist
from medmnist import INFO

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from PIL import Image
import cv2


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
    "results/figures/gradcam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("Using device:", DEVICE)


# ==========================================
# 2. Load Dataset
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

print("ResNet-18 model loaded successfully!")


# ==========================================
# 6. Grad-CAM Hook
# ==========================================

activations = None
gradients = None


def forward_hook(
    module,
    input,
    output
):
    global activations

    activations = output


def backward_hook(
    module,
    grad_input,
    grad_output
):
    global gradients

    gradients = grad_output[0]


# Last convolutional layer of ResNet-18
target_layer = model.layer4[-1].conv2


target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


# ==========================================
# 7. Generate Grad-CAM
# ==========================================

def generate_gradcam(
    image,
    target_class
):

    global activations
    global gradients

    model.zero_grad()

    output = model(
        image
    )

    score = output[
        0,
        target_class
    ]

    score.backward()

    activation = activations[
        0
    ]

    gradient = gradients[
        0
    ]

    # Global average pooling
    weights = gradient.mean(
        dim=(1, 2)
    )

    cam = torch.zeros(
        activation.shape[1:],
        device=DEVICE
    )

    for i, weight in enumerate(weights):

        cam += (
            weight *
            activation[i]
        )

    cam = torch.relu(cam)

    cam -= cam.min()

    if cam.max() != 0:

        cam /= cam.max()

    cam = cam.detach().cpu().numpy()

    cam = cv2.resize(
        cam,
        (224, 224)
    )

    return cam


# ==========================================
# 8. Select Test Images
# ==========================================

sample_indices = [
    0,
    1,
    2,
    3
]


# ==========================================
# 9. Process Images
# ==========================================

for index in sample_indices:

    image, label = test_dataset[index]

    true_label = int(
        label[0]
    )

    input_tensor = image.unsqueeze(
        0
    ).to(DEVICE)

    # Prediction
    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()


    # Grad-CAM
    cam = generate_gradcam(
        input_tensor,
        predicted_class
    )


    # ======================================
    # Prepare Original Image
    # ======================================

    original_image = image[0].cpu().numpy()

    original_image = (
        original_image -
        original_image.min()
    )

    original_image /= (
        original_image.max() + 1e-8
    )


    # ======================================
    # Create Heatmap
    # ======================================

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    original_uint8 = np.uint8(
        255 * original_image
    )

    original_rgb = cv2.cvtColor(
        original_uint8,
        cv2.COLOR_GRAY2RGB
    )


    # ======================================
    # Overlay
    # ======================================

    overlay = cv2.addWeighted(
        original_rgb,
        0.6,
        heatmap,
        0.4,
        0
    )


    # ======================================
    # Labels
    # ======================================

    class_names = {
        0: "Normal",
        1: "Pneumonia"
    }

    true_name = class_names[
        true_label
    ]

    predicted_name = class_names[
        predicted_class
    ]


    # ======================================
    # Visualization
    # ======================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        original_image,
        cmap="gray"
    )

    axes[0].set_title(
        f"Original\nTrue: {true_name}"
    )

    axes[0].axis("off")


    axes[1].imshow(
        cam,
        cmap="jet"
    )

    axes[1].set_title(
        "Grad-CAM"
    )

    axes[1].axis("off")


    axes[2].imshow(
        overlay
    )

    axes[2].set_title(
        f"Prediction: {predicted_name}\n"
        f"Confidence: {confidence:.2%}"
    )

    axes[2].axis("off")


    plt.tight_layout()


    # ======================================
    # Save
    # ======================================

    output_path = (
        OUTPUT_DIR /
        f"gradcam_sample_{index}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


print("\nGrad-CAM analysis completed successfully!")