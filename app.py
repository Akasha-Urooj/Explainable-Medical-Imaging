import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Explainable Pneumonia Detection",
  page_icon="🔬",
    layout="wide"
)


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = Path(
    "models/resnet18_transfer_learning.pth"
)


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():

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

    return model


# =========================================================
# IMAGE TRANSFORMATION
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# GRAD-CAM
# =========================================================

def generate_gradcam(
    model,
    image_tensor,
    target_class
):

    activations = []
    gradients = []

    target_layer = model.layer4[-1].conv2

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    model.zero_grad()

    output = model(
        image_tensor
    )

    score = output[
        0,
        target_class
    ]

    score.backward()

    activation = activations[0][0]

    gradient = gradients[0][0]

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

    if cam.max() > 0:
        cam /= cam.max()

    cam = cam.detach().cpu().numpy()

    cam = cv2.resize(
        cam,
        (224, 224)
    )

    forward_handle.remove()
    backward_handle.remove()

    return cam


# =========================================================
# TITLE
# =========================================================

st.title("Explainable Pneumonia Detection")

st.markdown(
    """
### Deep Learning + Explainable AI

Upload a chest X-ray image to obtain a model prediction
and visualize the regions that influenced the prediction
using **Grad-CAM**.
"""
)


# =========================================================
# DISCLAIMER
# =========================================================

st.warning(
    """
⚠️ **Research/Educational Prototype**

This application is not a medical device and should not
be used for clinical diagnosis or medical decision-making.
"""
)


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = load_model()

    st.success(
        f"ResNet-18 model loaded successfully | Device: {DEVICE}"
    )

except Exception as e:

    st.error(
        f"Unable to load model: {e}"
    )

    st.stop()


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("L")

    image = image.resize(
        (224, 224)
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Original X-ray"
        )

        st.image(
            image,
            use_container_width=True
        )


    # ---------------------------------------------
    # Prepare tensor
    # ---------------------------------------------

    image_tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(DEVICE)


    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------

    model.zero_grad()

    output = model(
        image_tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted_class = torch.argmax(
        output,
        dim=1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()


    class_names = [
        "Normal",
        "Pneumonia"
    ]

    prediction = class_names[
        predicted_class
    ]


    # ---------------------------------------------
    # Result
    # ---------------------------------------------

    with col2:

        st.subheader(
            "Prediction"
        )

        if prediction == "Pneumonia":

            st.error(
                f"Prediction: {prediction}"
            )

        else:

            st.success(
                f"Prediction: {prediction}"
            )

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )


    # =================================================
    # CLASS PROBABILITIES
    # =================================================

    st.subheader(
        "Class Probabilities"
    )

    normal_probability = probabilities[
        0,
        0
    ].item()

    pneumonia_probability = probabilities[
        0,
        1
    ].item()


    probability_col1, probability_col2 = st.columns(2)

    with probability_col1:

        st.metric(
            "Normal",
            f"{normal_probability:.2%}"
        )

    with probability_col2:

        st.metric(
            "Pneumonia",
            f"{pneumonia_probability:.2%}"
        )


    # =================================================
    # GRAD-CAM
    # =================================================

    st.subheader(
        "🔍 Grad-CAM Explanation"
    )

    st.write(
        """
Grad-CAM highlights image regions that contributed
to the model's prediction. The visualization is intended
for model interpretation and does not represent a
clinically validated diagnostic explanation.
"""
    )


    cam = generate_gradcam(
        model,
        image_tensor,
        predicted_class
    )


    # ---------------------------------------------
    # Original image
    # ---------------------------------------------

    original = np.array(
        image
    )

    original = cv2.resize(
        original,
        (224, 224)
    )

    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_GRAY2RGB
    )


    # ---------------------------------------------
    # Heatmap
    # ---------------------------------------------

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


    # ---------------------------------------------
    # Overlay
    # ---------------------------------------------

    overlay = cv2.addWeighted(
        original_rgb,
        0.6,
        heatmap,
        0.4,
        0
    )


    gradcam_col1, gradcam_col2 = st.columns(2)

    with gradcam_col1:

        st.image(
            heatmap,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

    with gradcam_col2:

        st.image(
            overlay,
            caption="Grad-CAM Overlay",
            use_container_width=True
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Explainable Medical Imaging Research Project | "
    "ResNet-18 + Grad-CAM"
)