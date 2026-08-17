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
    page_title="Explainable Medical Imaging",
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
# HEADER
# =========================================================

st.title("🔬 Explainable Medical Imaging")

st.markdown(
    """
    ### AI-Powered Chest X-ray Analysis

    Upload a chest X-ray image to obtain a pneumonia prediction
    and visualize the regions that influenced the model's decision
    using **Grad-CAM Explainable AI**.
    """
)

st.caption(
    "🧠 ResNet-18  •  🔍 Grad-CAM  •  🩻 Chest X-ray Analysis"
)

st.divider()


# =========================================================
# DISCLAIMER
# =========================================================

st.warning(
    """
    ⚠️ **Research & Educational Prototype**

    This application is intended for research and educational
    purposes only. It is not a medical device and should not
    be used for clinical diagnosis or medical decision-making.
    """
)


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = load_model()

    st.success(
        f"✓ ResNet-18 model loaded successfully | Device: {DEVICE}"
    )

except Exception as e:

    st.error(
        f"Unable to load model: {e}"
    )

    st.stop()


# =========================================================
# UPLOAD SECTION
# =========================================================

st.header("📤 Upload Chest X-ray")

st.write(
    "Upload a chest X-ray image in PNG, JPG, or JPEG format."
)

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# =========================================================
# ANALYSIS
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # LOAD IMAGE
    # -----------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("L")

    image = image.resize(
        (224, 224)
    )


    # -----------------------------------------------------
    # IMAGE + PREDICTION COLUMNS
    # -----------------------------------------------------

    col1, col2 = st.columns(
        [1, 1],
        gap="large"
    )


    # -----------------------------------------------------
    # ORIGINAL IMAGE
    # -----------------------------------------------------

    with col1:

        st.subheader("🩻 Original X-ray")

        st.image(
            image,
            use_container_width=True
        )


    # -----------------------------------------------------
    # PREPARE TENSOR
    # -----------------------------------------------------

    image_tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(DEVICE)


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # CLASS NAMES
    # -----------------------------------------------------

    class_names = [
        "Normal",
        "Pneumonia"
    ]

    prediction = class_names[
        predicted_class
    ]


    # -----------------------------------------------------
    # PROBABILITIES
    # -----------------------------------------------------

    normal_probability = probabilities[
        0,
        0
    ].item()

    pneumonia_probability = probabilities[
        0,
        1
    ].item()


    # -----------------------------------------------------
    # PREDICTION RESULT
    # -----------------------------------------------------

    with col2:

        st.subheader("🤖 Model Prediction")

        if prediction == "Pneumonia":

            st.error(
                f"⚠️ Prediction: {prediction}"
            )

        else:

            st.success(
                f"✓ Prediction: {prediction}"
            )

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

        st.divider()

        st.write("### Class Probabilities")

        st.write(
            f"**Normal:** {normal_probability:.2%}"
        )

        st.progress(
            normal_probability
        )

        st.write(
            f"**Pneumonia:** {pneumonia_probability:.2%}"
        )

        st.progress(
            pneumonia_probability
        )


    # =====================================================
    # GRAD-CAM
    # =====================================================

    st.divider()

    st.header("🔍 Grad-CAM Explainability")

    st.write(
        """
        Grad-CAM highlights the regions of the chest X-ray
        that contributed to the model's prediction.

        Warmer regions indicate areas receiving stronger
        attention from the neural network.
        """
    )


    # -----------------------------------------------------
    # GENERATE GRAD-CAM
    # -----------------------------------------------------

    with st.spinner("Generating Grad-CAM explanation..."):

        cam = generate_gradcam(
            model,
            image_tensor,
            predicted_class
        )


    # -----------------------------------------------------
    # ORIGINAL IMAGE
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # HEATMAP
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # OVERLAY
    # -----------------------------------------------------

    overlay = cv2.addWeighted(
        original_rgb,
        0.6,
        heatmap,
        0.4,
        0
    )


    # -----------------------------------------------------
    # DISPLAY GRAD-CAM
    # -----------------------------------------------------

    gradcam_col1, gradcam_col2 = st.columns(
        2,
        gap="large"
    )


    with gradcam_col1:

        st.subheader("🌡️ Grad-CAM Heatmap")

        st.image(
            heatmap,
            use_container_width=True
        )

        st.caption(
            "Model attention visualization"
        )


    with gradcam_col2:

        st.subheader("🩻 Grad-CAM Overlay")

        st.image(
            overlay,
            use_container_width=True
        )

        st.caption(
            "Heatmap overlaid on the original X-ray"
        )


    # =====================================================
    # INTERPRETATION NOTE
    # =====================================================

    st.info(
        """
        **Interpretation Note:** Grad-CAM is an explainability
        technique used to visualize model attention. The highlighted
        regions should not be interpreted as a clinically validated
        diagnosis or exact disease localization.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🔬 Explainable Medical Imaging | "
    "ResNet-18 Transfer Learning + Grad-CAM | "
    "Research & Educational Project"
)
