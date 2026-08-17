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
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS — UI POLISH
# =========================================================

st.markdown("""
<style>

    /* ---------- Global ---------- */

    .stApp {
        background: #f7f9fc;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }


    /* ---------- Header ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #172554 55%,
            #0f766e 100%
        );

        padding: 38px 42px;
        border-radius: 22px;
        margin-bottom: 25px;
        color: white;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 17px;
        opacity: 0.9;
        line-height: 1.6;
        max-width: 850px;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 18px;
        padding: 7px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        font-size: 13px;
    }


    /* ---------- Section titles ---------- */

    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: #0f172a;
        margin-top: 25px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 18px;
    }


    /* ---------- Upload area ---------- */

    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #cbd5e1;
        border-radius: 18px;
        padding: 12px;
        transition: 0.2s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #0f766e;
        background: #f8fafc;
    }


    /* ---------- Cards ---------- */

    .info-card {
        background: white;
        border-radius: 18px;
        padding: 22px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
        margin-bottom: 18px;
    }

    .card-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .card-value {
        color: #0f172a;
        font-size: 27px;
        font-weight: 800;
        margin-top: 6px;
    }


    /* ---------- Prediction ---------- */

    .prediction-normal {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        font-size: 26px;
        font-weight: 800;
    }

    .prediction-pneumonia {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        color: #9f1239;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        font-size: 26px;
        font-weight: 800;
    }


    /* ---------- Model status ---------- */

    .model-status {
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 11px 16px;
        color: #166534;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 25px;
    }


    /* ---------- Explanation ---------- */

    .explanation-card {
        background: #f8fafc;
        border-left: 4px solid #0f766e;
        border-radius: 10px;
        padding: 16px 20px;
        color: #475569;
        line-height: 1.6;
        margin-bottom: 20px;
    }


    /* ---------- Disclaimer ---------- */

    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 14px;
        padding: 16px 20px;
        color: #9a3412;
        font-size: 13px;
        line-height: 1.6;
        margin-top: 30px;
    }


    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        padding: 20px 0 5px 0;
    }

</style>
""", unsafe_allow_html=True)


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
# HERO HEADER
# =========================================================

st.markdown("""
<div class="hero">

    <div class="hero-title">
        🔬 Explainable Medical Imaging
    </div>

    <div class="hero-subtitle">
        AI-powered chest X-ray analysis using deep learning
        and Grad-CAM explainability to visualize the regions
        influencing the model's prediction.
    </div>

    <div class="hero-badge">
        🧠 ResNet-18 &nbsp; • &nbsp;
        🔍 Grad-CAM &nbsp; • &nbsp;
        🩻 Chest X-ray Analysis
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOAD
# =========================================================

try:

    model = load_model()

    st.markdown(
        f"""
        <div class="model-status">
            ✓ Model loaded successfully
            &nbsp;&nbsp;|&nbsp;&nbsp;
            ResNet-18
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Device: {DEVICE}
        </div>
        """,
        unsafe_allow_html=True
    )

except Exception as e:

    st.error(
        f"Unable to load model: {e}"
    )

    st.stop()


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("""
<div class="disclaimer">

    ⚠️ <strong>Research & Educational Prototype</strong><br>

    This application is intended for research and educational
    purposes only. It is not a medical device and should not
    be used for clinical diagnosis or medical decision-making.

</div>
""", unsafe_allow_html=True)


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    '<div class="section-title">📤 Upload Chest X-ray</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-description">
        Upload a chest X-ray image in PNG, JPG, or JPEG format
        to analyze it using the trained deep learning model.
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ],
    label_visibility="collapsed"
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

    st.markdown(
        '<div class="section-title">🩻 Analysis Results</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # IMAGE + PREDICTION
    # -----------------------------------------------------

    col1, col2 = st.columns(
        [1.1, 0.9],
        gap="large"
    )


    # -----------------------------------------------------
    # ORIGINAL IMAGE
    # -----------------------------------------------------

    with col1:

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card-label">Input Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
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
    # PREDICTION
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


    class_names = [
        "Normal",
        "Pneumonia"
    ]

    prediction = class_names[
        predicted_class
    ]


    normal_probability = probabilities[
        0,
        0
    ].item()

    pneumonia_probability = probabilities[
        0,
        1
    ].item()


    # -----------------------------------------------------
    # RESULT CARD
    # -----------------------------------------------------

    with col2:

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card-label">Model Prediction</div>',
            unsafe_allow_html=True
        )

        if prediction == "Pneumonia":

            st.markdown(
                f"""
                <div class="prediction-pneumonia">
                    ⚠️ Pneumonia
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="prediction-normal">
                    ✓ Normal
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2%}"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # PROBABILITIES
        # -------------------------------------------------

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card-label">Class Probabilities</div>',
            unsafe_allow_html=True
        )

        st.write("")

        st.write(
            f"**Normal** — {normal_probability:.2%}"
        )

        st.progress(
            normal_probability
        )

        st.write(
            f"**Pneumonia** — {pneumonia_probability:.2%}"
        )

        st.progress(
            pneumonia_probability
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # =====================================================
    # GRAD-CAM SECTION
    # =====================================================

    st.markdown(
        '<div class="section-title">🔍 Explainable AI — Grad-CAM</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="explanation-card">

        <strong>How to interpret this visualization:</strong><br>

        Grad-CAM highlights image regions that contributed to
        the model's prediction. Warmer regions indicate areas
        receiving stronger attention from the neural network.

        <br><br>

        The heatmap is intended for model interpretation and
        does not represent a clinically validated diagnostic
        explanation.

        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # GENERATE CAM
    # -----------------------------------------------------

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
    # VISUALIZATIONS
    # -----------------------------------------------------

    gradcam_col1, gradcam_col2 = st.columns(
        2,
        gap="large"
    )

    with gradcam_col1:

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card-label">Grad-CAM Heatmap</div>',
            unsafe_allow_html=True
        )

        st.image(
            heatmap,
            use_container_width=True
        )

        st.caption(
            "Model attention visualization"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with gradcam_col2:

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card-label">Grad-CAM Overlay</div>',
            unsafe_allow_html=True
        )

        st.image(
            overlay,
            use_container_width=True
        )

        st.caption(
            "Heatmap overlaid on the original X-ray"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">

        <strong>Explainable Medical Imaging</strong><br>

        ResNet-18 Transfer Learning • Grad-CAM Explainability<br>

        AI Research & Educational Project

    </div>
    """,
    unsafe_allow_html=True
)
