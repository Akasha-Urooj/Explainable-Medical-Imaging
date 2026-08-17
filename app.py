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
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROFESSIONAL UI CSS
# =========================================================

st.html("""
<style>

html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
}

.stApp {
    background: #f5f7fb;
}

.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}


/* ================= SIDEBAR ================= */

[data-testid="stSidebar"] {
    background: #0b1220;
}

[data-testid="stSidebar"] * {
    color: #e5edf8;
}

.sidebar-title {
    font-size: 26px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 4px;
}

.sidebar-subtitle {
    color: #91a1b7;
    font-size: 13px;
    line-height: 1.5;
}

.sidebar-line {
    height: 1px;
    background: #26344a;
    margin: 22px 0;
}

.sidebar-card {
    background: #121d30;
    border: 1px solid #26364e;
    border-radius: 12px;
    padding: 13px;
    margin: 9px 0;
}

.sidebar-card-title {
    font-size: 11px;
    color: #8ea0b8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.sidebar-card-value {
    font-size: 14px;
    color: #ffffff;
    font-weight: 700;
    margin-top: 3px;
}


/* ================= HERO ================= */

.hero {
    background: linear-gradient(
        135deg,
        #0b1220 0%,
        #12294d 55%,
        #087f73 100%
    );
    border-radius: 24px;
    padding: 38px 42px;
    margin-bottom: 24px;
    box-shadow: 0 15px 40px rgba(15, 23, 42, 0.18);
}

.hero-title {
    color: #ffffff;
    font-size: 40px;
    font-weight: 850;
    letter-spacing: -1px;
}

.hero-text {
    color: #d8e3f1;
    font-size: 15px;
    line-height: 1.7;
    max-width: 900px;
    margin-top: 9px;
}

.badges {
    margin-top: 19px;
}

.badge {
    display: inline-block;
    padding: 7px 13px;
    margin-right: 7px;
    border-radius: 50px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.20);
    color: #ffffff;
    font-size: 12px;
}


/* ================= STATUS ================= */

.status {
    background: #ffffff;
    border: 1px solid #dce3ec;
    border-radius: 14px;
    padding: 13px 17px;
    margin-bottom: 22px;
    box-shadow: 0 4px 14px rgba(15,23,42,0.04);
}

.status-online {
    color: #039855;
    font-weight: 800;
    font-size: 13px;
}

.status-detail {
    color: #667085;
    font-size: 13px;
}


/* ================= SECTION ================= */

.section-title {
    color: #101828;
    font-size: 23px;
    font-weight: 800;
    margin-top: 22px;
    margin-bottom: 3px;
}

.section-text {
    color: #667085;
    font-size: 14px;
    margin-bottom: 16px;
}


/* ================= RESULT ================= */

.prediction-normal {
    background: #ecfdf3;
    border: 1px solid #abefc6;
    border-radius: 15px;
    padding: 18px;
    color: #027a48;
    font-size: 27px;
    font-weight: 800;
    text-align: center;
}

.prediction-pneumonia {
    background: #fff1f3;
    border: 1px solid #fecdd3;
    border-radius: 15px;
    padding: 18px;
    color: #c01048;
    font-size: 27px;
    font-weight: 800;
    text-align: center;
}


/* ================= CARD ================= */

.card {
    background: #ffffff;
    border: 1px solid #dce3ec;
    border-radius: 20px;
    padding: 21px;
    box-shadow: 0 7px 22px rgba(15,23,42,0.05);
}

.card-label {
    color: #667085;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
}


/* ================= EXPLANATION ================= */

.explain {
    background: #edfafa;
    border: 1px solid #b9e6df;
    border-left: 5px solid #087f73;
    border-radius: 14px;
    padding: 17px 19px;
    color: #175e59;
    line-height: 1.65;
    font-size: 14px;
}


/* ================= DISCLAIMER ================= */

.disclaimer {
    background: #fffaeb;
    border: 1px solid #fedf89;
    border-radius: 14px;
    padding: 15px 18px;
    color: #7a2e0e;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 20px;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #98a2b3;
    font-size: 12px;
    line-height: 1.8;
    padding: 18px;
}

</style>
""")


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
# TRANSFORM
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
# SIDEBAR
# =========================================================

with st.sidebar:

    st.html("""
    <div class="sidebar-title">
        🩻 MedExplain AI
    </div>

    <div class="sidebar-subtitle">
        Explainable Medical Imaging
    </div>

    <div class="sidebar-line"></div>

    <div class="sidebar-card">
        <div class="sidebar-card-title">
            Architecture
        </div>
        <div class="sidebar-card-value">
            ResNet-18
        </div>
    </div>

    <div class="sidebar-card">
        <div class="sidebar-card-title">
            Task
        </div>
        <div class="sidebar-card-value">
            Pneumonia Classification
        </div>
    </div>

    <div class="sidebar-card">
        <div class="sidebar-card-title">
            Explainability
        </div>
        <div class="sidebar-card-value">
            Grad-CAM
        </div>
    </div>

    <div class="sidebar-line"></div>
    """)

    st.markdown("### ⚙️ System")

    st.info(
        f"Device: {str(DEVICE).upper()}"
    )

    st.caption(
        "Research & Educational Prototype"
    )


# =========================================================
# HERO
# =========================================================

st.html("""
<div class="hero">

    <div class="hero-title">
        🔬 Explainable Medical Imaging
    </div>

    <div class="hero-text">
        AI-powered chest X-ray analysis using deep learning
        with transparent visual explanations through Grad-CAM.
        Upload an X-ray and explore the model's prediction,
        confidence and attention regions.
    </div>

    <div class="badges">

        <span class="badge">
            🧠 ResNet-18
        </span>

        <span class="badge">
            🔍 Grad-CAM
        </span>

        <span class="badge">
            🩻 Chest X-ray
        </span>

        <span class="badge">
            ⚡ AI Analysis
        </span>

    </div>

</div>
""")


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = load_model()

    st.html(f"""
    <div class="status">

        <span class="status-online">
            ● SYSTEM ONLINE
        </span>

        <span class="status-detail">
            &nbsp; ResNet-18 loaded successfully
            &nbsp; • &nbsp;
            Device: {DEVICE}
        </span>

    </div>
    """)

except Exception as e:

    st.error(
        f"Unable to load model: {e}"
    )

    st.stop()


# =========================================================
# DISCLAIMER
# =========================================================

st.html("""
<div class="disclaimer">

    ⚠️ <b>Research & Educational Prototype</b><br>

    This application is intended for research and educational
    purposes only. It is not a medical device and should not
    be used for clinical diagnosis or medical decision-making.

</div>
""")


# =========================================================
# UPLOAD
# =========================================================

st.html("""
<div class="section-title">
    📤 Upload Chest X-ray
</div>

<div class="section-text">
    Upload a chest X-ray image to begin AI-powered analysis.
    Supported formats: PNG, JPG and JPEG.
</div>
""")

uploaded_file = st.file_uploader(
    "Drop your chest X-ray here",
    type=[
        "png",
        "jpg",
        "jpeg"
    ],
    label_visibility="visible"
)


# =========================================================
# ANALYSIS
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("L")

    image = image.resize(
        (224, 224)
    )


    # =====================================================
    # ANALYSIS TITLE
    # =====================================================

    st.html("""
    <div class="section-title">
        📊 Analysis Dashboard
    </div>

    <div class="section-text">
        Deep learning prediction and model explainability results.
    </div>
    """)


    # =====================================================
    # PREPARE IMAGE
    # =====================================================

    image_tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(DEVICE)


    # =====================================================
    # PREDICTION
    # =====================================================

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


    # =====================================================
    # IMAGE + RESULT
    # =====================================================

    col1, col2 = st.columns(
        [1.15, 0.85],
        gap="large"
    )


    # =====================================================
    # IMAGE CARD
    # =====================================================

    with col1:

        st.html("""
        <div class="card">
            <div class="card-label">
                INPUT CHEST X-RAY
            </div>
        </div>
        """)

        st.image(
            image,
            use_container_width=True
        )

        st.caption(
            f"📁 {uploaded_file.name}"
        )


    # =====================================================
    # RESULT CARD
    # =====================================================

    with col2:

        st.html("""
        <div class="card">
            <div class="card-label">
                MODEL PREDICTION
            </div>
        </div>
        """)

        if prediction == "Pneumonia":

            st.html("""
            <div class="prediction-pneumonia">
                ⚠️ Pneumonia
            </div>
            """)

        else:

            st.html("""
            <div class="prediction-normal">
                ✓ Normal
            </div>
            """)

        st.write("")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2%}"
        )

        st.progress(
            confidence
        )


    # =====================================================
    # PROBABILITY SECTION
    # =====================================================

    st.write("")

    st.html("""
    <div class="section-title">
        📈 Class Probabilities
    </div>

    <div class="section-text">
        Probability distribution produced by the classification model.
    </div>
    """)

    p1, p2 = st.columns(2)

    with p1:

        st.metric(
            "🟢 Normal",
            f"{normal_probability:.2%}"
        )

        st.progress(
            normal_probability
        )


    with p2:

        st.metric(
            "🔴 Pneumonia",
            f"{pneumonia_probability:.2%}"
        )

        st.progress(
            pneumonia_probability
        )


    # =====================================================
    # EXPLAINABILITY
    # =====================================================

    st.divider()

    st.html("""
    <div class="section-title">
        🔍 Explainable AI
    </div>

    <div class="section-text">
        Visualizing the regions that influenced the neural network.
    </div>

    <div class="explain">

        <b>How Grad-CAM works</b><br>

        Grad-CAM highlights regions of the X-ray that contributed
        to the model's prediction. Warmer regions represent stronger
        model attention.

        <br><br>

        This visualization is intended for model interpretation
        and is not a clinically validated diagnostic explanation.

    </div>
    """)


    # =====================================================
    # GENERATE GRAD-CAM
    # =====================================================

    with st.spinner(
        "Generating Grad-CAM explanation..."
    ):

        cam = generate_gradcam(
            model,
            image_tensor,
            predicted_class
        )


    # =====================================================
    # ORIGINAL
    # =====================================================

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


    # =====================================================
    # HEATMAP
    # =====================================================

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


    # =====================================================
    # OVERLAY
    # =====================================================

    overlay = cv2.addWeighted(
        original_rgb,
        0.6,
        heatmap,
        0.4,
        0
    )


    # =====================================================
    # GRAD-CAM TABS
    # =====================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "🩻 Original X-ray",
            "🌡️ Grad-CAM Heatmap",
            "🔬 Attention Overlay"
        ]
    )


    with tab1:

        st.image(
            original,
            use_container_width=True
        )

        st.caption(
            "Original chest X-ray used as model input."
        )


    with tab2:

        st.image(
            heatmap,
            use_container_width=True
        )

        st.caption(
            "Grad-CAM heatmap showing model attention."
        )


    with tab3:

        st.image(
            overlay,
            use_container_width=True
        )

        st.caption(
            "Grad-CAM heatmap overlaid on the original X-ray."
        )


    # =====================================================
    # INTERPRETATION
    # =====================================================

    st.info(
        """
        **Interpretation Note:** Grad-CAM provides an approximate
        visualization of model attention. Highlighted areas should
        not be interpreted as exact disease localization or as a
        clinical diagnosis.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.html("""
<div class="footer">

    <b>🔬 Explainable Medical Imaging</b><br>

    ResNet-18 Transfer Learning
    &nbsp; • &nbsp;
    Grad-CAM Explainability
    &nbsp; • &nbsp;
    Chest X-ray Analysis

    <br>

    Research & Educational Prototype

</div>
""")
