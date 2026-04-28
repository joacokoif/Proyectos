"""
app.py — Streamlit Deployment App
==================================
Interactive web app for galaxy morphology classification.

Features:
    - Upload any galaxy image for classification
    - Real-time prediction with confidence scores
    - Grad-CAM attention overlay
    - NASA image demo gallery
    - Model selection (ResNet18 / EfficientNet-B0)

Run:
    streamlit run app.py

Author: Galaxy Classifier Project
"""

import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from pathlib import Path
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import config
from dataset import get_transforms
from model import create_model, get_target_layer


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Galaxy Classifier",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stMetric {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING (CACHED)
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(model_name: str):
    """Load trained model (cached to avoid reloading)."""
    checkpoint_path = config.MODELS_DIR / f"best_{model_name}.pth"
    
    if not checkpoint_path.exists():
        return None
    
    model = create_model(name=model_name, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


# ═══════════════════════════════════════════════════════════════
# CLASSIFICATION FUNCTION
# ═══════════════════════════════════════════════════════════════

def classify_galaxy(image: Image.Image, model) -> dict:
    """Classify a galaxy image and generate Grad-CAM."""
    transform = get_transforms("test")
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(config.DEVICE)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_idx = output.argmax(dim=1).item()
    
    prediction = config.IDX_TO_CLASS[pred_idx]
    confidence = probs[pred_idx].item()
    class_probs = {config.IDX_TO_CLASS[i]: probs[i].item() for i in range(config.NUM_CLASSES)}
    
    # Grad-CAM
    gradcam_img = None
    try:
        target_layers = [get_target_layer(model)]
        cam = GradCAM(model=model, target_layers=target_layers)
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        
        img_resized = image.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        
        gradcam_img = show_cam_on_image(img_array, grayscale_cam[0], use_rgb=True)
    except Exception:
        pass
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'class_probs': class_probs,
        'gradcam': gradcam_img
    }


# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown('<h1 class="main-header">🌌 Galaxy Classifier</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Deep Learning Galaxy Morphology Classification<br>'
        'Powered by Galaxy Zoo 2 + Transfer Learning</p>',
        unsafe_allow_html=True
    )
    
    # ─── Sidebar ───
    st.sidebar.title("⚙️ Settings")
    
    model_name = st.sidebar.selectbox(
        "Model Architecture",
        config.AVAILABLE_MODELS,
        index=0,
        help="ResNet18 is faster, EfficientNet-B0 may be more accurate"
    )
    
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=True)
    blur_level = st.sidebar.slider("NASA Mode (Blur)", 0, 15, 0, help="High-res photos confuse the model. Blur them to match the SDSS training data!")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Galaxy Types
    - **🌀 Spiral**: Disk with spiral arms
    - **🟡 Elliptical**: Smooth, featureless
    - **💫 Irregular**: No defined shape
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    Trained on Galaxy Zoo 2 citizen science labels 
    using transfer learning (ImageNet → galaxies).
    
    **Training data**: SDSS galaxy images  
    **NASA images**: Inference demo only
    """)
    
    # ─── Load Model ───
    model = load_model(model_name)
    
    if model is None:
        st.error(
            f"⚠️ No trained model found for **{model_name}**. "
            f"Please run `python train.py --model {model_name}` first."
        )
        return
    
    st.success(f"✅ Model loaded: **{model_name}**")
    
    # ─── Upload Section ───
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload a Galaxy Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload any galaxy image for classification"
        )
    
    with col2:
        st.subheader("🖼️ Or Try a NASA Image")
        nasa_images = list(config.NASA_DIR.glob("*.jpg")) if config.NASA_DIR.exists() else []
        
        if nasa_images:
            selected_nasa = st.selectbox(
                "Select NASA image",
                [img.name for img in nasa_images[:20]],
                index=0
            )
            use_nasa = st.button("🚀 Classify NASA Image")
        else:
            st.info("No NASA images available. Run `python download_data.py` to download.")
            use_nasa = False
    
    # ─── Classification ───
    image = None
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif use_nasa:
        image = Image.open(config.NASA_DIR / selected_nasa)
    
    if image is not None:
        st.markdown("---")
        
        if blur_level > 0:
            image = image.filter(ImageFilter.GaussianBlur(blur_level))
        
        # Classify
        with st.spinner("🔭 Analyzing galaxy morphology..."):
            result = classify_galaxy(image, model)
        
        # Display results
        col_img, col_cam, col_result = st.columns([1, 1, 1])
        
        with col_img:
            st.image(image, caption="Input Galaxy", use_container_width=True)
        
        with col_cam:
            if show_gradcam and result['gradcam'] is not None:
                st.image(result['gradcam'], caption="Grad-CAM Attention Map", use_container_width=True)
            else:
                st.info("Grad-CAM not available")
        
        with col_result:
            # Prediction
            emojis = {'spiral': '🌀', 'elliptical': '🟡', 'irregular': '💫'}
            emoji = emojis.get(result['prediction'], '🔭')
            
            st.markdown(f"""
            <div class="prediction-box">
                <h1>{emoji}</h1>
                <h2>{result['prediction'].upper()}</h2>
                <h3>{result['confidence']:.1%} confidence</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability bars
            st.markdown("#### Class Probabilities")
            for cls, prob in result['class_probs'].items():
                st.progress(prob, text=f"{cls}: {prob:.1%}")
    
    # ─── Footer ───
    st.markdown("---")
    st.markdown(
        "<center><small>Galaxy Classifier • Powered by PyTorch + Galaxy Zoo 2 • "
        "NASA images used for demo only</small></center>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
