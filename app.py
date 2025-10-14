import os
import sys
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import gdown

# --- Ensure we can import from src/ even if __init__.py is missing ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.predict import load_model_and_classes, predict_image, preprocess_image
from src.gradcam import make_gradcam_heatmap

# --- Streamlit page config ---
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor Detection")
st.caption("Upload a brain MRI image. The app loads a Keras model from `models/brain_tumor_classifier.keras`.")

# --- Model paths ---
MODEL_PATH = os.path.join("models", "brain_tumor_classifier.keras")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")

# Your existing Google Drive share link (kept the same)
GDRIVE_MODEL_URL = "https://drive.google.com/file/d/1zuxX0LishfUDIT7hNunj9-tsR-mMgxFq/view?usp=drive_link"

def ensure_model_files(model_path: str = MODEL_PATH, class_names_path: str = CLASS_NAMES_PATH):
    """
    Make sure the model and class_names.json exist locally.
    Downloads the .keras from your Google Drive link using gdown if missing.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Download model if missing
    if not os.path.isfile(model_path):
        # gdown can parse 'view' links if fuzzy=True
        st.info("Downloading model weights from Google Drive...")
        gdown.download(GDRIVE_MODEL_URL, model_path, quiet=False, fuzzy=True)

    # We don't auto-download class_names.json (keep your original logic).
    # Just ensure the models folder exists.
    return model_path, class_names_path

@st.cache_resource(show_spinner=True)
def _load_model(model_path: str = MODEL_PATH, class_names_path: str = CLASS_NAMES_PATH):
    """
    Load model and class names. Returns (model, class_names, err_msg_or_None).
    """
    # Ensure files are present (downloads model if needed)
    ensure_model_files(model_path, class_names_path)

    if not os.path.isfile(model_path):
        return None, None, f"Model file not found: {model_path}"
    if not os.path.isfile(class_names_path):
        return None, None, f"class_names.json not found: {class_names_path}"

    try:
        model, class_names = load_model_and_classes(model_path, class_names_path)
        return model, class_names, None
    except Exception as e:
        return None, None, f"Error loading model: {e}"

with st.sidebar:
    st.header("Settings")
    show_gradcam = st.checkbox("Show Grad-CAM (experimental)", value=False)
    threshold = st.slider("Binary threshold", 0.1, 0.9, 0.5, 0.05)

model, class_names, err = _load_model()
if err:
    st.warning(err)

uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    if st.button("Predict", disabled=(model is None)):
        if model is None:
            st.error("Model not loaded.")
        else:
            out = predict_image(model, class_names, img, threshold=threshold)
            st.subheader("Prediction")
            st.write(f"**Predicted class:** {out['pred_class']}")

            # Probabilities plot
            if len(class_names) == 2:
                probs = out["probs"]
            else:
                probs = np.array(out["probabilities"])

            fig, ax = plt.subplots()
            ax.bar(class_names, probs)
            ax.set_ylabel("Probability")
            ax.set_title("Class Probabilities")
            # ensure ticks exist before setting labels to avoid warnings
            ax.set_xticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            st.pyplot(fig)

            if show_gradcam:
                x = preprocess_image(img)
                try:
                    heatmap = make_gradcam_heatmap(x, model)
                    heatmap = (heatmap * 255).astype(np.uint8)
                    heatmap_img = Image.fromarray(heatmap).resize(img.size)
                    st.subheader("Grad-CAM")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img, caption="Original", use_column_width=True)
                    with col2:
                        st.image(heatmap_img, caption="Heatmap", use_column_width=True)
                except Exception as e:
                    st.info(f"Grad-CAM not available: {e}")
else:
    st.info("Upload an image to get started.")
