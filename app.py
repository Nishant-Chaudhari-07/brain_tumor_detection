import os
import sys
import json
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
from src.gradcam import make_gradcam_heatmap  # updated per the last message

# --- Streamlit page config ---
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor Detection")
st.caption("Upload a brain MRI image. The app loads a Keras model from `models/brain_tumor_classifier.keras`.")

# --- Paths & Drive link ---
MODEL_PATH = os.path.join("models", "brain_tumor_classifier.keras")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")

# Optional diagnostics artifacts (place them next to the model)
TRAIN_PNG = os.path.join("models", "training_curves.png")
CM_PNG = os.path.join("models", "confusion_matrix.png")
METRICS_JSON = os.path.join("models", "metrics.json")  # optional

# Keep your Google Drive model link
GDRIVE_MODEL_URL = "https://drive.google.com/file/d/1zuxX0LishfUDIT7hNunj9-tsR-mMgxFq/view?usp=drive_link"

def ensure_model_files(model_path: str = MODEL_PATH, class_names_path: str = CLASS_NAMES_PATH):
    """
    Make sure the model and class_names.json exist locally.
    Downloads the .keras from your Google Drive link using gdown if missing.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.isfile(model_path):
        st.info("Downloading model weights from Google Drive...")
        gdown.download(GDRIVE_MODEL_URL, model_path, quiet=False, fuzzy=True)
    return model_path, class_names_path

@st.cache_resource(show_spinner=True)
def _load_model(model_path: str = MODEL_PATH, class_names_path: str = CLASS_NAMES_PATH):
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

# ------------- Sidebar -------------
with st.sidebar:
    st.header("Settings")
    show_gradcam = st.checkbox("Show Grad-CAM (experimental)", value=False)
    threshold = st.slider("Binary threshold", 0.1, 0.9, 0.5, 0.05)

    st.markdown("---")
    st.subheader("Model Diagnostics")
    show_diagnostics = st.checkbox("Show training curves & confusion matrix", value=True)

# ------------- Load model -------------
model, class_names, err = _load_model()
if err:
    st.warning(err)

# ------------- Upload & Predict -------------
uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    if st.button("Predict", disabled=(model is None)):
        if model is None:
            st.error("Model not loaded.")
        else:
            # Inference
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
            ax.set_xticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            st.pyplot(fig)

            # Grad-CAM (define x *inside* try so it's always in scope)
            if show_gradcam:
                try:
                    # Choose a last conv layer if you know the backbone:
                    # last_layer = "block5_conv3"    # VGG16
                    # last_layer = "out_relu"        # MobileNetV2
                    last_layer = None  # auto-detect

                    x = preprocess_image(img)  # <--- define x here
                    heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name=last_layer)
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

# ------------- Diagnostics section -------------
if show_diagnostics:
    st.markdown("---")
    st.header("📊 Model Diagnostics")

    if os.path.isfile(METRICS_JSON):
        try:
            with open(METRICS_JSON, "r") as f:
                m = json.load(f)
            cols = st.columns(3)
            cols[0].metric("Val Accuracy", f"{m.get('val_accuracy', '—')}")
            cols[1].metric("Val Loss", f"{m.get('val_loss', '—')}")
            cols[2].metric("AUC", f"{m.get('auc', '—')}")
        except Exception:
            st.info("metrics.json found but could not be parsed.")

    if os.path.isfile(TRAIN_PNG):
        st.subheader("Training curves")
        st.image(TRAIN_PNG, caption="Loss & Accuracy over epochs", use_column_width=True)
        st.caption("Interpretation: Training/validation loss should trend down; accuracy should trend up. "
                   "Divergence between training and validation curves can indicate overfitting.")
    else:
        st.info("Add your `training_curves.png` to the models/ folder to display training curves.")

    if os.path.isfile(CM_PNG):
        st.subheader("Confusion matrix")
        st.image(CM_PNG, caption="Normalized confusion matrix", use_column_width=True)
        st.caption("Interpretation: Rows are true classes, columns are predicted classes. "
                   "Higher diagonal values indicate better per-class accuracy; off-diagonal cells indicate misclassifications.")
    else:
        st.info("Add your `confusion_matrix.png` to the models/ folder to display the confusion matrix.")

    st.subheader("About this model")
    st.markdown(
        """
**Architecture:** Transfer learning with a pretrained CNN backbone (e.g., VGG16/MobileNetV2) and a small classification head.  
**Explainability:** Grad-CAM highlights salient regions that most influence predictions.  
**Notes:** These visuals support understanding but should not be used for clinical decision-making.
        """
    )
