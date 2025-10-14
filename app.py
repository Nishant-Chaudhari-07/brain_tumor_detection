import os
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import gdown, os

from src.predict import load_model_and_classes, predict_image, preprocess_image
from src.gradcam import make_gradcam_heatmap

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor Detection")
st.caption("Upload a brain MRI image. The app loads a Keras model from `models/brain_tumor_classifier.keras`.")

MODEL_PATH = "models/brain_tumor_classifier.keras"
CLASS_NAMES_PATH = "models/class_names.json"

@st.cache_resource(show_spinner=True)
def _load_model():
    if not os.path.exists(model_path):
        url = "https://drive.google.com/file/d/1zuxX0LishfUDIT7hNunj9-tsR-mMgxFq/view?usp=drive_link"  # shareable link
        gdown.download(url, model_path, quiet=False)
    if not os.path.isfile(CLASS_NAMES_PATH):
        return None, None, "class_names.json not found in models/"
    try:
        model, class_names = load_model_and_classes(MODEL_PATH, CLASS_NAMES_PATH)
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
            fig, ax = plt.subplots()
            if len(class_names) == 2:
                probs = out["probs"]
            else:
                probs = np.array(out["probabilities"])
            ax.bar(class_names, probs)
            ax.set_ylabel("Probability")
            ax.set_title("Class Probabilities")
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            st.pyplot(fig)

            if show_gradcam:
                x = preprocess_image(img)
                try:
                    heatmap = make_gradcam_heatmap(x, model)
                    heatmap = (heatmap * 255).astype(np.uint8)
                    from PIL import Image as _Image
                    heatmap_img = _Image.fromarray(heatmap).resize(img.size)
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
