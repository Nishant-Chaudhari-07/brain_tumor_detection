import json
import numpy as np
from typing import List, Dict, Tuple
import tensorflow as tf
from tensorflow import keras
from PIL import Image

def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    img = img.convert("RGB").resize(target_size)
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return x

def load_model_and_classes(model_path: str, class_names_path: str) -> Tuple[keras.Model, List[str]]:
    model = keras.models.load_model(model_path)
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    return model, class_names

def predict_image(model: keras.Model, class_names: List[str], img: Image.Image, threshold: float = 0.5) -> Dict:
    x = preprocess_image(img)
    probs = model.predict(x, verbose=0)
    num_classes = len(class_names)

    if num_classes == 2:
        p = float(probs.ravel()[0])
        pred_idx = int(p >= threshold)
        return {
            "pred_class": class_names[pred_idx],
            "prob_positive": p,
            "probs": [1.0 - p, p],
            "class_names": class_names
        }
    else:
        probs = probs[0].tolist()
        pred_idx = int(np.argmax(probs))
        return {
            "pred_class": class_names[pred_idx],
            "probabilities": probs,
            "class_names": class_names
        }