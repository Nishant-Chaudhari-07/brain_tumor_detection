import numpy as np
import tensorflow as tf
from tensorflow import keras

def find_last_conv_layer(model: keras.Model):
    """
    Return the name of the last Conv2D layer in the model, searching into nested models.
    """
    last_name = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            inner = find_last_conv_layer(layer)
            if inner:
                last_name = inner
        else:
            if isinstance(layer, keras.layers.Conv2D):
                last_name = layer.name
    return last_name

def _ensure_4d(x):
    """
    Ensure input is a float32 tensor of shape (1, H, W, C).
    """
    if isinstance(x, np.ndarray):
        x = tf.convert_to_tensor(x)
    x = tf.cast(x, tf.float32)
    if x.shape.rank == 3:
        x = tf.expand_dims(x, 0)
    return x

def _first_tensor(x):
    """
    If x is a list/tuple, return its first element. Otherwise return x.
    """
    if isinstance(x, (list, tuple)):
        return x[0]
    return x

def make_gradcam_heatmap(
    img_array,
    model: keras.Model,
    last_conv_layer_name: str = None,
    pred_index: int = None
):
    """
    img_array: np.ndarray or tf.Tensor, shape (1, H, W, 3) or (H, W, 3) — preprocessed for the model
    model:     keras.Model
    last_conv_layer_name: optional override (e.g., 'block5_conv3' for VGG16, 'out_relu' for MobileNetV2)
    pred_index: optional class index to explain; defaults to model prediction
    """
    # Ensure correct type/shape
    img_array = _ensure_4d(img_array)

    # Find conv layer
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)
        if last_conv_layer_name is None:
            raise ValueError("No Conv2D layer found for Grad-CAM. Provide last_conv_layer_name explicitly.")
    last_conv = model.get_layer(last_conv_layer_name)

    # Build model that returns (conv_activations, predictions)
    grad_model = keras.models.Model([model.inputs], [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array, training=False)

        # Unwrap possible lists/tuples
        conv_outputs = _first_tensor(conv_outputs)
        preds = _first_tensor(preds)

        # Determine class index robustly
        # preds shape could be (1, 1) [binary sigmoid] or (1, num_classes) [softmax]
        if pred_index is None:
            if preds.shape[-1] == 1:  # binary sigmoid
                p = tf.squeeze(preds, axis=-1)  # (1,)
                pred_index = int((p[0] >= 0.5).numpy())
            else:
                pred_index = int(tf.argmax(preds[0]).numpy())

        # Build scalar class channel
        class_channel = preds[:, pred_index] if preds.shape[-1] > 1 else tf.squeeze(preds, axis=-1)

    # Compute gradients of target class w.r.t conv outputs
    grads = tape.gradient(class_channel, conv_outputs)   # (1, Hc, Wc, C)
    if grads is None:
        raise RuntimeError(
            "Gradients are None. Check last_conv_layer_name and ensure the model graph is connected."
        )

    # Global average pooling over spatial dims -> (C,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)

    # Weight conv outputs by pooled grads and collapse channels -> (Hc, Wc)
    conv_outputs = conv_outputs[0]                          # (Hc, Wc, C)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)  # (Hc, Wc)

    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()
