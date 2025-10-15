import numpy as np
import tensorflow as tf
from tensorflow import keras

def find_last_conv_layer(model: keras.Model):
    """
    Return the name of the last Conv2D layer in the model (searches deeply).
    """
    last = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            # recurse into nested models
            nested = find_last_conv_layer(layer)
            if nested:
                last = nested
        else:
            if isinstance(layer, keras.layers.Conv2D):
                last = layer.name
    return last

def make_gradcam_heatmap(img_array, model: keras.Model, last_conv_layer_name: str = None, pred_index: int = None):
    """
    img_array: (1, H, W, 3) preprocessed tensor/ndarray
    model:     keras.Model
    last_conv_layer_name: override if auto-detect fails (e.g., 'block5_conv3' for VGG16, 'Conv_1' or 'out_relu' for MobileNetV2)
    pred_index: class index to explain; defaults to model's argmax for the input
    """
    if isinstance(img_array, np.ndarray):
        img_array = tf.convert_to_tensor(img_array)

    # Find last conv layer
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)
        if last_conv_layer_name is None:
            raise ValueError("No Conv2D layer found for Grad-CAM. Provide last_conv_layer_name explicitly.")

    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Build a model that maps the input image to the activations of the last conv layer + predictions
    grad_model = keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    # Compute the gradient of the top predicted class for the input image with respect to the activations of the last conv layer
    grads = tape.gradient(class_channel, conv_outputs)          # shape: (1, Hc, Wc, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))        # shape: (C,)

    # Weight the channels by importance to this class, then reduce across channels
    conv_outputs = conv_outputs[0]                              # (Hc, Wc, C)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)  # (Hc, Wc)

    # Normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    denom = tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap / denom

    return heatmap.numpy()
