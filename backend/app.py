
# Update par Youness Aibech

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, HTTPException

if TYPE_CHECKING:
    import tensorflow as tf


MODEL_IMAGE_SIZE = int(os.getenv("MODEL_IMAGE_SIZE", "150"))
IMAGE_SIZE = (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.45"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "pneumonia_cnn_model.keras"))
_TF = None
_MODEL = None

#main
app = Flask(__name__)
CORS(app)

#the get tf function is used to lazily import TensorFlow only when needed, which can help reduce startup time and memory usage if the model is not immediately required. The get_model function loads the Keras model from disk on first use and caches it for subsequent requests. The decode_image function converts uploaded file bytes into an OpenCV image, while validate_xray_candidate checks if the image has characteristics consistent with a chest X-ray. The preprocess_image function prepares the image for model input, and extract_probability_normal retrieves the predicted probability of the "Normal" class from the model's output. The classify function determines the predicted class label based on the probability and a defined threshold. The generate_gradcam_overlay function creates a Grad-CAM visualization overlay for the input image, highlighting areas that contributed to the model's prediction. Finally, Flask routes are defined for health checks, predictions, and Grad-CAM generation, along with error handlers for various exceptions.
def get_tf():
    global _TF

    if _TF is None:
        import tensorflow as tf

        _TF = tf

    return _TF


def load_keras_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH.resolve()}'. "
            "Place pneumonia_cnn_model.keras in the backend folder or set MODEL_PATH."
        )

    tf = get_tf()
    return tf.keras.models.load_model(MODEL_PATH)


def get_model():
    global _MODEL

    if _MODEL is None:
        _MODEL = load_keras_model()

    return _MODEL


def decode_image(file_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise BadRequest("The uploaded file could not be decoded as an image.")

    return image


def validate_xray_candidate(image_bgr: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    if min(height, width) < 96:
        raise BadRequest("The image is too small. Please upload a clearer chest X-ray.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sampled = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

    color_delta = 0.0
    if image_bgr.ndim == 3:
        sampled_color = cv2.resize(image_bgr, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
        blue, green, red = cv2.split(sampled_color)
        color_delta = float(
            (
                np.mean(np.abs(blue - green))
                + np.mean(np.abs(blue - red))
                + np.mean(np.abs(green - red))
            )
            / 3.0
        )

    contrast = float(np.percentile(sampled, 95) - np.percentile(sampled, 5))
    edge_ratio = float(np.count_nonzero(cv2.Canny(sampled, 50, 150)) / sampled.size)

    # This is a lightweight heuristic to catch obvious non X-ray uploads.
    if color_delta > 14.0:
        raise BadRequest("This image does not look like a chest X-ray. Please upload a grayscale chest radiograph.")

    if contrast < 28.0 or edge_ratio < 0.01:
        raise BadRequest("This image is not a usable chest X-ray. Please upload a clearer chest radiograph.")

    return gray


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    image = decode_image(file_bytes)
    grayscale = validate_xray_candidate(image)
    return preprocess_grayscale_image(grayscale)


def preprocess_grayscale_image(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    return image


def extract_probability_normal(prediction: Any) -> float:
    values = np.asarray(prediction, dtype=np.float32).reshape(-1)

    if values.size == 0:
        raise RuntimeError("Model returned an empty prediction.")

    if values.size == 1:
        probability_normal = float(values[0])
    else:
        # For a two-neuron softmax model, class index 1 represents "Normal".
        probability_normal = float(values[1])

    return min(max(probability_normal, 0.0), 1.0)


def classify(probability_normal: float) -> tuple[int, str]:
    predicted_class = 1 if probability_normal >= THRESHOLD else 0
    label = "Normal" if predicted_class == 1 else "Pneumonia"
    return predicted_class, label


def get_last_conv_layer_name() -> str:
    tf = get_tf()

    for layer in reversed(get_model().layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise RuntimeError("Grad-CAM requires a model with at least one Conv2D layer.")


def build_gradcam_model(layer_name: str):
    tf = get_tf()
    model = get_model()
    inputs = tf.keras.Input(shape=model.input_shape[1:], name="gradcam_input")
    x = inputs
    conv_outputs = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        x = layer(x)
        if layer.name == layer_name:
            conv_outputs = x

    if conv_outputs is None:
        raise RuntimeError(f"Could not find convolutional layer '{layer_name}' for Grad-CAM.")

    return tf.keras.models.Model(inputs=inputs, outputs=[conv_outputs, x], name="gradcam_model")


def generate_gradcam_overlay(file_bytes: bytes, target_class: int | None = None) -> tuple[str, dict[str, Any]]:
    tf = get_tf()
    image_bgr = decode_image(file_bytes)
    grayscale = validate_xray_candidate(image_bgr)

    preprocessed = preprocess_grayscale_image(grayscale)
    layer_name = os.getenv("GRADCAM_LAYER") or get_last_conv_layer_name()
    grad_model = build_gradcam_model(layer_name)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(preprocessed, training=False)
        probability_normal = extract_probability_normal(predictions)
        inferred_class, label = classify(probability_normal)
        class_index = inferred_class if target_class is None else target_class

        if class_index == 1:
            class_score = tf.reshape(predictions, (-1,))[-1]
        elif class_index == 0:
            class_score = 1.0 - tf.reshape(predictions, (-1,))[-1]
        else:
            raise BadRequest("target_class must be 0 for Pneumonia or 1 for Normal.")

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise RuntimeError("Could not compute Grad-CAM gradients for this model.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap_np = heatmap.numpy()

    heatmap_np = cv2.resize(heatmap_np, IMAGE_SIZE)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_np), cv2.COLORMAP_JET)
    base_image = cv2.resize(grayscale, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base_image, 0.62, heatmap_color, 0.38, 0)

    success, buffer = cv2.imencode(".png", overlay)

    if not success:
        raise RuntimeError("Could not encode Grad-CAM overlay.")

    metadata = {
        "label": label,
        "predicted_class": inferred_class,
        "probability_normal": probability_normal,
        "threshold": THRESHOLD,
        "gradcam_layer": layer_name,
        "target_class": class_index,
    }

    return base64.b64encode(buffer).decode("utf-8"), metadata


@app.get("/health")
def health() -> tuple[Any, int]:
    return jsonify(
        {
            "status": "ok",
            "model_path": str(MODEL_PATH),
            "model_loaded": _MODEL is not None,
            "model_exists": MODEL_PATH.exists(),
            "image_size": MODEL_IMAGE_SIZE,
            "threshold": THRESHOLD,
        }
    ), 200


@app.get("/")
def index() -> tuple[Any, int]:
    return jsonify(
        {
            "service": "Pneumonia AI Screening API",
            "status": "ok",
            "available_endpoints": ["/health", "/predict", "/gradcam"],
        }
    ), 200


@app.post("/predict")
def predict() -> tuple[Any, int]:
    if "file" not in request.files:
        raise BadRequest("Missing image file. Send multipart/form-data with a 'file' field.")

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        raise BadRequest("No file was selected.")

    image = preprocess_image(uploaded_file.read())
    prediction = get_model().predict(image, verbose=0)
    probability_normal = extract_probability_normal(prediction)
    predicted_class, label = classify(probability_normal)

    return jsonify(
        {
            "label": label,
            "predicted_class": predicted_class,
            "probability_normal": probability_normal,
            "threshold": THRESHOLD,
        }
    ), 200


@app.post("/gradcam")
def gradcam() -> tuple[Any, int]:
    if "file" not in request.files:
        raise BadRequest("Missing image file. Send multipart/form-data with a 'file' field.")

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        raise BadRequest("No file was selected.")

    target_class_value = request.form.get("target_class")
    try:
        target_class = int(target_class_value) if target_class_value not in (None, "") else None
    except ValueError as exc:
        raise BadRequest("target_class must be 0 for Pneumonia or 1 for Normal.") from exc

    overlay_base64, metadata = generate_gradcam_overlay(uploaded_file.read(), target_class)

    return jsonify({"gradcam_image_base64": overlay_base64, **metadata}), 200


@app.errorhandler(BadRequest)
def handle_bad_request(error: BadRequest) -> tuple[Any, int]:
    return jsonify({"error": error.description}), 400


@app.errorhandler(HTTPException)
def handle_http_exception(error: HTTPException) -> tuple[Any, int]:
    return jsonify({"error": error.description}), error.code or 500


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception) -> tuple[Any, int]:
    app.logger.exception("Unhandled error during request")
    return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG") == "1")
