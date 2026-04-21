from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest, HTTPException

if TYPE_CHECKING:
    import tensorflow as tf


MODEL_IMAGE_SIZE = int(os.getenv("MODEL_IMAGE_SIZE", "150"))
IMAGE_SIZE = (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "pneumonia_cnn_model.keras"))
THRESHOLD_PATH = Path(os.getenv("THRESHOLD_PATH", "pneumonia_threshold.json"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_TF = None
_MODEL = None
_GRADCAM_MODELS: dict[str, Any] = {}


app = Flask(__name__)
CORS(app)


def get_tf():
    global _TF

    if _TF is None:
        import tensorflow as tf

        _TF = tf

    return _TF


def load_threshold() -> float:
    env_threshold = os.getenv("PREDICTION_THRESHOLD")
    if env_threshold is not None:
        return float(env_threshold)

    if THRESHOLD_PATH.exists():
        payload = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
        if "threshold" not in payload:
            raise ValueError(f"Threshold file '{THRESHOLD_PATH}' must contain a 'threshold' field.")
        return float(payload["threshold"])

    return 0.45


def load_keras_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH.resolve()}'. "
            "Place pneumonia_cnn_model.keras in the backend folder or set MODEL_PATH."
        )

    tf = get_tf()
    return tf.keras.models.load_model(MODEL_PATH)


THRESHOLD = load_threshold()


def get_model():
    global _MODEL

    if _MODEL is None:
        _MODEL = load_keras_model()

    return _MODEL


def get_gradcam_model(layer_name: str):
    cached = _GRADCAM_MODELS.get(layer_name)

    if cached is None:
        cached = build_gradcam_model(layer_name)
        _GRADCAM_MODELS[layer_name] = cached

    return cached


def validate_file_extension(filename: str) -> None:
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise BadRequest(f"Unsupported file type. Please upload one of: {allowed}.")


def read_uploaded_image_bytes() -> bytes:
    if "file" not in request.files:
        raise BadRequest("Missing image file. Send multipart/form-data with a 'file' field.")

    uploaded_file: FileStorage = request.files["file"]

    if uploaded_file.filename == "":
        raise BadRequest("No file was selected.")

    validate_file_extension(uploaded_file.filename)
    file_bytes = uploaded_file.read()

    if not file_bytes:
        raise BadRequest("The uploaded file is empty.")

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        max_size_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise BadRequest(f"File is too large. Please upload an image under {max_size_mb:.0f} MB.")

    return file_bytes


def parse_target_class(value: str | None) -> int | None:
    if value in (None, ""):
        return None

    try:
        target_class = int(value)
    except ValueError as exc:
        raise BadRequest("target_class must be 0 for Pneumonia or 1 for Normal.") from exc

    if target_class not in (0, 1):
        raise BadRequest("target_class must be 0 for Pneumonia or 1 for Normal.")

    return target_class


def decode_image(file_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise BadRequest("The uploaded file could not be decoded as an image.")

    return image


def validate_xray_candidate(image_bgr: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]

    if min(height, width) < 150:
        raise BadRequest("The image is too small. Please upload a clearer chest X-ray.")

    # Chest X-rays are roughly square to slightly portrait (PA or AP view).
    aspect = width / height
    if aspect < 0.55 or aspect > 1.65:
        raise BadRequest("This image does not look like a chest X-ray. Please upload a frontal chest radiograph.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sampled = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

    # Reject colored images — X-rays are grayscale.
    if image_bgr.ndim == 3:
        sampled_color = cv2.resize(image_bgr, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
        b, g, r = cv2.split(sampled_color)
        color_delta = float((np.mean(np.abs(b - g)) + np.mean(np.abs(b - r)) + np.mean(np.abs(g - r))) / 3.0)
        if color_delta > 14.0:
            raise BadRequest("This image does not look like a chest X-ray. Please upload a grayscale chest radiograph.")

    # Dynamic range: X-rays span a wide intensity range.
    contrast = float(np.percentile(sampled, 95) - np.percentile(sampled, 5))
    if contrast < 55.0:
        raise BadRequest("This image has insufficient contrast. Please upload a proper chest X-ray.")

    # Pixel distribution: chest X-rays have dark lung fields and bright bone/tissue regions.
    flat = sampled.flatten().astype(np.float32)
    dark_ratio = float(np.mean(flat < 85))    # lung air → dark pixels
    bright_ratio = float(np.mean(flat > 155)) # bone / mediastinum → bright pixels

    if dark_ratio < 0.10 or bright_ratio < 0.07:
        raise BadRequest("This image does not match the expected pixel distribution of a chest X-ray.")

    # Edge density: too sparse = blank image, too dense = document/text.
    edge_ratio = float(np.count_nonzero(cv2.Canny(sampled, 50, 150)) / sampled.size)
    if edge_ratio < 0.01 or edge_ratio > 0.28:
        raise BadRequest("This image does not look like a chest X-ray. Please upload a chest radiograph.")

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
    grad_model = get_gradcam_model(layer_name)

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
            "gradcam_model_cache_size": len(_GRADCAM_MODELS),
            "image_size": MODEL_IMAGE_SIZE,
            "threshold": THRESHOLD,
            "threshold_source": "env" if os.getenv("PREDICTION_THRESHOLD") is not None else ("file" if THRESHOLD_PATH.exists() else "default"),
            "threshold_path": str(THRESHOLD_PATH),
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
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
    image = preprocess_image(read_uploaded_image_bytes())
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
    target_class = parse_target_class(request.form.get("target_class"))
    overlay_base64, metadata = generate_gradcam_overlay(read_uploaded_image_bytes(), target_class)

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
