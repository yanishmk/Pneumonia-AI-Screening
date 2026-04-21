from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf


MODEL_IMAGE_SIZE = int(os.getenv("MODEL_IMAGE_SIZE", "150"))
IMAGE_SIZE = (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.45"))
MODEL_FILENAME = "pneumonia_cnn_model.keras"
GRADCAM_LAYER = os.getenv("GRADCAM_LAYER")
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(12_000_000)))
MIN_IMAGE_DIMENSION = int(os.getenv("MIN_IMAGE_DIMENSION", "96"))

_MODEL: tf.keras.Model | None = None
_GRADCAM_MODELS: dict[str, tf.keras.Model] = {}


def resolve_model_path() -> Path:
    override = os.getenv("MODEL_PATH")
    if override:
        return Path(override)

    current_dir = Path(__file__).resolve().parent
    candidates = [
        current_dir / MODEL_FILENAME,
        current_dir.parent / "backend" / MODEL_FILENAME,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


MODEL_PATH = resolve_model_path()


def get_model() -> tf.keras.Model:
    global _MODEL

    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Copy pneumonia_cnn_model.keras into the Space folder or set MODEL_PATH."
            )

        _MODEL = tf.keras.models.load_model(MODEL_PATH)

    return _MODEL


def get_gradcam_model(layer_name: str) -> tf.keras.Model:
    cached = _GRADCAM_MODELS.get(layer_name)

    if cached is None:
        cached = build_gradcam_model(get_model(), layer_name)
        _GRADCAM_MODELS[layer_name] = cached

    return cached


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image

    clipped = np.clip(image, 0, 255)
    return clipped.astype(np.uint8)


def to_bgr(image: np.ndarray) -> np.ndarray:
    image = ensure_uint8(image)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def validate_xray_candidate(image_bgr: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]

    if height * width > MAX_IMAGE_PIXELS:
        raise gr.Error("The uploaded image is too large to analyze. Please upload a smaller chest X-ray.")

    if min(height, width) < MIN_IMAGE_DIMENSION:
        raise gr.Error("The image is too small. Please upload a clearer chest X-ray.")

    aspect = width / height
    if aspect < 0.55 or aspect > 1.65:
        raise gr.Error("This image does not look like a frontal chest X-ray.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sampled = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

    color_delta = 0.0
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
    flat = sampled.flatten().astype(np.float32)
    dark_ratio = float(np.mean(flat < 85))
    bright_ratio = float(np.mean(flat > 155))
    edge_ratio = float(np.count_nonzero(cv2.Canny(sampled, 50, 150)) / sampled.size)

    if color_delta > 14.0:
        raise gr.Error(
            "This image does not look like a chest X-ray. Please upload a grayscale chest radiograph."
        )

    if contrast < 28.0 or edge_ratio < 0.01:
        raise gr.Error("This image is not a usable chest X-ray. Please upload a clearer chest radiograph.")

    if dark_ratio < 0.10 or bright_ratio < 0.05:
        raise gr.Error("This image does not match the expected intensity profile of a chest X-ray.")

    return gray


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
        probability_normal = float(values[1])

    return min(max(probability_normal, 0.0), 1.0)


def classify(probability_normal: float) -> tuple[int, str]:
    predicted_class = 1 if probability_normal >= THRESHOLD else 0
    label = "Normal appearance" if predicted_class == 1 else "Pneumonia suspicion"
    return predicted_class, label


def confidence_label(confidence: float) -> str:
    if confidence >= 0.9:
        return "High"
    if confidence >= 0.75:
        return "Moderate"
    return "Low"


def triage_label(predicted_class: int, confidence: float) -> str:
    if predicted_class == 1:
        return "Routine review"
    if confidence >= 0.9:
        return "Priority review"
    return "Needs review"


def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise RuntimeError("Grad-CAM requires a model with at least one Conv2D layer.")


def build_gradcam_model(model: tf.keras.Model, layer_name: str) -> tf.keras.Model:
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


def generate_gradcam_overlay(grayscale: np.ndarray, target_class: int | None = None) -> tuple[np.ndarray, str]:
    model = get_model()
    preprocessed = preprocess_grayscale_image(grayscale)
    layer_name = GRADCAM_LAYER or get_last_conv_layer_name(model)
    grad_model = get_gradcam_model(layer_name)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(preprocessed, training=False)
        probability_normal = extract_probability_normal(predictions)
        inferred_class, _ = classify(probability_normal)
        class_index = inferred_class if target_class is None else target_class
        flattened = tf.reshape(predictions, (-1,))

        if class_index == 1:
            class_score = flattened[-1]
        elif class_index == 0:
            class_score = 1.0 - flattened[-1]
        else:
            raise gr.Error("target_class must be 0 for Pneumonia or 1 for Normal.")

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
    overlay_bgr = cv2.addWeighted(base_image, 0.62, heatmap_color, 0.38, 0)

    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB), layer_name


def summarize_result(label: str, predicted_class: int, confidence: float) -> str:
    if predicted_class == 1:
        return (
            f"Model output: {label}. Confidence is {confidence_label(confidence).lower()} "
            "for a chest X-ray with no pneumonia pattern flagged by the model."
        )

    return (
        f"Model output: {label}. Confidence is {confidence_label(confidence).lower()}, "
        "so this image should receive clinical review."
    )


def analyze_image(image: np.ndarray | None) -> tuple[str, str, str, str, str, np.ndarray | None, np.ndarray | None]:
    if image is None:
        raise gr.Error("Please upload a chest X-ray image first.")

    image_bgr = to_bgr(image)
    grayscale = validate_xray_candidate(image_bgr)
    preprocessed = preprocess_grayscale_image(grayscale)
    prediction = get_model().predict(preprocessed, verbose=0)
    probability_normal = extract_probability_normal(prediction)
    predicted_class, label = classify(probability_normal)
    confidence = max(probability_normal, 1.0 - probability_normal)
    original_preview = cv2.cvtColor(cv2.resize(grayscale, IMAGE_SIZE), cv2.COLOR_GRAY2RGB)
    gradcam_overlay, layer_name = generate_gradcam_overlay(grayscale, target_class=predicted_class)

    suspicion_score = 1.0 - probability_normal

    summary = summarize_result(label, predicted_class, confidence)
    confidence_text = f"{confidence * 100:.1f}% ({confidence_label(confidence)})"
    suspicion_text = f"{suspicion_score * 100:.1f}%"
    triage_text = triage_label(predicted_class, confidence)
    label_text = f"{label} (threshold {THRESHOLD:.2f})"
    return (
        summary,
        label_text,
        suspicion_text,
        confidence_text,
        triage_text,
        original_preview,
        gradcam_overlay,
    )


with gr.Blocks(theme=gr.themes.Soft(), title="Pneumonia AI Screening") as demo:
    gr.Markdown(
        """
        # Pneumonia AI Screening
        Educational chest X-ray screening demo based on the CNN model from the project notebook.

        This tool accepts an image, checks whether it looks like a usable chest radiograph,
        runs a binary pneumonia screening prediction, and shows a Grad-CAM attention map.
        """
    )

    with gr.Accordion("How to use this demo", open=False):
        gr.Markdown(
            """
            1. Upload a chest X-ray image.
            2. Click **Run analysis**.
            3. Review the prediction summary and the Grad-CAM attention map.

            Supported images should look like frontal grayscale chest radiographs.
            Non X-ray images are rejected with a clear message.
            """
        )

    with gr.Row():
        input_image = gr.Image(
            label="Chest X-ray input",
            type="numpy",
            image_mode="RGB",
            sources=["upload"],
        )

        with gr.Column():
            summary_output = gr.Textbox(label="Clinical-style summary", lines=3)
            label_output = gr.Textbox(label="Model decision")
            suspicion_output = gr.Textbox(label="Pneumonia suspicion")
            confidence_output = gr.Textbox(label="Model confidence")
            triage_output = gr.Textbox(label="Review priority")

    analyze_button = gr.Button("Run analysis", variant="primary")

    with gr.Row():
        original_output = gr.Image(label="Processed X-ray preview")
        gradcam_output = gr.Image(label="Grad-CAM attention map")

    gr.Markdown(
        """
        **Disclaimer:** This demo was developed for educational purposes only.
        It does not provide a medical diagnosis and does not replace a radiologist or physician.

        Technical defaults:
        - image size: `150 x 150`
        - threshold: `0.45`
        - class mapping: `0 = Pneumonia`, `1 = Normal`
        """
    )

    analyze_button.click(
        fn=analyze_image,
        inputs=input_image,
        outputs=[
            summary_output,
            label_output,
            suspicion_output,
            confidence_output,
            triage_output,
            original_output,
            gradcam_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()
