from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app import classify, extract_probability_normal, get_model, preprocess_grayscale_image, validate_xray_candidate


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CLASS_TO_INDEX = {"pneumonia": 0, "normal": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the pneumonia CNN on a dataset with 'pneumonia' and 'normal' subfolders."
    )
    parser.add_argument("dataset_dir", type=Path, help="Path to the evaluation dataset root directory.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the evaluation summary as JSON.",
    )
    return parser.parse_args()


def iter_image_paths(dataset_dir: Path) -> list[tuple[Path, int]]:
    image_entries: list[tuple[Path, int]] = []

    for class_name, class_index in CLASS_TO_INDEX.items():
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing expected class directory: '{class_dir}'.")

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
                image_entries.append((path, class_index))

    if not image_entries:
        raise FileNotFoundError("No supported evaluation images were found in the dataset directory.")

    return image_entries


def load_image_for_model(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError("Could not decode image.")

    grayscale = validate_xray_candidate(image_bgr)
    return preprocess_grayscale_image(grayscale)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)

    accuracy = safe_divide(tp + tn, len(y_true))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "accuracy": accuracy,
        "precision_pneumonia": precision,
        "recall_pneumonia": recall,
        "f1_pneumonia": f1,
        "confusion_matrix": {
            "true_pneumonia_pred_pneumonia": tp,
            "true_pneumonia_pred_normal": fn,
            "true_normal_pred_pneumonia": fp,
            "true_normal_pred_normal": tn,
        },
    }


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    args = parse_args()
    image_entries = iter_image_paths(args.dataset_dir)
    model = get_model()

    y_true: list[int] = []
    y_pred: list[int] = []
    skipped: list[dict[str, str]] = []
    inconclusive = 0

    for path, expected_class in image_entries:
        try:
            image = load_image_for_model(path)
            prediction = model.predict(image, verbose=0)
            probability_normal = extract_probability_normal(prediction)
            predicted_class, _, is_inconclusive = classify(probability_normal)

            if is_inconclusive:
                inconclusive += 1
                predicted_class = 1 if probability_normal >= 0.5 else 0

            y_true.append(expected_class)
            y_pred.append(predicted_class)
        except Exception as exc:
            skipped.append({"path": str(path), "error": str(exc)})

    if not y_true:
        raise RuntimeError("No valid evaluation samples were processed.")

    metrics = compute_metrics(y_true, y_pred)
    summary = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "total_images_found": len(image_entries),
        "processed_images": len(y_true),
        "skipped_images": len(skipped),
        "inconclusive_predictions": inconclusive,
        **metrics,
    }

    print("Evaluation summary")
    print(f"- Dataset: {summary['dataset_dir']}")
    print(f"- Processed images: {summary['processed_images']} / {summary['total_images_found']}")
    print(f"- Skipped images: {summary['skipped_images']}")
    print(f"- Inconclusive predictions: {summary['inconclusive_predictions']}")
    print(f"- Accuracy: {format_percent(summary['accuracy'])}")
    print(f"- Precision (Pneumonia): {format_percent(summary['precision_pneumonia'])}")
    print(f"- Recall (Pneumonia): {format_percent(summary['recall_pneumonia'])}")
    print(f"- F1 (Pneumonia): {format_percent(summary['f1_pneumonia'])}")
    print("- Confusion matrix:")
    for key, value in summary["confusion_matrix"].items():
        print(f"  - {key}: {value}")

    if skipped:
        print("- Skipped files:")
        for item in skipped[:10]:
            print(f"  - {item['path']}: {item['error']}")
        if len(skipped) > 10:
            print(f"  - ... {len(skipped) - 10} more")

    if args.output_json:
        payload = {**summary, "skipped": skipped}
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"- JSON report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
