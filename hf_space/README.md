---
title: Pneumonia AI Screening
emoji: 🩻
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.24.0
app_file: app.py
pinned: false
python_version: 3.11
---

# Pneumonia AI Screening

This folder is a Hugging Face Spaces-ready Gradio version of the project.

## What to upload to a Space repo

Copy these files into the root of your Hugging Face Space:

- `app.py`
- `requirements.txt`
- `pneumonia_cnn_model.keras`
- `pneumonia_threshold.json` (optional)

## Notes

- The app uses the notebook model with grayscale preprocessing at `150x150`.
- Default threshold: `0.45`
- Class mapping:
  - `0 = Pneumonia`
  - `1 = Normal`
- The model is loaded lazily on first analysis to reduce startup pressure.
- Grad-CAM helper models are cached after the first run for faster repeated analysis.
- Basic X-ray validation rejects images that are too small, too large, too colorful, or badly framed.
- If `PREDICTION_THRESHOLD` is not set, the app can load a threshold from `pneumonia_threshold.json`.

## Optional environment variables

- `MODEL_PATH`
- `THRESHOLD_PATH`
- `MODEL_IMAGE_SIZE`
- `PREDICTION_THRESHOLD`
- `GRADCAM_LAYER`
- `MAX_IMAGE_PIXELS`
- `MIN_IMAGE_DIMENSION`

## Disclaimer

Educational use only. This demo is not a medical diagnosis system.
