# Pneumonia AI Screening

## Project Name

Pneumonia AI Screening

## Description

Pneumonia AI Screening is a study-based full-stack project for pneumonia detection from chest X-ray images.  
It combines a training notebook, a Flask backend API, a Next.js frontend, and Grad-CAM visualization to create an end-to-end prototype for image upload, prediction, and result review.

The project is based on a chest X-ray classification study using a CNN and a GAN-based rebalancing strategy for the minority class (`Normal`).

## Features

- Binary classification of chest X-ray images:
  - `0 -> Pneumonia`
  - `1 -> Normal`
- CNN-based image classification pipeline
- GAN-based generation of additional `Normal` images to reduce class imbalance
- Flask API for prediction and Grad-CAM heatmaps
- Next.js frontend for image upload and result display
- Threshold-based final prediction logic
- Hugging Face Space demo option

Study highlights:

- Dataset: `paultimothymooney/chest-xray-pneumonia`
- CNN input size: `150 x 150`
- GAN image size: `128 x 128`
- Rebalanced train set after GAN:
  - `3099` Pneumonia
  - `1502` Normal

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Flask
- Next.js 14
- React
- TypeScript
- KaggleHub

## Installation

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

### Frontend

```powershell
cd frontend
copy .env.local.example .env.local
npm install
```

## Usage

### Run the backend

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python app.py
```

Default backend URL:

```text
http://127.0.0.1:5000
```

### Run the frontend

```powershell
cd frontend
npm run dev
```

Default frontend URL:

```text
http://localhost:3000
```

### Model behavior

The backend:

- decodes the uploaded image,
- validates that it looks like a chest X-ray,
- converts it to grayscale,
- resizes it to `150 x 150`,
- normalizes it to `[0, 1]`,
- predicts the probability of class `1` (`Normal`),
- returns the final class label using a configurable threshold.

Important note:

- The latest study notebook reported a best threshold of `0.41`
- The backend example file currently uses `PREDICTION_THRESHOLD=0.45`

If you want the web app to match the latest study result exactly, update:

```text
backend/.env
```

to:

```text
PREDICTION_THRESHOLD=0.41
```

## Project Structure

```text
pneumonia-ai-workspace/
  backend/
    app.py
    requirements.txt
    .env.example
    pneumonia_cnn_model.keras
  frontend/
    app/
      api/
        predict/
          route.ts
        gradcam/
          route.ts
      globals.css
      layout.tsx
      page.tsx
    lib/
      types.ts
    package.json
    .env.local.example
  hf_space/
    app.py
    requirements.txt
  xray_pneumonia.ipynb
  README.md
```

## Example Result

Latest reported study results:

| Metric | Value |
| --- | ---: |
| Accuracy | `94.07%` |
| Balanced Accuracy | `0.9261` |
| Macro F1 | `0.9353` |
| Weighted F1 | `0.9400` |
| AUC | `0.9622` |
| Best Threshold | `0.41` |
| Global Score | `0.9355` |

Example API response:

```json
{
  "label": "Normal",
  "predicted_class": 1,
  "probability_normal": 0.92,
  "threshold": 0.41
}
```

Per-class performance:

- Pneumonia:
  - Precision: `0.93`
  - Recall: `0.98`
  - F1-score: `0.95`
- Normal:
  - Precision: `0.97`
  - Recall: `0.87`
  - F1-score: `0.92`

## Contribution

Contributions are welcome for:

- model improvements,
- frontend improvements,
- backend API enhancements,
- Grad-CAM visualization improvements,
- documentation cleanup.

If you want to contribute, create a new branch, make your changes, and open a pull request.

## License

No formal open-source license file has been added yet.  
At the moment, this project should be considered an educational and portfolio project.

## Author

Yanis Hamek
