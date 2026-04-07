# Pneumonia Detection Flask Backend

Place your trained model file here:

```text
backend/pneumonia_cnn_model.keras
```

Or set `MODEL_PATH` to an absolute or relative path.

For this project, the expected model is the one exported from your notebook
`xray_pneumonia_final_(4) (2).ipynb`.

## Setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

The API runs on `http://localhost:5000`.

Notebook model assumptions:

- input image: grayscale `150x150`
- normalization: `[0, 1]`
- class mapping: `0 = Pneumonia`, `1 = Normal`
- default threshold: `0.45`

## Endpoints

- `GET /health`
- `POST /predict`
- `POST /gradcam`

`POST /predict` accepts `multipart/form-data` with a `file` field and returns:

```json
{
  "label": "Normal",
  "predicted_class": 1,
  "probability_normal": 0.92,
  "threshold": 0.45
}
```

`POST /gradcam` accepts the same `file` field and returns a base64 PNG overlay. It auto-detects the last `Conv2D` layer, or you can set `GRADCAM_LAYER`.
