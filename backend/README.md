# Pneumonia Detection Flask Backend

Place your trained model file here:

```text
backend/pneumonia_cnn_model.keras
```

Or set `MODEL_PATH` to an absolute or relative path.

You can also provide an optional threshold file:

```text
backend/pneumonia_threshold.json
```

Expected format:

```json
{
  "threshold": 0.41
}
```

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
- `GET /`
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

## Railway Deployment

Railway can deploy this backend directly from GitHub.

Recommended settings:

- Root Directory: `backend`
- Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1`
- Healthcheck Path: `/health`

Recommended environment variables:

- `MODEL_PATH=pneumonia_cnn_model.keras`
- `THRESHOLD_PATH=pneumonia_threshold.json`
- `MODEL_IMAGE_SIZE=150`
- `PREDICTION_THRESHOLD=0.45`
- `FLASK_DEBUG=0`

Notes:

- Railway injects the `PORT` variable automatically.
- The model now loads lazily on the first prediction request to reduce startup pressure.
- If `PREDICTION_THRESHOLD` is not set, the backend will try to read the threshold from `THRESHOLD_PATH`.
