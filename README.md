# Pneumonia Detection AI Web App
nb
Full-stack prototype for pneumonia detection from chest X-ray images.

The reference model for this workspace is the CNN from notebook:
`xray_pneumonia_final

## Folder Structure

```text
pneumonia-ai-workspace/
  backend/
    app.py
    requirements.txt
    .env.example
    README.md
    pneumonia_cnn_model.keras  
  frontend/
    app/
      api/
        predict/
          route.ts
      globals.css
      layout.tsx
      page.tsx
    lib/
      types.ts
    package.json
    tsconfig.json
    next.config.mjs
    .env.local.example
    README.md
```

## How It Works

1. The user uploads a chest X-ray in the Next.js UI.
2. `frontend/app/api/predict/route.ts` forwards the uploaded file to Flask.
3. Flask preprocesses the image as grayscale, `150x150`, normalized to `[0, 1]`, shaped as `(1, 150, 150, 1)`.
4. TensorFlow/Keras returns probability of class `1`, which is `Normal`.
5. The backend applies threshold `0.45`.

Class mapping:

```text
0 -> Pneumonia
1 -> Normal
```

## Run Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## Run Frontend

```powershell
cd frontend
copy .env.local.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`.

## API Response

```json
{
  "label": "Normal",
  "predicted_class": 1,
  "probability_normal": 0.92,
  "threshold": 0.45
}
```

## Optional Grad-CAM

The backend also includes `POST /gradcam`. It accepts the same `multipart/form-data` upload and returns `gradcam_image_base64`.

It auto-detects the final Keras `Conv2D` layer. If your model needs a specific layer, set:

```text
GRADCAM_LAYER=your_conv_layer_name
```



## Notes

This project is for educational prototyping only. It is not a medical diagnosis system.

## Deployment

Recommended setup:

- Frontend on Vercel
- Backend on Render

Alternative lightweight demo setup:

- `hf_space/` contains a Gradio version prepared for Hugging Face Spaces
- copy `hf_space/app.py`, `hf_space/requirements.txt`, and `backend/pneumonia_cnn_model.keras`
  into a Space repository to publish the model without the full Next.js + Flask stack

Suggested backend settings on Render:

- Root directory: `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Python version: `3.11.11` via `backend/.python-version`

Suggested frontend settings on Vercel:

- Root directory: `frontend`
- Environment variable: `FLASK_API_URL=https://your-render-backend.onrender.com`

Suggested backend settings on Railway:

- Root directory: `backend`
- Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1`
- Healthcheck path: `/health`
- Environment variables:
  - `MODEL_PATH=pneumonia_cnn_model.keras`
  - `MODEL_IMAGE_SIZE=150`
  - `PREDICTION_THRESHOLD=0.45`
  - `FLASK_DEBUG=0`
