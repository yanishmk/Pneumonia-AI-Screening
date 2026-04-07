# Pneumonia Detection Next.js Frontend

The frontend sends uploads to `app/api/predict/route.ts`, which forwards them to Flask.

## Setup

```powershell
cd frontend
copy .env.local.example .env.local
npm install
npm run dev
```

By default, the frontend expects Flask at:

```text
http://127.0.0.1:5000
```

To change it, edit `.env.local`:

```text
FLASK_API_URL=http://127.0.0.1:5000
```
