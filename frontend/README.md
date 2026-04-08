# Pneumonia Detection Next.js Frontend

The frontend can call Flask directly through `NEXT_PUBLIC_FLASK_API_URL`.
If this variable is not set, it falls back to the internal Next.js proxy routes in `app/api/`.

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
NEXT_PUBLIC_FLASK_API_URL=http://127.0.0.1:5000
```
