import { NextResponse } from "next/server";
import type { ApiError, PredictionResult } from "@/lib/types";

const FLASK_API_URL = process.env.FLASK_API_URL ?? "http://127.0.0.1:5000";
const MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024;
const RETRY_DELAY_MS = 6000;

export const runtime = "nodejs";
export const maxDuration = 60;

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithRetry(url: string, init: RequestInit) {
  try {
    return await fetch(url, init);
  } catch (firstError) {
    await sleep(RETRY_DELAY_MS);

    try {
      return await fetch(url, init);
    } catch {
      throw firstError;
    }
  }
}

async function readApiPayload<T>(response: Response): Promise<T | ApiError> {
  const raw = await response.text();

  if (!raw.trim()) {
    return { error: `Empty response from backend (${response.status} ${response.statusText}).` };
  }

  try {
    return JSON.parse(raw) as T | ApiError;
  } catch {
    return { error: `Backend returned a non-JSON response (${response.status} ${response.statusText}).` };
  }
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get("file");

    if (!(file instanceof File)) {
      return NextResponse.json<ApiError>({ error: "Missing image file." }, { status: 400 });
    }

    if (file.size === 0) {
      return NextResponse.json<ApiError>(
        { error: "The uploaded file is empty." },
        { status: 400 }
      );
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      return NextResponse.json<ApiError>(
        { error: "File is too large. Please upload an image under 8 MB." },
        { status: 400 }
      );
    }

    const forwardFormData = new FormData();
    forwardFormData.append("file", file, file.name);

    const response = await fetchWithRetry(`${FLASK_API_URL}/predict`, {
      method: "POST",
      body: forwardFormData
    });

    const payload = await readApiPayload<PredictionResult>(response);

    if (!response.ok) {
      return NextResponse.json<ApiError>(
        { error: "error" in payload ? payload.error : "Prediction failed." },
        { status: response.status }
      );
    }

    return NextResponse.json<PredictionResult>(payload as PredictionResult);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected prediction error.";

    return NextResponse.json<ApiError>(
      { error: `Could not reach the prediction service. ${message}` },
      { status: 502 }
    );
  }
}
