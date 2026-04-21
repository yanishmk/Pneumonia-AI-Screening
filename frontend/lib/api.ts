import type { ApiError } from "@/lib/types";

export const FLASK_API_URL = process.env.FLASK_API_URL ?? "http://127.0.0.1:5000";
export const PUBLIC_BACKEND_URL = process.env.NEXT_PUBLIC_FLASK_API_URL?.trim().replace(/\/$/, "");
export const MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024;
export const MAX_FILE_SIZE_LABEL = "8 MB";
export const RETRY_DELAY_MS = 6000;
export const ACCEPTED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"] as const;

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function fetchWithRetry(url: string, init: RequestInit) {
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

export async function readApiPayload<T>(response: Response): Promise<T | ApiError> {
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

export function extractUploadedFile(formData: FormData): File | null {
  const file = formData.get("file");
  return file instanceof File ? file : null;
}

export function extractOptionalTextField(formData: FormData, fieldName: string): string | null {
  const value = formData.get(fieldName);

  if (typeof value !== "string") {
    return null;
  }

  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

export function buildImageFormData(file: File, extraFields?: Record<string, string>) {
  const formData = new FormData();
  formData.append("file", file, file.name);

  if (extraFields) {
    for (const [key, value] of Object.entries(extraFields)) {
      formData.append(key, value);
    }
  }

  return formData;
}

export function isAcceptedImageFile(file: File) {
  if (file.type) {
    return ACCEPTED_IMAGE_TYPES.includes(file.type as (typeof ACCEPTED_IMAGE_TYPES)[number]);
  }

  return /\.(png|jpe?g|webp)$/i.test(file.name);
}
