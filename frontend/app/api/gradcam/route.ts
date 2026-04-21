import { NextResponse } from "next/server";
import {
  buildImageFormData,
  extractUploadedFile,
  fetchWithRetry,
  FLASK_API_URL,
  MAX_FILE_SIZE_BYTES,
  readApiPayload
} from "@/lib/api";
import type { ApiError, GradcamResult } from "@/lib/types";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = extractUploadedFile(formData);

    if (!file) {
      return NextResponse.json<ApiError>({ error: "Missing image file." }, { status: 400 });
    }

    if (file.size === 0) {
      return NextResponse.json<ApiError>({ error: "The uploaded file is empty." }, { status: 400 });
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      return NextResponse.json<ApiError>(
        { error: "File is too large. Please upload an image under 8 MB." },
        { status: 400 }
      );
    }

    const response = await fetchWithRetry(`${FLASK_API_URL}/gradcam`, {
      method: "POST",
      body: buildImageFormData(file)
    });

    const payload = await readApiPayload<GradcamResult>(response);

    if (!response.ok) {
      return NextResponse.json<ApiError>(
        { error: "error" in payload ? payload.error : "Grad-CAM generation failed." },
        { status: response.status }
      );
    }

    return NextResponse.json<GradcamResult>(payload as GradcamResult);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected Grad-CAM error.";

    return NextResponse.json<ApiError>(
      { error: `Could not reach the Grad-CAM service. ${message}` },
      { status: 502 }
    );
  }
}
