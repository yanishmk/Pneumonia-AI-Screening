import { proxyImageUpload } from "@/lib/server-proxy";
import type { PredictionResult } from "@/lib/types";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(request: Request) {
  return proxyImageUpload<PredictionResult>(request, "/predict", "Prediction failed.");
}
