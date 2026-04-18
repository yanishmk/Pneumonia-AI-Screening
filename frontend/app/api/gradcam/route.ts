import { proxyImageUpload } from "@/lib/server-proxy";
import type { GradcamResult } from "@/lib/types";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(request: Request) {
  return proxyImageUpload<GradcamResult>(request, "/gradcam", "Grad-CAM generation failed.");
}
