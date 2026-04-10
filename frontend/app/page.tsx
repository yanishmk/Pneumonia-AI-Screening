"use client";

import { ChangeEvent, DragEvent, FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import type { ApiError, GradcamResult, PredictionResult } from "@/lib/types";

const PUBLIC_BACKEND_URL = process.env.NEXT_PUBLIC_FLASK_API_URL?.trim().replace(/\/$/, "");

function formatPercent(value: number) {
  return new Intl.NumberFormat("en", {
    style: "percent",
    maximumFractionDigits: 2
  }).format(value);
}

function getDisplayedLabel(label: PredictionResult["label"]) {
  return label === "Normal" ? "Normal appearance" : "Pneumonia suspicion";
}

function getAlertLevel(probabilityPneumonia: number) {
  if (probabilityPneumonia >= 0.8) {
    return "High";
  }

  if (probabilityPneumonia >= 0.55) {
    return "Moderate";
  }

  return "Low";
}

function getInterpretation(label: PredictionResult["label"], probabilityPneumonia: number) {
  if (label === "Normal") {
    return "No strong pneumonia pattern was flagged by the model on this image.";
  }

  if (probabilityPneumonia >= 0.8) {
    return "The image shows strong signs that may be consistent with pneumonia and should be reviewed promptly.";
  }

  return "The image shows findings that may need clinical review for possible pneumonia.";
}

function escapePdfText(value: string) {
  return value.replace(/\\/g, "\\\\").replace(/\(/g, "\\(").replace(/\)/g, "\\)");
}

function buildPdfBlob(lines: string[]) {
  const escapedLines = lines.map((line) => `(${escapePdfText(line)}) Tj`);
  const content = `BT
/F1 12 Tf
40 800 Td
16 TL
${escapedLines.join("\nT*\n")}
ET`;

  const objects = [
    "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj",
    "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj",
    "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj",
    `4 0 obj\n<< /Length ${content.length} >>\nstream\n${content}\nendstream\nendobj`,
    "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj"
  ];

  let pdf = "%PDF-1.4\n";
  const offsets = [0];

  for (const object of objects) {
    offsets.push(pdf.length);
    pdf += `${object}\n`;
  }

  const xrefOffset = pdf.length;
  pdf += `xref
0 ${objects.length + 1}
0000000000 65535 f 
`;

  for (let index = 1; index < offsets.length; index += 1) {
    pdf += `${offsets[index].toString().padStart(10, "0")} 00000 n \n`;
  }

  pdf += `trailer
<< /Size ${objects.length + 1} /Root 1 0 R >>
startxref
${xrefOffset}
%%EOF`;

  return new Blob([pdf], { type: "application/pdf" });
}

export default function HomePage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isStudyInfoOpen, setIsStudyInfoOpen] = useState(false);
  const [isReportOpen, setIsReportOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [gradcam, setGradcam] = useState<GradcamResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [gradcamError, setGradcamError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const probabilityPneumonia = useMemo(() => {
    if (!prediction) {
      return 0;
    }

    return Math.max(0, Math.min(1, 1 - prediction.probability_normal));
  }, [prediction]);

  const modelConfidence = useMemo(() => {
    if (!prediction) {
      return 0;
    }

    return Math.max(prediction.probability_normal, probabilityPneumonia);
  }, [prediction, probabilityPneumonia]);

  const reportLines = useMemo(() => {
    if (!prediction) {
      return [];
    }

    const now = new Date();
    const formattedDate = now.toLocaleString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    });

    return [
      "Pneumonia AI Screening Report",
      `Generated: ${formattedDate}`,
      "",
      `Image file: ${file?.name ?? "Uploaded image"}`,
      `Result: ${getDisplayedLabel(prediction.label)}`,
      `Pneumonia suspicion: ${formatPercent(probabilityPneumonia)}`,
      `Confidence: ${formatPercent(modelConfidence)}`,
      `Attention level: ${getAlertLevel(probabilityPneumonia)}`,
      `Decision threshold: ${formatPercent(prediction.threshold)}`,
      "",
      `Interpretation: ${getInterpretation(prediction.label, probabilityPneumonia)}`,
      "",
      "Educational use only. This tool does not replace a radiologist or physician."
    ];
  }, [file?.name, modelConfidence, prediction, probabilityPneumonia]);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  function selectFile(nextFile: File | null) {
    setPrediction(null);
    setGradcam(null);
    setError(null);
    setGradcamError(null);
    setIsReportOpen(false);

    if (!nextFile) {
      setFile(null);
      return;
    }

    if (nextFile.size === 0) {
      setFile(null);
      setError("This file is empty. Please choose a valid image.");
      return;
    }

    setFile(nextFile);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    selectFile(event.target.files?.[0] ?? null);
    event.target.value = "";
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files?.[0] ?? null);
  }

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  function handleDropzoneKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openFilePicker();
    }
  }

  async function readClientPayload<T>(response: Response): Promise<T | ApiError> {
    const raw = await response.text();

    if (!raw.trim()) {
      return { error: `Empty response from frontend API (${response.status} ${response.statusText}).` };
    }

    try {
      return JSON.parse(raw) as T | ApiError;
    } catch {
      return { error: raw };
    }
  }

  async function postImage<T>(url: string, imageFile: File): Promise<T> {
    const formData = new FormData();
    formData.append("file", imageFile);

    const endpoint = PUBLIC_BACKEND_URL ? `${PUBLIC_BACKEND_URL}${url.replace("/api", "")}` : url;

    const response = await fetch(endpoint, {
      method: "POST",
      body: formData
    });

    const payload = await readClientPayload<T>(response);
    const apiError = payload as ApiError;

    if (!response.ok) {
      throw new Error(typeof apiError.error === "string" ? apiError.error : "Request failed.");
    }

    return payload as T;
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!file) {
      setError("Upload a chest X-ray image first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setGradcamError(null);
    setPrediction(null);
    setGradcam(null);
    setIsReportOpen(false);

    try {
      const predictionResult = await postImage<PredictionResult>("/api/predict", file);
      setPrediction(predictionResult);

      try {
        const gradcamResult = await postImage<GradcamResult>("/api/gradcam", file);
        setGradcam(gradcamResult);
      } catch (gradcamRequestError) {
        setGradcamError(
          gradcamRequestError instanceof Error ? gradcamRequestError.message : "Grad-CAM could not be generated."
        );
      }
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  }

  function handleDownloadPdf() {
    if (reportLines.length === 0) {
      return;
    }

    const blob = buildPdfBlob(reportLines);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "pneumonia-screening-report.pdf";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  const analysisHeadline = prediction ? getDisplayedLabel(prediction.label) : "Screening idle";
  const analysisNarrative = prediction
    ? getInterpretation(prediction.label, probabilityPneumonia)
    : "Upload a chest X-ray and launch the model.";

  return (
    <main className="pageShell">
      <section className="heroSection">
        <div className="heroCopy">
          <p className="eyebrow">Pneumonia Detection AI</p>
          <h1>Fast screening for chest X-rays.</h1>
          <p className="heroLead">Upload, analyze, and review the result in one clean workspace.</p>
        </div>

        <aside className="heroPanel" aria-label="Session overview">
          <div className="heroPanelTop">
            <div>
              <p className="sectionKicker">Session</p>
              <h2>{prediction ? "Analysis ready" : "Ready to analyze"}</h2>
            </div>
            <span className={`statusBadge ${prediction ? "statusBadgeLive" : ""}`}>
              {prediction ? "Live" : "Idle"}
            </span>
          </div>

          <p className="heroPanelText">
            {prediction ? "Prediction complete. You can review the result and export the report." : "Ready for a new analysis."}
          </p>

          <button
            type="button"
            className="aboutToggle"
            aria-expanded={isStudyInfoOpen}
            onClick={() => setIsStudyInfoOpen((current) => !current)}
          >
            <span>About</span>
            <span>{isStudyInfoOpen ? "Hide" : "Show"}</span>
          </button>

          {isStudyInfoOpen ? (
            <p className="aboutText">
              Educational tool for AI-assisted chest X-ray screening. It supports review and does not replace medical
              diagnosis.
            </p>
          ) : null}
        </aside>
      </section>

      <section className="workspaceSection" aria-label="Screening workspace">
        <form onSubmit={handleSubmit} className="workspaceGrid">
          <section className="uploadPanel">
            <div className="panelHeader">
              <div>
                <p className="sectionKicker">Input</p>
                <h2>Upload chest X-ray</h2>
              </div>
              <p className="panelMeta">PNG, JPG, JPEG</p>
            </div>

            <div className="uploadStage">
              <div
                className={`dropzone ${isDragging ? "dropzoneActive" : ""}`}
                role="button"
                tabIndex={0}
                onClick={openFilePicker}
                onKeyDown={handleDropzoneKeyDown}
                onDragOver={(event) => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
              >
                <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} />
                <span className="dropzoneIcon">+</span>
                <strong>{file ? file.name : "Drop image or click to browse"}</strong>
                <small>Best with frontal chest X-ray images.</small>

                <button
                  className="primaryButton"
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    openFilePicker();
                  }}
                >
                  Browse file
                </button>
              </div>

              <div className="previewStage" aria-label="Uploaded image preview">
                <div className="previewTopbar">
                  <span>Preview</span>
                  <span>{file ? "Loaded" : "Awaiting image"}</span>
                </div>

                <div className="previewFrame">
                  {previewUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={previewUrl} alt="Uploaded chest X-ray preview" className="previewImage" />
                  ) : (
                    <div className="previewPlaceholder">No image selected</div>
                  )}
                </div>
              </div>
            </div>

            <div className="actionBar">
              <button className="analyzeButton" type="submit" disabled={!file || isLoading}>
                {isLoading ? "Analyzing..." : "Run analysis"}
              </button>
              <p className="actionHint">{file ? "Image ready for screening." : "Select one image to start."}</p>
            </div>
          </section>

          <aside className="resultPanel" aria-label="Prediction output">
            <div className="panelHeader panelHeaderCompact">
              <div>
                <p className="sectionKicker">Output</p>
                <h2>{analysisHeadline}</h2>
              </div>
              {prediction ? (
                <span className={`resultTone ${prediction.label === "Normal" ? "resultToneNormal" : "resultToneAlert"}`}>
                  {getAlertLevel(probabilityPneumonia)}
                </span>
              ) : null}
            </div>

            <p className="resultNarrative">{analysisNarrative}</p>

            {prediction ? (
              <>
                <div className="scoreHero">
                  <div>
                    <span className="scoreLabel">Pneumonia suspicion</span>
                    <strong>{formatPercent(probabilityPneumonia)}</strong>
                  </div>
                </div>

                <div className="metricList">
                  <div className="metricRow">
                    <span>Confidence</span>
                    <strong>{formatPercent(modelConfidence)}</strong>
                  </div>
                  <div className="metricRow">
                    <span>Threshold</span>
                    <strong>{formatPercent(prediction.threshold)}</strong>
                  </div>
                  <div className="metricRow">
                    <span>Class</span>
                    <strong>{prediction.label}</strong>
                  </div>
                </div>

                <div
                  className="confidenceBar"
                  aria-label={`Pneumonia probability ${Math.round(probabilityPneumonia * 100)}%`}
                >
                  <span style={{ width: `${Math.round(probabilityPneumonia * 100)}%` }} />
                </div>

                <div className="reportTools">
                  <button
                    type="button"
                    className="secondaryButton"
                    onClick={() => setIsReportOpen((current) => !current)}
                  >
                    {isReportOpen ? "Hide report" : "Preview report"}
                  </button>
                  <button type="button" className="secondaryButton secondaryButtonStrong" onClick={handleDownloadPdf}>
                    Download PDF
                  </button>
                </div>

                {isReportOpen && reportLines.length > 0 ? (
                  <div className="reportPreview">
                    {reportLines.map((line, index) => (
                      <p key={`${line}-${index}`}>{line || "\u00A0"}</p>
                    ))}
                  </div>
                ) : null}
              </>
            ) : (
              <div className="emptyPanel">
                <span className="emptyPulse" aria-hidden="true" />
                <p>Prediction results will appear here after analysis.</p>
              </div>
            )}
          </aside>
        </form>

        {error ? (
          <div className="alertBanner" role="alert">
            {error}
          </div>
        ) : null}

        <section className="visualSection" aria-label="Visual explanation">
          <div className="panelHeader">
            <div>
              <p className="sectionKicker">Grad-CAM</p>
              <h2>Model focus map</h2>
            </div>
            <p className="panelMeta">Original image and AI attention overlay</p>
          </div>

          {prediction ? (
            gradcam ? (
              <>
                <div className="visualGrid">
                  <figure className="visualCard">
                    <figcaption>Original</figcaption>
                    {previewUrl ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={previewUrl} alt="Original uploaded chest X-ray" className="visualImage" />
                    ) : (
                      <div className="visualEmpty">Unavailable</div>
                    )}
                  </figure>

                  <figure className="visualCard">
                    <figcaption>Grad-CAM</figcaption>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`data:image/png;base64,${gradcam.gradcam_image_base64}`}
                      alt="Grad-CAM heatmap overlay"
                      className="visualImage"
                    />
                  </figure>
                </div>

                <div className="visualNotes">
                  <p>Warmer regions indicate stronger influence on the prediction.</p>
                </div>
              </>
            ) : (
              <div className="visualEmpty">
                {gradcamError ? gradcamError : "Generating visual explanation..."}
              </div>
            )
          ) : (
            <div className="visualEmpty">Run an analysis to unlock the visual explanation.</div>
          )}
        </section>
      </section>

      <p className="footerNote">Educational use only. This tool does not replace a doctor.</p>
    </main>
  );
}
