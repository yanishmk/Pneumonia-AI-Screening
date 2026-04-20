"use client";

import { ChangeEvent, DragEvent, FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import type { ApiError, GradcamResult, PredictionResult } from "@/lib/types";

const PUBLIC_BACKEND_URL = process.env.NEXT_PUBLIC_FLASK_API_URL?.trim().replace(/\/$/, "");
const ACCEPTED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const ACCEPTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"];

type PatientInfo = {
  firstName: string;
  lastName: string;
  age: string;
};

type ReportRow = {
  label: string;
  value: string;
};

type ReportSection = {
  title: string;
  rows: ReportRow[];
};

type ReportData = {
  title: string;
  generatedAt: string;
  patientLabel: string;
  imageLabel: string;
  summary: string;
  alertLevel: string;
  sections: ReportSection[];
  interpretation: string;
  disclaimer: string;
};

function formatPercent(value: number) {
  return new Intl.NumberFormat("en", {
    style: "percent",
    maximumFractionDigits: 2
  }).format(value);
}

function formatReportDate(date: Date) {
  return date.toLocaleString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function getDisplayedLabel(label: PredictionResult["label"]) {
  if (label === "Normal") {
    return "Normal appearance";
  }

  if (label === "Pneumonia") {
    return "Pneumonia suspicion";
  }

  return "Inconclusive result";
}

function getAlertLevel(probabilityPneumonia: number, isInconclusive: boolean) {
  if (isInconclusive) {
    return "Review";
  }

  if (probabilityPneumonia >= 0.8) {
    return "High";
  }

  if (probabilityPneumonia >= 0.55) {
    return "Moderate";
  }

  return "Low";
}

function getInterpretation(prediction: PredictionResult, probabilityPneumonia: number) {
  if (prediction.is_inconclusive) {
    return "The model score is too close to the decision threshold to support a confident screening label. Clinical review is recommended.";
  }

  const { label } = prediction;

  if (label === "Normal") {
    return "No strong pneumonia pattern was flagged by the model on this image.";
  }

  if (probabilityPneumonia >= 0.8) {
    return "The image shows strong signs that may be consistent with pneumonia and should be reviewed promptly.";
  }

  return "The image shows findings that may need clinical review for possible pneumonia.";
}

function getFilledValue(value: string) {
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : "Not provided";
}

function getPdfSafeValue(value: string) {
  return getFilledValue(value).normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

function getPatientLabel(patientInfo: PatientInfo) {
  const name = [patientInfo.firstName.trim(), patientInfo.lastName.trim()].filter(Boolean).join(" ");
  const age = patientInfo.age.trim();
  const parts = [name, age ? `${age} years` : ""].filter(Boolean);

  return parts.length > 0 ? parts.join(" | ") : "Patient details pending";
}

function escapePdfText(value: string) {
  return value.replace(/\\/g, "\\\\").replace(/\(/g, "\\(").replace(/\)/g, "\\)");
}

function isSupportedImageFile(file: File) {
  if (file.type && ACCEPTED_IMAGE_TYPES.has(file.type.toLowerCase())) {
    return true;
  }

  const lowerName = file.name.toLowerCase();
  return ACCEPTED_IMAGE_EXTENSIONS.some((extension) => lowerName.endsWith(extension));
}

function wrapPdfText(value: string, maxChars: number) {
  const sanitized = value.trim();

  if (!sanitized) {
    return [""];
  }

  const words = sanitized.split(/\s+/);
  const lines: string[] = [];
  let currentLine = "";

  for (const word of words) {
    const candidate = currentLine ? `${currentLine} ${word}` : word;

    if (candidate.length <= maxChars) {
      currentLine = candidate;
      continue;
    }

    if (currentLine) {
      lines.push(currentLine);
      currentLine = "";
    }

    if (word.length <= maxChars) {
      currentLine = word;
      continue;
    }

    for (let index = 0; index < word.length; index += maxChars) {
      lines.push(word.slice(index, index + maxChars));
    }
  }

  if (currentLine) {
    lines.push(currentLine);
  }

  return lines;
}

function buildPdfBlob(report: ReportData) {
  const commands: string[] = [];
  const marginX = 46;
  const contentWidth = 503;

  const pushText = (text: string, options: { x?: number; y?: number; size?: number; font?: string; color?: [number, number, number] } = {}) => {
    const { x = marginX, y = 780, size = 12, font = "F1", color = [0.07, 0.11, 0.13] } = options;
    const [red, green, blue] = color;
    commands.push(`BT /${font} ${size} Tf ${red} ${green} ${blue} rg 1 0 0 1 ${x} ${y} Tm (${escapePdfText(text)}) Tj ET`);
  };

  const pushRule = (fromX: number, toX: number, y: number, color = [0.82, 0.88, 0.89], width = 1) => {
    const [red, green, blue] = color;
    commands.push(`${width} w ${red} ${green} ${blue} RG ${fromX} ${y} m ${toX} ${y} l S`);
  };

  const pushFilledRect = (x: number, y: number, width: number, height: number, color: [number, number, number]) => {
    const [red, green, blue] = color;
    commands.push(`${red} ${green} ${blue} rg ${x} ${y} ${width} ${height} re f`);
  };

  const pushStrokedRect = (
    x: number,
    y: number,
    width: number,
    height: number,
    color: [number, number, number] = [0.84, 0.89, 0.9],
    lineWidth = 1
  ) => {
    const [red, green, blue] = color;
    commands.push(`${lineWidth} w ${red} ${green} ${blue} RG ${x} ${y} ${width} ${height} re S`);
  };

  const pushWrappedText = (
    text: string,
    options: { x?: number; y: number; maxChars: number; size?: number; font?: string; color?: [number, number, number]; lineGap?: number }
  ) => {
    const { x = marginX, y, maxChars, size = 12, font = "F1", color = [0.07, 0.11, 0.13], lineGap = 14 } = options;
    const lines = wrapPdfText(text, maxChars);

    lines.forEach((line, index) => {
      pushText(line, { x, y: y - index * lineGap, size, font, color });
    });

    return lines.length;
  };

  const pushCardTitle = (title: string, y: number) => {
    pushText(title.toUpperCase(), { x: marginX + 18, y, size: 9.5, font: "F2", color: [0.06, 0.46, 0.43] });
  };

  pushFilledRect(marginX, 724, contentWidth, 84, [0.08, 0.25, 0.3]);
  pushText(report.title, { x: marginX + 18, y: 780, size: 22, font: "F2", color: [1, 1, 1] });
  pushText(`Generated ${report.generatedAt}`, { x: marginX + 18, y: 758, size: 10.5, color: [0.86, 0.92, 0.93] });
  pushWrappedText(report.patientLabel, {
    x: marginX + 18,
    y: 740,
    maxChars: 62,
    size: 11,
    font: "F2",
    color: [0.9, 0.96, 0.96],
    lineGap: 12
  });

  pushFilledRect(marginX, 642, contentWidth, 58, [0.93, 0.97, 0.97]);
  pushStrokedRect(marginX, 642, contentWidth, 58, [0.82, 0.89, 0.9]);
  pushText("AI SUMMARY", { x: marginX + 18, y: 682, size: 9.5, font: "F2", color: [0.06, 0.46, 0.43] });
  pushText(report.summary, { x: marginX + 18, y: 660, size: 18, font: "F2", color: [0.05, 0.08, 0.1] });
  pushFilledRect(438, 656, 92, 26, report.alertLevel === "High" ? [0.99, 0.92, 0.88] : report.alertLevel === "Moderate" ? [0.99, 0.96, 0.88] : [0.9, 0.97, 0.93]);
  pushText(`${report.alertLevel} alert`, {
    x: 458,
    y: 665,
    size: 10,
    font: "F2",
    color: report.alertLevel === "High" ? [0.74, 0.28, 0.08] : report.alertLevel === "Moderate" ? [0.64, 0.47, 0.06] : [0.08, 0.45, 0.31]
  });

  pushFilledRect(marginX, 534, contentWidth, 88, [1, 1, 1]);
  pushStrokedRect(marginX, 534, contentWidth, 88, [0.84, 0.89, 0.9]);
  pushCardTitle("Patient information", 600);
  pushText("First name", { x: marginX + 18, y: 574, size: 10, font: "F2", color: [0.4, 0.47, 0.5] });
  pushText(report.sections[0]?.rows[0]?.value ?? "Not provided", { x: marginX + 120, y: 574, size: 11.5, font: "F2" });
  pushText("Last name", { x: marginX + 18, y: 554, size: 10, font: "F2", color: [0.4, 0.47, 0.5] });
  pushText(report.sections[0]?.rows[1]?.value ?? "Not provided", { x: marginX + 120, y: 554, size: 11.5, font: "F2" });
  pushText("Age", { x: 330, y: 574, size: 10, font: "F2", color: [0.4, 0.47, 0.5] });
  pushText(report.sections[0]?.rows[2]?.value ?? "Not provided", { x: 368, y: 574, size: 11.5, font: "F2" });

  pushFilledRect(marginX, 372, contentWidth, 140, [1, 1, 1]);
  pushStrokedRect(marginX, 372, contentWidth, 140, [0.84, 0.89, 0.9]);
  pushCardTitle("Exam summary", 490);
  pushText("Image file", { x: marginX + 18, y: 464, size: 10, font: "F2", color: [0.4, 0.47, 0.5] });
  const imageLineCount = pushWrappedText(report.imageLabel, {
    x: marginX + 110,
    y: 464,
    maxChars: 50,
    size: 11,
    font: "F2",
    color: [0.07, 0.11, 0.13],
    lineGap: 13
  });
  const metricsStartY = 464 - Math.max(1, imageLineCount) * 13 - 12;
  pushText("Result", { x: marginX + 18, y: metricsStartY, size: 10, font: "F2", color: [0.4, 0.47, 0.5] });
  pushText(report.sections[1]?.rows[1]?.value ?? report.summary, { x: marginX + 110, y: metricsStartY, size: 11.5, font: "F2" });
  pushText("Suspicion", { x: marginX + 18, y: metricsStartY - 20, size: 10, font: "F2", color: [0.4, 0.47, 0.5] });
  pushText(report.sections[1]?.rows[2]?.value ?? "N/A", { x: marginX + 110, y: metricsStartY - 20, size: 11.5, font: "F2" });
  pushText("Confidence", { x: 330, y: metricsStartY - 20, size: 10, font: "F2", color: [0.4, 0.47, 0.5] });
  pushText(report.sections[1]?.rows[3]?.value ?? "N/A", { x: 392, y: metricsStartY - 20, size: 11.5, font: "F2" });
  pushText("Attention level", { x: marginX + 18, y: metricsStartY - 40, size: 10, font: "F2", color: [0.4, 0.47, 0.5] });
  pushText(report.sections[1]?.rows[4]?.value ?? report.alertLevel, { x: marginX + 110, y: metricsStartY - 40, size: 11.5, font: "F2" });

  pushFilledRect(marginX, 236, contentWidth, 112, [0.98, 0.99, 0.99]);
  pushStrokedRect(marginX, 236, contentWidth, 112, [0.84, 0.89, 0.9]);
  pushCardTitle("Interpretation", 326);
  pushWrappedText(report.interpretation, {
    x: marginX + 18,
    y: 298,
    maxChars: 78,
    size: 11.5,
    color: [0.07, 0.11, 0.13],
    lineGap: 15
  });

  pushRule(marginX, marginX + contentWidth, 92, [0.84, 0.89, 0.9]);
  pushWrappedText(report.disclaimer, {
    x: marginX,
    y: 74,
    maxChars: 86,
    size: 9.5,
    color: [0.4, 0.47, 0.5],
    lineGap: 12
  });

  const content = commands.join("\n");

  const objects = [
    "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj",
    "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj",
    "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>\nendobj",
    `4 0 obj\n<< /Length ${content.length} >>\nstream\n${content}\nendstream\nendobj`,
    "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj",
    "6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj"
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
  const [isReportOpen, setIsReportOpen] = useState(false);
  const [hasAnalysisRun, setHasAnalysisRun] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [gradcam, setGradcam] = useState<GradcamResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [gradcamError, setGradcamError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [patientInfo, setPatientInfo] = useState<PatientInfo>({
    firstName: "",
    lastName: "",
    age: ""
  });

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

  const reportData = useMemo<ReportData | null>(() => {
    if (!prediction) {
      return null;
    }

    const generatedAt = formatReportDate(new Date());
    const patientLabel = getPdfSafeValue(getPatientLabel(patientInfo));
    const imageLabel = getPdfSafeValue(file?.name ?? "Uploaded image");
    const reportSummary = getDisplayedLabel(prediction.label);
    const alertLevel = getAlertLevel(probabilityPneumonia, prediction.is_inconclusive);
    const interpretation = getInterpretation(prediction, probabilityPneumonia);

    return {
      title: "Pneumonia AI Screening Report",
      generatedAt,
      patientLabel,
      imageLabel,
      summary: reportSummary,
      alertLevel,
      sections: [
        {
          title: "Patient information",
          rows: [
            { label: "First name", value: getPdfSafeValue(patientInfo.firstName) },
            { label: "Last name", value: getPdfSafeValue(patientInfo.lastName) },
            { label: "Age", value: getPdfSafeValue(patientInfo.age) }
          ]
        },
        {
          title: "Exam summary",
          rows: [
            { label: "Image file", value: imageLabel },
            { label: "Result", value: reportSummary },
            { label: "Pneumonia suspicion", value: formatPercent(probabilityPneumonia) },
            { label: "Confidence", value: formatPercent(modelConfidence) },
            { label: "Attention level", value: alertLevel }
          ]
        }
      ],
      interpretation,
      disclaimer: "Educational use only. This tool does not replace a radiologist or physician."
    };
  }, [file?.name, modelConfidence, patientInfo, prediction, probabilityPneumonia]);

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
    setHasAnalysisRun(false);

    if (!nextFile) {
      setFile(null);
      return;
    }

    if (nextFile.size === 0) {
      setFile(null);
      setError("This file is empty. Please choose a valid image.");
      return;
    }

    if (!isSupportedImageFile(nextFile)) {
      setFile(null);
      setError("Unsupported file type. Please upload a PNG, JPG, JPEG, or WEBP image.");
      return;
    }

    setFile(nextFile);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    selectFile(event.target.files?.[0] ?? null);
    event.target.value = "";
  }

  function handlePatientInfoChange(field: keyof PatientInfo, value: string) {
    const nextValue = field === "age" ? value.replace(/[^\d]/g, "").slice(0, 3) : value;

    setPatientInfo((current) => ({
      ...current,
      [field]: nextValue
    }));
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

    setHasAnalysisRun(true);
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
    if (!reportData) {
      return;
    }

    const blob = buildPdfBlob(reportData);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "pneumonia-screening-report.pdf";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  const analysisHeadline = prediction ? getDisplayedLabel(prediction.label) : "Your results";
  const analysisNarrative = prediction
    ? getInterpretation(prediction, probabilityPneumonia)
    : "Upload a chest X-ray and launch the model.";
  const shouldShowResultPanel = hasAnalysisRun;

  return (
    <main className="pageShell">
      <section className="heroSection">
        <div className="heroCopy">
          <h1>Pneumonia screening.</h1>
        </div>

        <aside className="heroPanel" aria-label="Session overview">
          <div className="heroStatusRow">
            <span className={`statusDot ${prediction ? "statusDotLive" : ""}`} aria-hidden="true" />
            <div className="heroStatusCopy">
              <p className="heroStatusLabel">{prediction ? "Analysis ready" : "Ready to analyze"}</p>
              <p className="heroPanelText">
                {prediction ? "Review the result below." : "Upload a chest X-ray to start."}
              </p>
            </div>
          </div>
        </aside>
      </section>

      <section className="workspaceSection" aria-label="Screening workspace">
        <form onSubmit={handleSubmit} className={`workspaceGrid ${shouldShowResultPanel ? "" : "workspaceGridSingle"}`}>
          <section className="uploadPanel">
            <div className="panelHeader">
              <div>
                <h2>Upload chest X-ray</h2>
              </div>
              <p className="panelMeta">PNG, JPG, JPEG</p>
            </div>

            <div className="patientGrid" aria-label="Patient information">
              <label className="field">
                <span>First name</span>
                <input
                  type="text"
                  value={patientInfo.firstName}
                  onChange={(event) => handlePatientInfoChange("firstName", event.target.value)}
                  placeholder="Enter first name"
                />
              </label>

              <label className="field">
                <span>Last name</span>
                <input
                  type="text"
                  value={patientInfo.lastName}
                  onChange={(event) => handlePatientInfoChange("lastName", event.target.value)}
                  placeholder="Enter last name"
                />
              </label>

              <label className="field fieldCompact">
                <span>Age</span>
                <input
                  type="text"
                  inputMode="numeric"
                  value={patientInfo.age}
                  onChange={(event) => handlePatientInfoChange("age", event.target.value)}
                  placeholder="Age"
                />
              </label>
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
                {isLoading ? (
                  <>
                    <span className="buttonSpinner" aria-hidden="true" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  "Run analysis"
                )}
              </button>
              {file ? <p className="actionHint">Image ready for screening.</p> : null}
            </div>
          </section>

          {shouldShowResultPanel ? (
            <aside className="resultPanel" aria-label="Prediction output">
              <div className="panelHeader panelHeaderCompact">
                <div>
                  <h2>{analysisHeadline}</h2>
                </div>
                {prediction ? (
                  <span
                    className={`resultTone ${prediction.label === "Normal" && !prediction.is_inconclusive ? "resultToneNormal" : "resultToneAlert"}`}
                  >
                    {getAlertLevel(probabilityPneumonia, prediction.is_inconclusive)}
                  </span>
                ) : null}
              </div>

              {prediction ? <p className="resultNarrative">{analysisNarrative}</p> : null}

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
                      <span>Class</span>
                      <strong>{prediction.label}</strong>
                    </div>
                    <div className="metricRow">
                      <span>Review</span>
                      <strong>{prediction.review_recommendation}</strong>
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

                  {isReportOpen && reportData ? (
                    <div className="reportPreview">
                      <div className="reportPreviewHeader">
                        <div>
                          <p className="reportPreviewKicker">Clinical report</p>
                          <h3>{reportData.title}</h3>
                          <p className="reportHeaderLead">{getPatientLabel(patientInfo)}</p>
                        </div>
                        <div className="reportPreviewMeta">
                          <span>{reportData.generatedAt}</span>
                          <span
                            className={`reportBadge ${prediction.label === "Normal" && !prediction.is_inconclusive ? "reportBadgeNormal" : "reportBadgeAlert"}`}
                          >
                            {reportData.alertLevel} attention
                          </span>
                        </div>
                      </div>

                      <div className="reportSummaryCard">
                        <p className="reportSummaryLabel">AI summary</p>
                        <div className="reportSummaryRow">
                          <strong>{reportData.summary}</strong>
                          <span>{formatPercent(probabilityPneumonia)} suspicion</span>
                        </div>
                        <p className="reportSummaryMeta">{file?.name ?? "Uploaded image"}</p>
                      </div>

                      <div className="reportGrid">
                        <div className="reportSection">
                          <p className="reportSectionTitle">Patient information</p>
                          <div className="reportRow">
                            <span>First name</span>
                            <strong>{getFilledValue(patientInfo.firstName)}</strong>
                          </div>
                          <div className="reportRow">
                            <span>Last name</span>
                            <strong>{getFilledValue(patientInfo.lastName)}</strong>
                          </div>
                          <div className="reportRow">
                            <span>Age</span>
                            <strong>{getFilledValue(patientInfo.age)}</strong>
                          </div>
                        </div>

                        <div className="reportSection">
                          <p className="reportSectionTitle">Exam summary</p>
                          <div className="reportRow reportRowStacked">
                            <span>Image</span>
                            <strong className="reportValueWrap">{file?.name ?? "Uploaded image"}</strong>
                          </div>
                          <div className="reportRow">
                            <span>Result</span>
                            <strong>{getDisplayedLabel(prediction.label)}</strong>
                          </div>
                          <div className="reportRow">
                            <span>Review</span>
                            <strong>{prediction.review_recommendation}</strong>
                          </div>
                          <div className="reportRow">
                            <span>Pneumonia suspicion</span>
                            <strong>{formatPercent(probabilityPneumonia)}</strong>
                          </div>
                          <div className="reportRow">
                            <span>Confidence</span>
                            <strong>{formatPercent(modelConfidence)}</strong>
                          </div>
                          <div className="reportRow">
                            <span>Alert level</span>
                            <strong>{getAlertLevel(probabilityPneumonia, prediction.is_inconclusive)}</strong>
                          </div>
                        </div>
                      </div>

                      <div className="reportSection">
                        <p className="reportSectionTitle">Interpretation</p>
                        <p className="reportNarrativeText">{getInterpretation(prediction, probabilityPneumonia)}</p>
                      </div>

                      <p className="reportFootnote">
                        Educational use only. This tool does not replace a radiologist or physician.
                      </p>
                    </div>
                  ) : null}
                </>
              ) : (
                <div className="emptyPanel">
                  {isLoading ? (
                    <div className="analysisLoader" aria-hidden="true">
                      <div className="loaderOrbit loaderOrbitOuter" />
                      <div className="loaderOrbit loaderOrbitInner" />
                      <div className="loaderCore">
                        <span className="loaderCoreDot" />
                      </div>
                    </div>
                  ) : (
                    <span className="emptyPulse" aria-hidden="true">
                      <span className="emptyPulseInner" />
                    </span>
                  )}
                  <p>{isLoading ? "Analyzing image..." : error ? "Analysis unavailable. Try again." : "Results will appear here."}</p>
                </div>
              )}
            </aside>
          ) : null}
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
                {gradcamError ? gradcamError : (
                  <div className="visualLoaderWrap">
                    <div className="analysisLoader" aria-hidden="true">
                      <div className="loaderOrbit loaderOrbitOuter" />
                      <div className="loaderOrbit loaderOrbitInner" />
                      <div className="loaderCore">
                        <span className="loaderCoreDot" />
                      </div>
                    </div>
                    <span>Generating visual explanation...</span>
                  </div>
                )}
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
