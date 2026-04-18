"use client";

import { ChangeEvent, DragEvent, FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import type { ApiError, GradcamResult, PredictionResult } from "@/lib/types";

const PUBLIC_BACKEND_URL = process.env.NEXT_PUBLIC_FLASK_API_URL?.trim().replace(/\/$/, "");

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
  const [termsAccepted, setTermsAccepted] = useState(false);

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
    const alertLevel = getAlertLevel(probabilityPneumonia);
    const interpretation = getInterpretation(prediction.label, probabilityPneumonia);

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

    setFile(nextFile);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    selectFile(event.target.files?.[0] ?? null);
    event.target.value = "";
  }

  function handlePatientInfoChange(field: keyof PatientInfo, value: string) {
    setPatientInfo((current) => ({
      ...current,
      [field]: value
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

    if (!termsAccepted) {
      setError("Please accept the terms of use before running the analysis.");
      return;
    }

    if (!patientInfo.firstName.trim() || !patientInfo.lastName.trim() || !patientInfo.age.trim()) {
      setError("Please fill in all patient information before running the analysis.");
      return;
    }

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

  const analysisHeadline = prediction ? getDisplayedLabel(prediction.label) : "Awaiting analysis";
  const analysisNarrative = prediction
    ? getInterpretation(prediction.label, probabilityPneumonia)
    : "Upload a chest X-ray and run the model.";
  const shouldShowResultPanel = hasAnalysisRun;

  return (
    <>
      {/* ── NAVBAR ── */}
      <nav className="navbar">
        <div className="navInner">
          <div className="navBrand">
            <div className="brandIcon" aria-hidden="true">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="3" />
                <path d="M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12" />
              </svg>
            </div>
            <span className="brandName">PneumoAI</span>
            <span className="brandBeta">Beta</span>
          </div>

          <nav className="navLinks" aria-label="Page navigation">
            <a href="#how" className="navLink">How it works</a>
            <a href="#faq" className="navLink">FAQ</a>
            <a href="#about" className="navLink">About</a>
          </nav>

          <div className="navRight">
            <a
              href="https://www.google.com/maps/search/radiologist+near+me"
              target="_blank"
              rel="noopener noreferrer"
              className="navRadiologist"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
                <circle cx="12" cy="10" r="3" />
              </svg>
              Find a radiologist
            </a>
            <div className="navStatusPill">
              <span className={`navDot ${prediction ? "navDotLive" : ""}`} aria-hidden="true" />
              <span>{prediction ? "Analysis complete" : "Ready"}</span>
            </div>
          </div>
        </div>
      </nav>

      {/* ── MAIN ── */}
      <main className="pageShell">

        {/* ── HERO ── */}
        <section className="heroSection">
          <div className="heroCopy">
            <h1>Pneumonia AI Screening</h1>
            <p className="heroSub">Upload a chest X-ray and get an instant AI-assisted screening result.</p>
          </div>
        </section>

        {/* ── HOW IT WORKS ── */}
        <section className="howSection" id="how">
          <div className="howGrid">
            <div className="howCard">

              <div className="howIcon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <h3>Upload your X-ray</h3>
              <p>Drop or browse a chest X-ray image. PNG, JPG or JPEG.</p>
            </div>

            <div className="howCard">

              <div className="howIcon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="3" />
                  <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" />
                </svg>
              </div>
              <h3>AI analysis</h3>
              <p>The AI model analyzes the image and returns a pneumonia probability score.</p>
            </div>

            <div className="howCard">

              <div className="howIcon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="9" y1="13" x2="15" y2="13" />
                  <line x1="9" y1="17" x2="13" y2="17" />
                </svg>
              </div>
              <h3>Review &amp; export</h3>
              <p>View the result, explore the Grad-CAM heatmap and download a PDF report.</p>
            </div>
          </div>
        </section>

        {/* ── WORKSPACE ── */}
        <section className="workspaceSection" aria-label="Screening workspace">
          <form onSubmit={handleSubmit} className={`workspaceGrid ${shouldShowResultPanel ? "" : "workspaceGridSingle"}`}>

            {/* ── UPLOAD PANEL ── */}
            <section className="uploadPanel">
              <div className="panelHeader">
                <h2>Upload chest X-ray</h2>
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
                  <span className="dropzoneIcon" aria-hidden="true">+</span>
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
                    <div className="previewTopbarLeft">
                      <span
                        className={`previewStatusDot ${file ? "previewStatusDotLoaded" : ""}`}
                        aria-hidden="true"
                      />
                      <span>Preview</span>
                    </div>
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

              <label className="termsRow">
                <input
                  type="checkbox"
                  checked={termsAccepted}
                  onChange={(e) => setTermsAccepted(e.target.checked)}
                  className="termsCheckbox"
                />
                <span>I understand this tool is for educational use only and does not replace a medical professional.</span>
              </label>

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
                {file ? (
                  <p className="actionHint">
                    <span className="actionHintDot" aria-hidden="true" />
                    Image ready
                  </p>
                ) : null}
              </div>
            </section>

            {/* ── RESULT PANEL ── */}
            {shouldShowResultPanel ? (
              <aside className="resultPanel" aria-label="Prediction output">
                <div className="resultPanelHeader">
                  <div>
                    <h2>{analysisHeadline}</h2>
                  </div>
                  {prediction ? (
                    <span
                      className={`resultTone ${prediction.label === "Normal" ? "resultToneNormal" : "resultToneAlert"}`}
                    >
                      {getAlertLevel(probabilityPneumonia)}
                    </span>
                  ) : null}
                </div>

                {prediction ? <p className="resultNarrative">{analysisNarrative}</p> : null}

                {prediction ? (
                  <>
                    <div className="scoreHero">
                      <span className="scoreLabel">Pneumonia suspicion</span>
                      <strong className={prediction.label === "Normal" ? "scoreNormal" : "scoreAlert"}>
                        {formatPercent(probabilityPneumonia)}
                      </strong>
                    </div>

                    <div className="metricGrid">
                      <div className="metricCard">
                        <span className="metricLabel">Confidence</span>
                        <strong>{formatPercent(modelConfidence)}</strong>
                      </div>
                      <div className="metricCard">
                        <span className="metricLabel">Classification</span>
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
                      <button
                        type="button"
                        className="secondaryButton secondaryButtonStrong"
                        onClick={handleDownloadPdf}
                      >
                        Download PDF
                      </button>
                    </div>

                    {isReportOpen && reportData ? (
                      <div className="reportPreview">
                        <div className="reportPreviewInner">
                          <div className="reportPreviewHeader">
                            <div>
                              <p className="reportPreviewKicker">Clinical report</p>
                              <h3>{reportData.title}</h3>
                              <p className="reportHeaderLead">{getPatientLabel(patientInfo)}</p>
                            </div>
                            <div className="reportPreviewMeta">
                              <span>{reportData.generatedAt}</span>
                              <span
                                className={`reportBadge ${prediction.label === "Normal" ? "reportBadgeNormal" : "reportBadgeAlert"}`}
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
                                <span>Suspicion</span>
                                <strong>{formatPercent(probabilityPneumonia)}</strong>
                              </div>
                              <div className="reportRow">
                                <span>Confidence</span>
                                <strong>{formatPercent(modelConfidence)}</strong>
                              </div>
                              <div className="reportRow">
                                <span>Alert level</span>
                                <strong>{getAlertLevel(probabilityPneumonia)}</strong>
                              </div>
                            </div>
                          </div>

                          <div className="reportSection">
                            <p className="reportSectionTitle">Interpretation</p>
                            <p className="reportNarrativeText">
                              {getInterpretation(prediction.label, probabilityPneumonia)}
                            </p>
                          </div>

                          <p className="reportFootnote">
                            Educational use only. This tool does not replace a radiologist or physician.
                          </p>
                        </div>
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
                    <p>
                      {isLoading
                        ? "Analyzing image..."
                        : error
                          ? "Analysis unavailable. Try again."
                          : "Results will appear here."}
                    </p>
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

          {/* ── VISUAL SECTION ── */}
          <section className="visualSection" aria-label="Visual explanation">
            <div className="visualSectionHeader">
              <h2>Model focus map</h2>
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
                    <p>Warmer regions indicate stronger model attention influencing the prediction.</p>
                  </div>
                </>
              ) : (
                <div className="visualEmpty">
                  {gradcamError ? (
                    gradcamError
                  ) : (
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

        {/* ── FAQ ── */}
        <section className="faqSection" id="faq">
          <h2 className="faqTitle">Frequently asked questions</h2>
          <div className="faqList">
            <details className="faqItem">
              <summary className="faqQuestion">
                What image formats are supported?
                <span className="faqChevron" aria-hidden="true" />
              </summary>
              <p className="faqAnswer">PNG, JPG and JPEG. Use a standard frontal chest X-ray for best results.</p>
            </details>

            <details className="faqItem">
              <summary className="faqQuestion">
                How accurate is the AI model?
                <span className="faqChevron" aria-hidden="true" />
              </summary>
              <p className="faqAnswer">A custom CNN trained on chest X-ray data. Always confirm results with a qualified radiologist.</p>
            </details>

            <details className="faqItem">
              <summary className="faqQuestion">
                What is Grad-CAM?
                <span className="faqChevron" aria-hidden="true" />
              </summary>
              <p className="faqAnswer">Grad-CAM highlights the regions the model focused on. Warmer colors indicate higher attention.</p>
            </details>

            <details className="faqItem">
              <summary className="faqQuestion">
                What do the alert levels mean?
                <span className="faqChevron" aria-hidden="true" />
              </summary>
              <p className="faqAnswer"><strong>High</strong> ≥ 80% · <strong>Moderate</strong> 55–80% · <strong>Low</strong> below 55%.</p>
            </details>

            <details className="faqItem">
              <summary className="faqQuestion">
                Is my data stored or shared?
                <span className="faqChevron" aria-hidden="true" />
              </summary>
              <p className="faqAnswer">Images are processed in real time and not stored. No data is retained after the session.</p>
            </details>

            <details className="faqItem">
              <summary className="faqQuestion">
                Can this replace a radiologist?
                <span className="faqChevron" aria-hidden="true" />
              </summary>
              <p className="faqAnswer">No. This tool is for educational use only and does not replace a radiologist or physician.</p>
            </details>
          </div>
        </section>

        {/* ── ABOUT ── */}
        <section className="aboutSection" id="about">
          <div className="aboutGrid">
            <div>
              <h2 className="aboutTitle">About PneumoAI</h2>
              <p className="aboutText">PneumoAI is an educational AI screening tool that analyzes chest X-ray images for signs of pneumonia. It is designed for research and learning purposes only.</p>
            </div>
            <div>
              <h3 className="aboutSubtitle">Disclaimer</h3>
              <p className="aboutText">This tool is not a certified medical device. Results must always be confirmed by a qualified radiologist or physician. Do not use for clinical diagnosis.</p>
            </div>
            <div>
              <h3 className="aboutSubtitle">Contact</h3>
              <p className="aboutText">For questions or feedback, contact us at <a href="mailto:yanishamek14@gmail.com" className="aboutLink">yanishamek14@gmail.com</a></p>
            </div>
          </div>
        </section>

        <p className="footerNote">© {new Date().getFullYear()} PneumoAI · Educational use only · Not a medical device</p>
      </main>
    </>
  );
}
