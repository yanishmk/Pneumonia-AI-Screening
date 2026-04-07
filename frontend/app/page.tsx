"use client";

import { ChangeEvent, DragEvent, FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import type { ApiError, GradcamResult, PredictionResult } from "@/lib/types";

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

export default function HomePage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isStudyInfoOpen, setIsStudyInfoOpen] = useState(false);
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

  async function postImage<T>(url: string, imageFile: File): Promise<T> {
    const formData = new FormData();
    formData.append("file", imageFile);

    const response = await fetch(url, {
      method: "POST",
      body: formData
    });

    const payload = (await response.json()) as T | ApiError;
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

  return (
    <main className="page-shell">
      <section className="hero">
        <div className="heroHeader">
          <div className="heroCopy">
            <p className="eyebrow">AI Radiology Workspace</p>
            <h1>Pneumonia AI Screening</h1>
            <p className="subtitle">
              Upload an image to assess pneumonia suspicion from a chest X-ray. Images that do not resemble a usable
              chest radiograph will be rejected with a clear message.
            </p>
          </div>

          <aside className="heroSummary" aria-label="Study summary">
            <button
              type="button"
              className="heroSummaryToggle"
              aria-expanded={isStudyInfoOpen}
              onClick={() => setIsStudyInfoOpen((current) => !current)}
            >
              <div className="heroSummaryHead">
                <div className="heroInfoIcon" aria-hidden="true">i</div>
                <div>
                  <p className="heroSummaryLabel">Learn more</p>
                  <h2 className="heroSummaryTitle">About this study</h2>
                </div>
              </div>
              <span className="heroSummaryChevron" aria-hidden="true">{isStudyInfoOpen ? "-" : "+"}</span>
            </button>

            {isStudyInfoOpen ? (
              <p className="heroSummaryText">
                Educational project focused on pneumonia detection from chest radiographs using a CNN model trained in
                the reference notebook.
              </p>
            ) : null}
          </aside>
        </div>
      </section>

      <section className="infoSection" aria-label="How it works">
        <div className="infoCard">
          <span className="infoStep">1</span>
          <h3>Upload</h3>
          <p>Add a chest X-ray image from your device.</p>
        </div>
        <div className="infoCard">
          <span className="infoStep">2</span>
          <h3>Analyze</h3>
          <p>The model evaluates the image and checks whether it is a compatible chest radiograph.</p>
        </div>
        <div className="infoCard">
          <span className="infoStep">3</span>
          <h3>Review</h3>
          <p>Read the prediction, confidence level, and Grad-CAM heatmap.</p>
        </div>
      </section>

      <section className="card" aria-label="Prediction form">
        <form onSubmit={handleSubmit} className="upload-grid">
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
            <strong>{file ? file.name : "Choose or drop a chest X-ray"}</strong>
            <small>Any image file is accepted. Non X-ray images will be rejected after analysis.</small>
            <button
              className="browseButton"
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                openFilePicker();
              }}
            >
              Browse Files
            </button>
          </div>

          <div className="previewPanel">
            {previewUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={previewUrl} alt="Uploaded chest X-ray preview" className="previewImage" />
            ) : (
              <div className="previewPlaceholder">Image preview will appear here</div>
            )}
          </div>

          <button className="submitButton" type="submit" disabled={!file || isLoading}>
            {isLoading ? "Analyzing..." : "Run Analysis"}
          </button>
        </form>

        {error ? (
          <div className="alert" role="alert">
            {error}
          </div>
        ) : null}

        {prediction ? (
          <section className="resultPanel" aria-label="Prediction result">
            <div>
              <p className="resultLabel">Prediction</p>
              <h2 className={prediction.label === "Normal" ? "normalText" : "pneumoniaText"}>
                {getDisplayedLabel(prediction.label)}
              </h2>
            </div>

            <div className="metrics">
              <div>
                <span>Pneumonia suspicion</span>
                <strong>{formatPercent(probabilityPneumonia)}</strong>
              </div>
              <div>
                <span>Model confidence</span>
                <strong>{formatPercent(modelConfidence)}</strong>
              </div>
              <div>
                <span>Alert level</span>
                <strong>{getAlertLevel(probabilityPneumonia)}</strong>
              </div>
            </div>

            <div className="confidenceBar" aria-label={`Pneumonia probability ${Math.round(probabilityPneumonia * 100)}%`}>
              <span style={{ width: `${Math.round(probabilityPneumonia * 100)}%` }} />
            </div>
            <p className="thresholdNote">Model decision threshold: {prediction.threshold}</p>
          </section>
        ) : null}

        {prediction ? (
          <section className="gradcamPanel" aria-label="Grad-CAM result">
            <div className="gradcamHeader">
              <div>
                <p className="resultLabel">Grad-CAM</p>
                <h3 className="gradcamTitle">Model attention map</h3>
              </div>
              <p className="gradcamDescription">
                Highlights the image regions that most influenced the model during prediction.
              </p>
            </div>

            {gradcam ? (
              <div className="gradcamContent">
                <div className="gradcamCompareGrid">
                  <div className="gradcamImageCard">
                    <div className="gradcamImageHeader">Original image</div>
                    {previewUrl ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={previewUrl} alt="Original uploaded chest X-ray" className="gradcamImage" />
                    ) : (
                      <div className="gradcamEmpty">Original image preview is unavailable.</div>
                    )}
                  </div>

                  <div className="gradcamImageCard">
                    <div className="gradcamImageHeader">Grad-CAM overlay</div>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`data:image/png;base64,${gradcam.gradcam_image_base64}`}
                      alt="Grad-CAM heatmap overlay"
                      className="gradcamImage"
                    />
                  </div>
                </div>

                <div className="gradcamNotes">
                  <div className="gradcamNoteCard">
                    <span>Visual explanation</span>
                    <strong>Highlighted regions influenced the model most.</strong>
                  </div>
                  <div className="gradcamNoteCard">
                    <span>Predicted label</span>
                    <strong>{getDisplayedLabel(prediction.label)}</strong>
                  </div>
                  <div className="gradcamNoteCard">
                    <span>Interpretation note</span>
                    <strong>Use as a visual aid, not as a diagnosis.</strong>
                  </div>
                </div>
              </div>
            ) : gradcamError ? (
              <div className="gradcamEmpty">{gradcamError}</div>
            ) : (
              <div className="gradcamEmpty">Grad-CAM will appear here after a successful analysis.</div>
            )}
          </section>
        ) : null}
      </section>

      <p className="disclaimer">
        Tool developed for educational purposes only, intended for visual exploration and triage support. This result
        is not a medical diagnosis, does not replace the opinion of a radiologist or physician, and must always be
        interpreted with the patient&apos;s clinical context.
      </p>
    </main>
  );
}
