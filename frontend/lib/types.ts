export type PredictionResult = {
  label: "Normal" | "Pneumonia";
  predicted_class: 0 | 1;
  probability_normal: number;
  threshold: number;
};

export type GradcamResult = {
  gradcam_image_base64: string;
  gradcam_layer: string;
  label: "Normal" | "Pneumonia";
  predicted_class: 0 | 1;
  probability_normal: number;
  target_class: 0 | 1;
  threshold: number;
};

export type ApiError = {
  error: string;
};
