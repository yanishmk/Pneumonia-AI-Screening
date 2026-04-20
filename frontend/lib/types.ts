export type PredictionResult = {
  label: "Normal" | "Pneumonia" | "Inconclusive";
  predicted_class: -1 | 0 | 1;
  probability_normal: number;
  threshold: number;
  inconclusive_margin: number;
  is_inconclusive: boolean;
  review_recommendation: string;
};

export type GradcamResult = {
  gradcam_image_base64: string;
  gradcam_layer: string;
  label: "Normal" | "Pneumonia" | "Inconclusive";
  predicted_class: -1 | 0 | 1;
  probability_normal: number;
  target_class: 0 | 1;
  threshold: number;
  inconclusive_margin: number;
  is_inconclusive: boolean;
  review_recommendation: string;
};

export type ApiError = {
  error: string;
};
