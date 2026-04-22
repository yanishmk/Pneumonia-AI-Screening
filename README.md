# Pneumonia AI Screening

Pneumonia AI Screening is a study and prototyping repository centered on chest X-ray pneumonia classification.

The main artifact in this repository is the notebook [xray_pneumonia.ipynb](./xray_pneumonia.ipynb), which contains the end-to-end experimentation workflow used to:

- download the dataset with KaggleHub,
- audit the image files and remove corrupted samples,
- perform exploratory analysis and visualization,
- split the data into train / validation / test,
- rebalance the training set with a GAN trained on the minority `Normal` class,
- train a CNN classifier with data augmentation,
- evaluate the model with multiple metrics,
- visualize model behavior with ROC, precision-recall, confusion matrix, and Grad-CAM,
- export the trained model and result bundle for later deployment.

## Notebook Workflow

The notebook is organized into the following stages:

1. imports and configuration
2. dataset download
3. utility functions and corrupted-image audit
4. exploratory analysis and class visualization
5. NumPy loading of grayscale chest X-ray images
6. train / validation / test split
7. GAN training on the minority class
8. synthetic image generation and train-set rebalancing
9. CNN training with augmentation
10. final evaluation and threshold analysis
11. ROC / PR curves, confusion matrix, and Grad-CAM
12. model export and ZIP bundle generation

## Current Notebook Features

- binary classification:
  - `0 -> Pneumonia`
  - `1 -> Normal`
- GAN-based minority-class rebalancing
- CNN training with batch normalization and dropout
- data augmentation before final training
- threshold sweep on test predictions with a weighted selection score:
  - `0.60 * accuracy + 0.40 * macro_f1`
- Grad-CAM visualization on test samples
- export of:
  - `pneumonia_cnn_model.keras`
  - `metrics.json`
  - `classification_report.txt`
  - `classification_report.json`
  - `test_predictions.csv`
  - `confusion_matrix.csv`
  - `confusion_matrix.png`
  - `threshold_search.csv`
  - ZIP archive with the full run bundle

## Example Result From The Notebook

The notebook currently reports the following saved test metrics in its output cells:

| Metric | Value |
| --- | ---: |
| Accuracy | `92.79%` |
| Balanced Accuracy | `0.9184` |
| Macro F1 | `0.9223` |
| Weighted F1 | `0.9275` |
| Global Score | `0.9240` |
| AUC | `0.9633` |
| Selected Threshold | `0.31` |
| Threshold Strategy | `0.60 * accuracy + 0.40 * macro_f1` |

Per-class performance shown in the notebook:

- Pneumonia:
  - Precision: `0.93`
  - Recall: `0.96`
  - F1-score: `0.94`
- Normal:
  - Precision: `0.92`
  - Recall: `0.88`
  - F1-score: `0.90`

These numbers come from the notebook outputs currently stored in the repository and may vary slightly if the notebook is re-run.

## Repository Structure

```text
pneumonia-ai-workspace/
  xray_pneumonia.ipynb
  README.md
  backend/
    app.py
    requirements.txt
    README.md
  frontend/
    ...
  hf_space/
    app.py
    requirements.txt
```

## Installation

### Notebook Environment

Install the main Python dependencies used by the notebook:

```powershell
pip install tensorflow numpy pandas matplotlib scikit-learn pillow opencv-python kagglehub
```

You can run the notebook in:

- Jupyter Notebook
- JupyterLab
- Google Colab

### Optional Backend Setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

### Optional Frontend Setup

```powershell
cd frontend
copy .env.local.example .env.local
npm install
```

## Usage

### Main Usage

Open `xray_pneumonia.ipynb` and run the cells in order.

### Optional Web App Usage

Run the backend:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python app.py
```

Run the frontend:

```powershell
cd frontend
npm run dev
```

## Notes

- This repository is an educational deep-learning project, not a clinical diagnostic device.
- The notebook remains the primary source of truth for the model-building methodology.
- The web application extends the notebook work for deployment and interaction.

## Author

- Amira Djidjeli
