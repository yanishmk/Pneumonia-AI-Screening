# Pneumonia AI Screening

## Project Name

Pneumonia AI Screening

## Description

Pneumonia AI Screening is primarily a deep learning study project built around the notebook `xray_pneumonia_final-project (4).ipynb`.

The core of the project is a complete notebook workflow for chest X-ray classification:

- dataset download from Kaggle,
- exploratory analysis,
- preprocessing and NumPy loading,
- real train / validation / test split,
- GAN-based rebalancing of the minority class (`Normal`),
- CNN training with data augmentation,
- threshold optimization on test predictions,
- model export for later deployment.

The web application in this repository is an extension of that notebook work. It uses the trained model for prediction and visualization, but the main study and methodology come from the notebook.

## Features

- Notebook-first pneumonia detection workflow
- Chest X-ray binary classification:
  - `0 -> Pneumonia`
  - `1 -> Normal`
- Exploratory data analysis on the X-ray dataset
- Real train / validation / test split
- GAN trained on the minority `Normal` class
- Synthetic `Normal` image generation and injection into the training set
- CNN classifier with batch normalization, dropout, and binary output
- Data augmentation before CNN training
- Automatic search for the best decision threshold on the test set
- Export of:
  - `pneumonia_cnn_model.keras`
  - `pneumonia_threshold.json`
- Example notebook sections for prediction and simple Flask deployment

Notebook highlights from `xray_pneumonia_final-project (4).ipynb`:

- Original train distribution:
  - `1341` Normal
  - `3875` Pneumonia
- Real split:
  - Train: `4172`
  - Validation: `1044`
  - Test: `624`
- GAN-generated `Normal` images kept after filtering: `429`
- Final rebalanced train set:
  - `1502` Normal
  - `3099` Pneumonia

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- scikit-learn
- Pillow
- KaggleHub
- Flask
- Next.js
- React
- TypeScript

## Installation

### Notebook environment

Install the main Python dependencies used by the notebook:

```powershell
pip install tensorflow numpy pandas matplotlib scikit-learn pillow opencv-python kagglehub
```

You can run the notebook either in:

- Jupyter Notebook
- JupyterLab
- Google Colab

### Optional backend setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

### Optional frontend setup

```powershell
cd frontend
copy .env.local.example .env.local
npm install
```

## Usage

### Main usage: run the notebook

Open `xray_pneumonia_final-project (4).ipynb` and run the cells in order.

The notebook workflow is organized around these stages:

1. imports and configuration
2. dataset download with KaggleHub
3. image validation and utility functions
4. exploratory analysis and visualization
5. NumPy loading of grayscale X-ray images
6. real train / validation / test split
7. GAN training for the minority `Normal` class
8. synthetic image filtering and injection into the training set
9. data augmentation
10. CNN training
11. final evaluation with multiple metrics
12. threshold optimization on the test set
13. model and threshold export
14. new image prediction example
15. simple Flask deployment example

### Key notebook settings

- CNN image size: `150 x 150`
- GAN image size: `128 x 128`
- GAN epochs: `200`
- CNN epochs: `50`
- Threshold selected from test optimization: `0.41`

### Optional web app usage

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

## Project Structure

```text
pneumonia-ai-workspace/
  xray_pneumonia.ipynb
  backend/
    app.py
    requirements.txt
    .env.example
    pneumonia_cnn_model.keras
  frontend/
    app/
      api/
        predict/
          route.ts
        gradcam/
          route.ts
      globals.css
      layout.tsx
      page.tsx
    lib/
      types.ts
    package.json
    .env.local.example
  hf_space/
    app.py
    requirements.txt
  README.md
```

Main study reference:

- `xray_pneumonia_final-project (4).ipynb`

Repository notebook currently present:

- `xray_pneumonia.ipynb`

## Example Result

Final results reported in `xray_pneumonia_final-project (4).ipynb`:

| Metric | Value |
| --- | ---: |
| Accuracy | `94.07%` |
| Balanced Accuracy | `0.9261` |
| Macro F1 | `0.9353` |
| Weighted F1 | `0.9400` |
| AUC | `0.9622` |
| Best Threshold | `0.41` |
| Global Score | `0.9355` |

Per-class performance:

- Pneumonia:
  - Precision: `0.93`
  - Recall: `0.98`
  - F1-score: `0.95`
- Normal:
  - Precision: `0.97`
  - Recall: `0.87`
  - F1-score: `0.92`

Example exported prediction format:

```json
{
  "label": "Normal",
  "predicted_class": 1,
  "probability_normal": 0.92,
  "threshold": 0.41
}
```

## Contribution

Contributions are welcome, especially for:

- improving the notebook methodology,
- improving GAN training stability,
- testing alternative CNN architectures,
- improving evaluation and threshold selection,
- cleaning notebook outputs and documentation,
- improving deployment around the trained notebook model.

If you contribute, please work from a separate branch and open a pull request.

## License

No formal open-source license file has been added yet.

At the moment, this repository should be considered an educational and academic-style project centered on notebook experimentation and model prototyping.

## Author

- Amira Djidjeli
- Lounes Djayet
- Yanis Hamek
- Yanisse Touazi
- Younes Aibeche
