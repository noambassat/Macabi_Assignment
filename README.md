# Hypertension Risk Prediction - ML Pipeline

This repository contains a complete machine learning pipeline for predicting hypertensive disorders during pregnancy based on clinical, demographic, and textual features extracted from medical visit records.

---

## Project Structure


```
Home_Assignment/
                   
├── config.py                  # Centralized configuration paths and settings
├── data/                      # Input dataset(s)
├── main.py                    # Entry point for training, evaluation, and saving model
├── macabi_home_assignment.ipynb   # Original main notebook
├── macabi_home_assignment.html    # Exported HTML version of the notebook
├── models/                    # Trained model artifacts saved by joblib (may be empty initially)
├── requirements.txt           # Python dependencies for the environment
├── README.md                  # Project documentation (this file)
├── Clinical_Model_Summary.pdf # Formal PDF summary of methodology and results
└── src/
    ├── feature_engineering.py     # Data preparation, stratified splitting, and custom features
    ├── modeling.py                # Training & evaluation
    ├── pipeline.py                # sklearn-compatible modular pipeline
    ├── utils.py                   # Reusable utility functions
    ├── models.py                  # LightGBM model wrapper class
```

---

## Assignment main Notebook
https://github.com/noambassat/Macabi_Assignment/blob/main/macabi_home_assignment.ipynb

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/noambassat/Macabi_Assignment.git
cd Macabi_Assignment
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate       # On Linux/macOS
venv\Scripts\activate.bat      # On Windows
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

> **Note:** This project uses `sentence-transformers` for embedding clinical text. The default setup runs on CPU. If available, it is recommended to enable **GPU support** via CUDA-enabled PyTorch for significantly faster embedding inference.

---

## Model Overview

- Clinical features are selected using **ElasticNet** with scaling.
- Textual features from `clinical_sheet` are encoded via **TF-IDF + Mutual Information**.
- Paragraph embeddings are computed from the **last documented visit** using a pretrained **`multilingual-e5-base` SentenceTransformer**.
- All three sets are fused via **FeatureUnion** and passed to a **LightGBM classifier** with class imbalance handling.

All selection and transformation steps are fitted strictly on the **train set only**, to avoid data leakage.

---

## Running the Pipeline
After setup:
```bash
python main.py
```
This script:
1. Loads and splits the data.
2. Builds the unified feature pipeline.
3. Trains a LightGBM model.
4. Evaluates performance using classification metrics and a precision-recall curve.
5. Saves the trained pipeline to the `models/` directory.

---

##  Notes
- For best performance on large datasets with long text inputs, running on GPU is strongly recommended.
- Warnings are suppressed automatically for clean output.
- Confusion matrix and PR plots are shown interactively.

---
