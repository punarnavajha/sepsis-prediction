# Clinical Sepsis Prediction Pipeline
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ML Framework: XGBoost](https://img.shields.io/badge/ML-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Executive Summary
This project implements an end-to-end machine learning pipeline for the early detection of sepsis, leveraging high-frequency physiological vitals. By identifying subtle patterns in patient deterioration up to 6 hours in advance, this system provides a foundation for real-time clinical decision support (CDS).

## 🛠️ Technical Architecture
* **Data Simulation Engine:** Includes `generate_data.py` to create synthetic Patient-Stay-Vectors (PSV) that mimic the PhysioNet 2019 Challenge schema for unit testing and CI/CD pipelines.
* **Feature Engineering:** Implements temporal rolling averages (6-hour windows) to capture physiological trends rather than static data points.
* **Imbalance Management:** Utilizes cost-sensitive learning via XGBoost's `scale_pos_weight` to handle the high-sparsity nature of sepsis labels in ICU settings.
* **Evaluation Metric:** Prioritizes **AUPRC** (Area Under Precision-Recall Curve) to ensure clinical utility by balancing sensitivity with alarm fatigue prevention.



## 🚀 Getting Started
1. **Clone & Setup:**
   ```bash
   git clone https://github.com/punarnavajha/sepsis-prediction.git
   cd sepsis-prediction
   ```

2. **Environment Simulation:**
   Run the generator to create the local clinical data environment:
   ```bash
   python3 generate_data.py
   ```

3. **Model Execution:**
   Execute the training and evaluation script:
   ```bash
   python3 sepsis_prediction.py
   ```

## 📊 Methodology
The pipeline is designed for modularity and deployment:
1.  **Ingestion:** Parses pipe-separated clinical logs.
2.  **Imputation:** Employs Forward-Fill (FFill) to simulate the reality of bedside monitoring where the last recorded value is assumed valid until a new measurement is taken.
3.  **Classification:** Gradient Boosted Decision Trees (GBDT) optimized for non-linear physiological relationships.



---
*Maintained by Punarnava Jha*
