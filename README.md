# Early Prediction of Clinical Sepsis using XGBoost
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🏥 Project Overview
This repository contains a machine learning pipeline designed to predict the onset of sepsis in ICU patients, based on the **PhysioNet/Computing in Cardiology Challenge 2019**. The model utilizes high-frequency physiological vitals to identify risk patterns before clinical deterioration occurs.

## 🚀 Key Features
* **Temporal Feature Engineering:** Implements a 6-hour sliding window to capture trends in Heart Rate (HR) and Temperature.
* **Handling Class Imbalance:** Utilizes XGBoost's `scale_pos_weight` to manage the high ratio of non-septic to septic records (approx. 90:10).
* **Clinical Data Simulation:** Includes a robust synthetic data generator (`generate_data.py`) to ensure pipeline reproducibility and unit testing.
* **Performance Metric:** Optimized for **AUPRC** (Area Under Precision-Recall Curve), the clinical standard for rare event detection.

## 🛠️ Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/punarnavajha/sepsis-prediction.git
   cd sepsis-prediction
   ```

2. **Generate Synthetic Data:**
   ```bash
   python3 generate_data.py
   ```

3. **Train and Evaluate the Model:**
   ```bash
   python3 sepsis_prediction.py
   ```

## 📊 Methodology
The pipeline follows a standard Clinical Data Science workflow:
1.  **Data Ingestion:** Loading .psv (pipe-separated) clinical records.
2.  **Preprocessing:** Forward-filling (FFill) missing vitals to simulate real-time EHR bedside monitoring.
3.  **Modeling:** Training a Gradient Boosted Decision Tree (XGBoost) to classify risk.

## 🎓 UCL Application Context
This project demonstrates core competencies required for the **MSc Health Data Science** at UCL, specifically:
* Programming for Health Data (Python)
* Statistics for Health Data (Class imbalance & AUPRC)
* Data Engineering (EHR data structures)

---
*Developed by Punarnava Jha*
