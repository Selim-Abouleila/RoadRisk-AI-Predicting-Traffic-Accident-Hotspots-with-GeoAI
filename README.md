# �️ RoadRisk AI: Geospatial Accident Severity Pipeline

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Data Engineering](https://img.shields.io/badge/Focus-Data%20Engineering-orange?logo=apache-spark&logoColor=white)]()
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-red?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-green?logo=streamlit&logoColor=white)](https://streamlit.io)

> **An end-to-end Geospatial Machine Learning system designed to predict traffic accident severity in the Greater Paris area.**

---

## 📌 Project Overview

**RoadRisk AI** is a data engineering and machine learning project that processes large-scale accident data to provide actionable risk assessments. The system ingests raw government datasets, performs advanced geospatial feature engineering, and serves predictions via a low-latency interactive dashboard.

Built as part of an **M1 Data Engineering portfolio at EFREI Paris**, this project demonstrates the ability to architect robust data pipelines and deploy ML models into production-ready user interfaces.

### 🎯 Key Engineering Goals
*   **Pipeline Efficiency**: Optimized data handling using `Parquet` columnar storage for high-performance I/O.
*   **Geospatial Intelligence**: Integrated **Folium** and coordinate-based logic to map risk factors to specific geographic locations.
*   **Model Performance**: Fine-tuned **XGBoost Classifier** to handle imbalanced accident severity classes effectively.
*   **User Experience**: A responsive, professionally styled dashboard providing real-time inference.

---

## 🏗️ System Architecture

1.  **Data Ingestion**: Collection of historical traffic accident records from [data.gouv.fr](https://www.data.gouv.fr/).
2.  **Processing & Cleaning**:
    *   Normalization of categorical schemas (Vehicle types, Infrastructure).
    *   Imputation of missing demographic data.
    *   Standardization of geospatial coordinates (Lat/Lon).
3.  **Feature Engineering**:
    *   Spatial grouping and hotspot identification.
    *   Age and demographic scaling.
4.  **Model Training**:
    *   XGBoost implementation for high-speed gradient boosting.
    *   Serialized model artifacts for consistent inference.
5.  **Deployment**:
    *   Streamlit-based frontend for user interaction.
    *   Real-time inference engine (`run_xgboost_model.py`).

---

## 🚀 Key Features

*   **�️ Interactive Geo-Dashboard**: Clickable Folium map allowing users to pinpoint locations in Paris to simulate accident scenarios.
*   **🧠 Real-Time Inference**: Instant probability scoring (Minor vs. Serious/Fatal) based on location, time, and demographics.
*   **📊 Dynamic Visualization**: Custom CSS-styled metrics and risk indicators (Low, Elevated, Critical).
*   **🛠️ Optimized Stack**: Leverages `Pandas` and `NumPy` for vectorized operations and efficient memory usage.

---

## 🛠️ Technology Stack

| Domain | Technologies |
|--------|--------------|
| **Language** | Python 3.9+ |
| **Data Engineering** | Pandas, NumPy, Parquet |
| **Machine Learning** | XGBoost, Scikit-Learn (Pickle serialization) |
| **Geospatial** | Folium, Streamlit-Folium |
| **Visualization** | Streamlit, Matplotlib |
| **Environment** | Conda / Pip |

---

## 📂 Repository Structure

```bash
├── app.py                     # 📱 Production Interface (Streamlit)
├── run_xgboost_model.py       # ⚙️ Inference Engine Script
├── models/                    # 📦 Serialized Model Artifacts
│   └── xgboost_accident...pkl #    - Trained XGBoost Classifier
├── data/                      # 💾 Data Lake (Parquet/CSV)
├── notebooks/                 # � Experiments & Analysis
│   ├── 00_ingestion.ipynb     #    - Raw Data Loading
│   ├── 01_cleaning.ipynb      #    - Data Quality Checks
│   └── 02_training.ipynb      #    - Model Development
└── src/                       # 🧩 Utility Modules
```

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/RoadRisk-AI.git
cd RoadRisk-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```
*Access the dashboard at `http://localhost:8501`*

---

## �‍💻 Author

**Selim Abouleila**

