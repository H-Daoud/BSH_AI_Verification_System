🏭 **BSH_AI_Verification_System (Prototyp)**
⚡ **Enterprise AI Test_Verification_System for Dishwasher Production_Implemented_with_Hybrid_MachineLearnignModel+ Dockerized_in(Azure)**
**Pipeline:** `data_engineering` ➔ `data_science_research` ➔ `ml_engineering`➔ `AI_Test_Verification_System`
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Azure](https://img.shields.io/badge/Cloud-Azure-0078D4?logo=microsoftazure)
![Model](https://img.shields.io/badge/Router-DistilBERT-yellow)
![GenAI](https://img.shields.io/badge/Reasoning-OpenAI-green?logo=openai)
![Status](https://img.shields.io/badge/Status-Prototype-orange)
![DevOps](https://img.shields.io/badge/MLOps-red)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![Flutter](https://img.shields.io/badge/Flutter-3.0+-blue.svg)](https://flutter.dev)
[![MLflow](https://img.shields.io/badge/MLflow-2.10+-orange.svg)](https://mlflow.org)

<p align="center">
  <img src=" " width="800">
</p>


## 🎯 Project Overview
An end-to-end enterprise ML system for detecting acoustic anomalies in dishwasher production lines at BSH Hausgeräte Group.
<p align="center">
  <img src="https://github.com/H-Daoud/BSH_AI_Verification_System/blob/main/aws_terraform_infrastructure/AWS_Al%20Verification%20System_Project_Architecture.png" width="400">
</p>
## 📁 Repository Structure

```
bsh-verification-antigravity/
├── .azure-pipelines/          # CI/CD Pipeline Definitions
├── 00_business_context/       # Business Objectives & ROI
├── 01_data_engineering/       # ETL & Data Quality (Great Expectations)
├── 02_data_science_research/  # EDA & Experimentation
├── 03_ml_engineering/         # MLflow Training Pipeline
├── 04_governance_compliance/  # EU AI Act & Ethics
├── 05_backend_system/         # FastAPI Service & Docker
├── 06_frontend_interactive/   # Flutter Dashboard
└── data/                      # DVC-tracked Data
```

## 🚀 Quick Start

```bash
# Clone repository
git clone <repository-url>
cd bsh-verification-antigravity

# Setup Python environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Initialize DVC
dvc init
dvc pull

# Run API locally
cd 05_backend_system/api_service
uvicorn app.main:app --reload
```

## 🛠️ Tech Stack

| Domain | Tools |
|--------|-------|
| Data Engineering | DVC, Great Expectations |
| ML Pipeline | MLflow, SHAP |
| Backend | FastAPI, Docker |
| Frontend | Flutter |
| Compliance | Trivy, EU AI Act Framework |
| CI/CD | Azure Pipelines |

## 📊 Key Features

- **Acoustic Anomaly Detection**: ML-based quality verification
- **Real-time Inference**: <100ms latency prediction API
- **Explainability**: SHAP-based model interpretation
- **EU AI Act Compliance**: Full documentation and audit trails
- **ROI Dashboard**: Flutter-based visualization

## 📝 License

Proprietary - BSH
Author: HasSan Daoud

---
*Maintained for BSH MLOps Architecture Team*
