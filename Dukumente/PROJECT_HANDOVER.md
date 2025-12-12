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

# Technical Handover & Executive Summary

**To:** Head of AI Department
**From:** Antigravity Engineering Team
**Date:** 2025-12-11
**Subject:** Implementation of Acoustic Anomaly Detection System (Pilot Ready)

## 1. Executive Summary
We have successfully implemented the "Project Antigravity" end-to-end AI system for analyzing acoustic sensor data from dishwasher production lines. The system is designed to detect defects with a target False Negative Rate (FNR) of < 0.1%, while fully complying with the EU AI Act (High-Risk Classification).

## 2. System Architecture
The solution follows a domain-driven monorepo structure:
*   **01 Data Engineering**: Robust ETL pipeline using FFT for feature extraction (2048-sample windows) and Great Expectations for data quality.
*   **03 ML Engineering**: Random Forest & Gradient Boosting models with MLflow tracking, automatic Threshold Optimization, and SHAP explainability.
*   **04 Governance**: Fairlearn integration for bias auditing and EU AI Act risk documentation (Annex III).
*   **05 Backend System**: Production-ready FastAPI service (containerized) with drift detection hooks.
*   **06 Frontend Interface**: Flutter-based Real-Time ROI Dashboard.

### 2.1 Data Collection Strategy
*   **Sensors**: Industrial Accelerometers (Vibration) + High-Fidelity Microphones (Audio Spectrum).
*   **Sampling**: 44.1kHz sample rate, aggregated into 2048-point spectral windows.
*   **Current Status**: For this Pilot/PoC, we are using **Synthetic Data** (`generate_synthetic_data.py`) modeled after physical baseline measurements from Line A to ensure privacy and safety during development.

### 2.2 Non-Technical Analogy (The "Doctor Check")
*   **Accelerometer (The Hand)**: Imagine placing your hand on the dishwasher door while it runs. If it vibrates violently, you know a screw is loose. The accelerometer "feels" this shaking 44,000 times a second.
*   **Microphone (The Ear)**: Imagine listening to the machine. A healthy machine goes *"whoosh"*. A broken pump goes *"screech"* or *"clunk"*. The microphone "listens" for these bad noises that humans might miss.

### 2.3 Real-World Integration (The "Warranty Stamp" Process)
To get the final "Warranty Stamp" (Quality Pass), a BSH dishwasher must pass this sequence of End-of-Line (EOL) tests:

1.  **Visual Check (Cameras/People)**: Is the door scratched? Is the logo straight?
2.  **Safety Check (Electric)**: Will the user get shocked? (High Voltage Test).
3.  **Function Test (Water - 30 seconds)**:
    *   **Connect**: Automaton plugs in water/power.
    *   **Fill (5s)**: Water valve opens. Flow meter checks volume.
    *   **Circulate (15s)**: **Main Motor turns ON**. Water sprays. (This is where we listen!).
    *   **Drain (5s)**: Drain pump turns ON.
4.  **Acoustic AI Check (Project Antigravity)**: **[THIS IS US]**
    *   *While* the water test (#3) is running, our sensors listen to the pump and motor.
    *   We catch invisible defects like a slightly bent motor shaft or a loose bearing that passes the water test but will break in 6 months.

**Only if ALL 4 pass, the machine gets the stamp.**

> **What if the Sensor breaks?** (The "Heartbeat" Check)
> *   **Scenario A (Dead Motor)**: The machine is silent, but the sensor is working. The AI detects "Abnormal Silence" during the cycle --> **DEFECT** (Correct).
> *   **Scenario B (Dead Sensor)**: If the microphone is unplugged/broken, the system detects "Zero Signal" *before* the test starts (Heartbeat). --> **SYSTEM ERROR** (The conveyor stops; it does NOT blame the dishwasher).

> **Note on "Dirty Plates"**:
> You cannot wash dirty plates on the assembly line (it takes 2 hours!).
> *   **The 100% Test (Where WE are)** get the mechanical check above (18 seconds).
> *   **The 1% "Audit"** are pulled aside for a "Laboratory Audit" where they wash real dirty dishes to prove the cleaning design works.
> *   **Project Antigravity helps the 100% line.**

## Appendix C: Daily Operational Guide (Start Here)

Use this guide to restart the system from scratch (e.g., after rebooting).

### Terminal 1: Backend Service
```bash
# 1. Navigate to Project
cd ~/Desktop/bsh-verification-antigravity

# 2. Activate Python Environment
source venv/bin/activate

# 3. Start API
cd 05_backend_system/api_service
uvicorn app.main:app --reload
```

### Terminal 2: Frontend Dashboard
```bash
# 1. Navigate to Project
cd ~/Desktop/bsh-verification-antigravity

# 2. Setup Flutter (Run once per terminal session)
export PATH="$PWD/tools/flutter/bin:$PATH"

# 3. Launch App
cd 06_frontend_interactive
flutter run -d chrome
```

---

*Document approved for internal distribution. Contact MLOps team for questions.*

## 3. Operational Commands (Runbook)

### Prerequisite
Ensure Python 3.11+ and Virtual Environment are active.

### Glossary
| Term | Definition |
|---|---|
| **FFT** | Fast Fourier Transform - algorithm for frequency analysis |
| **SHAP** | SHapley Additive exPlanations - model interpretability method |
| **MES** | Manufacturing Execution System |
| **Synthetic Data** | Artificial data generated by code (`generate_synthetic_data.py`) to simulate real-world physics, used for testing without privacy risks. |

### 4.1 Automated System Verification (Recommended)
To verify the entire system integrity (Data Gen + ETL + Model + API), execute the system test runner. This is the primary command for proving system readiness.

```bash
venv/bin/python tests/run_e2e_demo.py
```

**Expected Output:**
*   Generation of 1000 synthetic acoustic samples.
*   ETL extraction of spectral features (FFT).
*   Model training (RF) logged to MLflow.
*   API endpoint `/verify` tested successfully (Http 200).
*   **Logs**: Audit trail available at `tests/system_test.log`.

### 4.2 Interactive Demo (Frontend Dashboard)
Showcase the **Real-Time ROI** capabilities to stakeholders using the Flutter Dashboard.

**Run via Web Browser (Zero Install):**
```bash
cd 06_frontend_interactive
flutter run -d chrome
```

**Key Talking Points for Demo:**
1.  **Live Connection**: "The 'API Connected' badge confirms we are live-streaming data to our specialized backend."
2.  **Financial Impact**: "The ROI counter demonstrates immediate warranty savings per caught defect."
3.  **Explainability**: "Each decision comes with a 'Why?' (XAI), critical for trust on the factory floor."

### 4.3 Manual Execution Steps
For detailed step-by-step debugging or demo purposes:

**1. Data Generation**
```bash
venv/bin/python 01_data_engineering/src/generate_synthetic_data.py
```

**2. ETL Processing**
```bash
venv/bin/python 01_data_engineering/src/etl_pipeline.py --source data/raw/sensor_data.csv --output data/processed --format csv
```

**3. Model Training**
```bash
# Finds the latest processed file automatically
LATEST_FILE=$(ls -t data/processed/*.csv | head -n1)
venv/bin/python 03_ml_engineering/ml_pipeline/training.py --data-path $LATEST_FILE
```

**4. Start Inference API**
```bash
venv/bin/uvicorn 05_backend_system.api_service.app.main:app --reload
```

## 5. Compliance & Next Steps
*   **Risk Assessment**: See `04_governance_compliance/eu_ai_act/risk_assessment.md`.
*   **Feature Importance**: JSON artifacts generated during training for transparency.
*   **Recommendation**: Proceed to Pilot deployment on Production Line A.
