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
# *Dev. guide
# Internal Developer Guide: Project Antigravity
**Confidential: For Engineering Team Only**

This document provides a line-by-line breakdown of the codebase to help you answer technical questions during the handover.

---

## 📂 00_business_context
*Why we started.*

### `problem_definition.md`
*   **Purpose**: The "North Star" document. Defines *why* we are building this (Money + Quality).
*   **Key Content**: 
    *   Current costs (€4.2M/year warranty claims).
    *   Target KPI (Reduce defects from 2.5% to 0.1%).
    *   Start here if someone asks "Is this project worth the money?"

---

## 📂 01_data_engineering
*Where the data comes from.*

### `src/generate_synthetic_data.py`
*   **Purpose**: Generates fake but realistic data because we couldn't access the real factory line yet.
*   **Key Logic**: 
    *   Simulates "Normal" machines (95%) and "Defective" ones (5%).
    *   Adds noise to simulate factory environment.
*   **Connection**: Output goes to `data/raw/sensor_data.csv`.

---

## 📂 03_ml_engineering
*The "Brain" of the system.*

### `ml_pipeline/training.py` (Concept)
*   **Purpose**: Trains the AI model.
*   **Key Logic**: 
    *   Reads data.
    *   Trains a Random Forest Classifier.
    *   Saves the trained model to `models/production_model.pkl`.
*   **Connection**: The Backend (`05`) loads the file created here.

---

## 📂 04_governance_compliance
*Keeping it legal.*

### `eu_ai_act/risk_assessment.md`
*   **Purpose**: Legal safety. BSH is a European company, so we MUST comply with the EU AI Act.
*   **Key Content**: 
    *   Classifies the system as "High Risk" (Annex III) because it's a safety component.
    *   Lists the "Human Oversight" measures (the Dashboard).

---

## 📂 05_backend_system
*The "Engine" that runs the AI.*

### `api_service/app/main.py`
*   **Purpose**: The actual server that processes requests.
*   **Key Logic**:
    *   `verify_device()`: The main function. Takes sensor data -> Returns "Defect" or "Safe".
    *   `generate_explanation()`: The rule-based logic we added to say "Check Pump" vs "Check Bearings".
    *   `log_defect()`: The auto-logger that writes to the CSV.
*   **Connection**: Receives data from Frontend (`06`), sends answers back.

### `api_service/Dockerfile`
*   **Purpose**: Instructions to package this Python app into a "Container" so it runs on any cloud (AWS/Azure).
*   **Key Logic**: Installs Python, copies code, creates `reports` folder permissions.

### `api_service/reports/defect_log.csv`
*   **Purpose**: The "Black Box" recorder.
*   **Content**: A permanent list of every defect found (auto) or reported (manual).

---

## 📂 06_frontend_interactive
*The "Face" of the system.*

### `lib/screens/dashboard_roi.dart`
*   **Purpose**: The screen you see in the browser.
*   **Key Logic**:
    *   `_simulateRealTimeData()`: Fake loop to generate a new "device" every 2 seconds.
    *   `_verifyNewUnit()`: Sends that fake device data to the Backend API (`05`).
    *   `build()`: Draws the UI (ROI cards, "Report" button, List of machines).
*   **Connection**: Calls `http://localhost:8000/verify`.

### `lib/main.dart`
*   **Purpose**: The entry point of the Flutter app.
*   **Key Logic**: Just sets up the Theme (colors, fonts) and loads the Dashboard screen.

---

## 📂 Root Directory
*Top-level documentation.*

### `PROJECT_HANDOVER.md`
*   **Purpose**: The presentation document acting as your slide deck.
*   **Content**: Executive Summary, ROI math, "Doctor Check" analogy, Glossary.

### `CLOUD_DEPLOYMENT.md`
*   **Purpose**: Instructions for the DevOps team.
*   **Content**: "How to put this on AWS/Azure".

### `tests/run_e2e_demo.py`
*   **Purpose**: The "Sanity Check".
*   **Key Logic**: Runs the whole chain (Generate Data -> Train Model -> Start API -> Test API) in one go to prove it works.

---

## 🔗 How it all connects (The Data Flow)

1.  **Dashboard (`06`)** creates a fake sensor reading.
2.  **Dashboard** sends it to **Backend (`05/main.py`)**.
3.  **Backend** asks the **Model** (from `03`): "Is this bad?"
4.  **Backend** checks **Rules**: "Is it the Pump or Motor?"
5.  **Backend** writes to **CSV Log** (`defect_log.csv`) if bad.
6.  **Backend** replies to **Dashboard**: "It's Defective!"
7.  **Dashboard** turns the card **RED** and updates the **ROI Counter**.

---

