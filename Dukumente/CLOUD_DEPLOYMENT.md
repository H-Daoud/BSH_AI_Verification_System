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

# Cloud Deployment Guide (Docker)

This guide explains how to package the BSH Antigravity Verification API and deploy it to major cloud providers.

## 1. Build Docker Image

First, ensure you are in the project root:
```bash
cd ~/Desktop/bsh-verification-antigravity
```

Build the image locally:
```bash
docker build -t bsh-verification-api -f 05_backend_system/api_service/Dockerfile 05_backend_system/api_service
```

Test valid build:
```bash
docker run -p 8000:8000 bsh-verification-api
# Access http://localhost:8000/docs
```

---

## 2. AWS Deployment (AWS App Runner)
*Best for: Simple, fully managed container deployment.*

1.  **Push to ECR:**
    ```bash
    aws ecr create-repository --repository-name bsh-api
    aws ecr get-login-password | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<region>.amazonaws.com
    docker tag bsh-verification-api:latest <aws_account_id>.dkr.ecr.<region>.amazonaws.com/bsh-api:latest
    docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/bsh-api:latest
    ```
2.  **Deploy via Console:**
    *   Go to **AWS App Runner Console**.
    *   Select **Source**: Container Registry (ECR).
    *   Choose the image you just pushed.
    *   **Port**: 8000.
    *   **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

---

## 3. Azure Deployment (Container Apps)
*Best for: Serverless containers with easy scaling.*

1.  **Push to ACR:**
    ```bash
    az acr create --resource-group BSH_RG --name bshregistry --sku Basic
    az acr login --name bshregistry
    docker tag bsh-verification-api bshregistry.azurecr.io/bsh-api:v1
    docker push bshregistry.azurecr.io/bsh-api:v1
    ```
2.  **Deploy:**
    ```bash
    az containerapp create \
      --name bsh-verification-api \
      --resource-group BSH_RG \
      --image bshregistry.azurecr.io/bsh-api:v1 \
      --target-port 8000 \
      --ingress 'external' \
      --query properties.configuration.ingress.fqdn
    ```

---

## 4. Google Cloud Deployment (Cloud Run)
*Best for: Fast scaling and simple HTTPS.*

1.  **Push to GCR/Artifact Registry:**
    ```bash
    gcloud auth configure-docker
    docker tag bsh-verification-api gcr.io/<PROJECT_ID>/bsh-api
    docker push gcr.io/<PROJECT_ID>/bsh-api
    ```
2.  **Deploy:**
    ```bash
    gcloud run deploy bsh-verification-api \
      --image gcr.io/<PROJECT_ID>/bsh-api \
      --platform managed \
      --port 8000 \
      --allow-unauthenticated \
      --region europe-west1
    ```

---

## Configuration Note for Frontend
Once deployed, copy the **Public URL** provided by the cloud service (e.g., `https://bsh-api-xyz.run.app`) and update the Flutter app code:

**File**: `lib/screens/dashboard_roi.dart`
```dart
// Change 'http://localhost:8000' to your real cloud URL
final String apiUrl = "https://your-cloud-app-url.com";
```
