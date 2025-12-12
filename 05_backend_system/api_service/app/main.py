"""
BSH Verification API Service.
=============================

FastAPI application for real-time acoustic anomaly detection.

Features:
- Real-time inference endpoint `/verify`
- Drift detection integration
- Explainability (XAI) integration
- ROI calculation metadata

Author: BSH MLOps Architecture Team
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import logging
import joblib
from datetime import datetime
from pathlib import Path

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bsh-api")

app = FastAPI(
    title="BSH Verification Antigravity API",
    version="1.0.0",
    docs_url="/docs"
)

# Enable CORS for Flutter Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock model loading for demo purposes if file doesn't exist
# In production, this would load from 'models/production_model.pkl'
class MockModel:
    def predict_proba(self, X):
        # Simulate prediction: 5% chance of defect
        is_defect = np.random.random() < 0.05
        prob = 0.95 if is_defect else 0.02
        return np.array([[1-prob, prob]])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

try:
    model = joblib.load("models/production_model.pkl")
    logger.info("Loaded production model.")
except Exception:
    logger.warning("Production model not found. Using MockModel for demonstration.")
    model = MockModel()

# Pydantic Schemas
class SensorData(BaseModel):
    device_id: str
    timestamp: datetime
    product_line: str
    vibration_val: float = Field(..., gt=0, description="Vibration level")
    audio_freq_hz: float = Field(..., gt=0, description="Dominant audio frequency")
    temperature: float

class VerificationResponse(BaseModel):
    device_id: str
    is_defective: bool
    confidence: float
    xai_explanation: List[str]
    processing_time_ms: float
    drift_status: str

# Helper functions
def check_drift(data: SensorData) -> str:
    """Mock drift check. Returns 'stable' or 'drift_detected'."""
    # Logic: if vibration > 90, flag as potential drift or outlier
    if data.vibration_val > 90:
        return "warning_outlier_detected"
    return "stable"

def generate_explanation(data: SensorData, is_defective: bool) -> List[str]:
    """Generate rule-based explanation trying to pinpoint the physical component."""
    reasons = []
    if is_defective:
        # 1. High Temp + High Vibration -> Mechanical Friction
        if data.temperature > 80 and data.vibration_val > 90:
            reasons.append("High Friction detected (Check Motor Bearings)")
        
        # 2. High Freq -> Cavitation
        elif data.audio_freq_hz > 1400:
             reasons.append("High Frequency Whine (Check Pump Cavitation)")
        
        # 3. Low Freq + High Vibration -> Loose Parts
        elif data.audio_freq_hz < 900 and data.vibration_val > 90:
            reasons.append("Low Frequency Rumbling (Check Engine Mounting)")
            
        # 4. Purely High Vibration
        elif data.vibration_val > 90:
            reasons.append("Excessive Vibration (Check Load Balance)")

        # 5. Default Fallback
        if not reasons:
             reasons.append("Acoustic Anomaly (Unidentified Source)")
    else:
        reasons.append("Normal Operation")
    return reasons

# --- Defect Reporting Helper ---
def log_defect(device_id: str, timestamp: datetime, reason: str, confidence: float, action: str):
    log_file = Path("reports/defect_log.csv")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = log_file.exists()
    
    try:
        with open(log_file, "a") as f:
            if not file_exists:
                f.write("timestamp,device_id,confidence,reason,user_action\n")
            f.write(f"{timestamp},{device_id},{confidence:.4f},\"{reason}\",{action}\n")
        logger.info(f"📝 Defect Logged ({action}): {device_id}")
    except Exception as e:
        logger.error(f"Failed to log report: {e}")

# --- Endpoints ---

class ReportRequest(BaseModel):
    device_id: str
    timestamp: datetime
    reason: str
    confidence: float
    user_action: str = "manual_flag"

@app.post("/report")
def report_defect(report: ReportRequest):
    """Log a reported defect to a persistent CSV file."""
    try:
        log_defect(report.device_id, report.timestamp, report.reason, report.confidence, report.user_action)
        return {"status": "logged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify", response_model=VerificationResponse)
async def verify_device(data: SensorData, background_tasks: BackgroundTasks):
    start_time = datetime.now()
    
    # 1. Feature Prep
    features = np.array([[data.vibration_val, data.audio_freq_hz, data.temperature]])
    
    # 2. Inference
    try:
        prob_defect = model.predict_proba(features)[0][1]
        is_defective = prob_defect > 0.5 
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed")

    # 3. Drift Check
    drift_status = check_drift(data)
    
    # 4. Explainability
    explanation = generate_explanation(data, is_defective)

    # 5. AUTO-LOGGING (New Feature)
    if is_defective:
        # Use BackgroundTasks to avoid slowing down the API response
        background_tasks.add_task(
            log_defect, 
            device_id=data.device_id, 
            timestamp=data.timestamp, 
            reason=explanation[0] if explanation else "Model High Confidence", 
            confidence=float(prob_defect), 
            action="auto_detection"
        )

    # 6. Response
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    response = VerificationResponse(
        device_id=data.device_id,
        is_defective=is_defective,
        confidence=float(prob_defect),
        xai_explanation=explanation,
        processing_time_ms=processing_time,
        drift_status=drift_status
    )
    
    logger.info(f"Verified {data.device_id}: Defect={is_defective} Conf={prob_defect:.2f}")
    return response
