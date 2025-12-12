"""
Pydantic Schemas for API Validation.

Defines request/response models for the verification API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request schema for anomaly prediction."""
    
    device_id: str = Field(..., description="Unique device identifier")
    acoustic_features: List[float] = Field(..., description="Acoustic feature vector")
    timestamp: Optional[datetime] = Field(default=None, description="Measurement timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "device_id": "DW-2024-001",
                "acoustic_features": [0.1, 0.2, 0.3, 0.4, 0.5],
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for anomaly prediction."""
    
    device_id: str
    prediction: str = Field(..., description="normal or anomaly")
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: Optional[dict] = Field(default=None, description="SHAP explanations")
    processed_at: datetime


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str
    service: str
    version: str
