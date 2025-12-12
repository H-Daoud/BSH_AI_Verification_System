"""
Model Explainability Module with SHAP Integration.
===================================================

This module provides model interpretation capabilities using SHAP
(SHapley Additive exPlanations) for EU AI Act transparency requirements
and production-line debugging at BSH.

Features:
- SHAP value computation for any sklearn model
- Top-N feature explanations for individual predictions
- Global feature importance analysis
- Human-readable explanation generation
- Visualization support for quality engineers

Author: BSH MLOps Architecture Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
try:
    import shap
except ImportError:
    shap = None # Graceful degradation if SHAP/Numba cannot be installed
from sklearn.base import BaseEstimator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Container for explanation results."""
    prediction: int
    probability: float
    top_features: List[Dict[str, Any]]
    plot_path: Optional[str] = None


class SHAPExplainer:
    """SHAP-based explainer for model interpretation."""

    def __init__(self, model: BaseEstimator, feature_names: List[str]):
        """
        Initialize explainer.
        
        Args:
            model: Trained sklearn model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def fit(self, X_background: pd.DataFrame) -> None:
        """
        Fit explainer using background data.
        """
        if shap:
            try:
                # Initialize appropriate explainer based on model type
                if hasattr(self.model, "tree_") or hasattr(self.model, "estimators_"):
                   self.explainer = shap.TreeExplainer(self.model)
                else:
                   # For KernelExplainer, we usually need a predict function, but passing model might work for some
                   # Better to pass predict_proba for classification
                   predict_fn = getattr(self.model, "predict_proba", self.model.predict)
                   self.explainer = shap.KernelExplainer(predict_fn, X_background)
            except Exception as e:
                logger.warning(f"Failed to initialize SHAP explainer: {e}")
        else:
            logger.warning("SHAP library not found. Explainability features will be disabled.")

    def explain_local(self, X_instance: pd.DataFrame, top_n: int = 3) -> ExplanationResult:
        """
        Explain a single prediction.
        """
        # Get prediction first
        prob = 0.0
        try:
            if hasattr(self.model, "predict_proba"):
                prob = self.model.predict_proba(X_instance)[0][1]
            else:
                 prob = float(self.model.predict(X_instance)[0])
        except Exception:
            pass # Fallback
            
        pred = int(prob > 0.5) # Assuming threshold 0.5 for explanation context

        if self.explainer is None:
            # Return dummy result if SHAP failed
            return ExplanationResult(
                prediction=pred,
                probability=prob,
                top_features=[{"name": "SHAP Disabled", "value": 0.0, "shap_value": 0.0, "impact": "N/A"}]
            )

        try:
            shap_values = self.explainer(X_instance)
            
            # Extract values for the first (and only) instance
            if hasattr(shap_values, 'values'):
                values = shap_values.values
                if len(values.shape) > 1 and values.shape[-1] > 1:
                     # multiclass or binary, take positive class (index 1)
                     values = values[0][:, 1] if values.shape[2] == 2 else values[0]
                else:
                     values = values[0]
            elif isinstance(shap_values, (list, np.ndarray)):
                 values = shap_values[0] # KernelExplainer might return list
            else:
                 values = np.zeros(len(self.feature_names))

            # Sort features by absolute SHAP value
            indices = np.argsort(np.abs(values))[::-1]
            
            top_features = []
            for i in indices[:top_n]:
                if i < len(self.feature_names):
                    val = float(X_instance.iloc[0, i])
                    shap_val = float(values[i])
                    feature = {
                        "name": self.feature_names[i],
                        "value": val,
                        "shap_value": shap_val,
                        "impact": "Increases Defect Risk" if shap_val > 0 else "Decreases Defect Risk"
                    }
                    top_features.append(feature)

            return ExplanationResult(
                prediction=pred,
                probability=prob,
                top_features=top_features
            )
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return ExplanationResult(
                prediction=pred,
                probability=prob,
                top_features=[{"name": "Error", "value": 0.0, "shap_value": 0.0, "impact": str(e)}]
            )

    def generate_summary_plot(self, X_data: pd.DataFrame, save_path: str) -> None:
        """Generate global summary plot."""
        if not shap or not self.explainer:
            return
            
        try:
            import matplotlib.pyplot as plt
            shap_values = self.explainer(X_data)
            plt.figure()
            shap.summary_plot(shap_values, X_data, show=False)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to generate summary plot: {e}")

def main():
    """Test usage."""
    print("SHAP Explainer Module")

if __name__ == "__main__":
    main()
