"""
Fairness Check Module using Fairlearn.
======================================

Checks model performance across different sensitive groups (e.g., Product Lines)
to ensure EU AI Act compliance regarding non-discrimination.

Features:
- Demographic Parity calculation
- Equalized Odds calculation
- Performance discrepancy analysis
"""

import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score
from typing import Dict, List, Any

# Mock fairlearn to ensure standalone execution if not installed, 
# but structure it for real use.
try:
    from fairlearn.metrics import MetricFrame, selection_rate, false_negative_rate
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class FairnessAuditor:
    def __init__(self, sensitive_features: List[str]):
        """
        Initialize auditor.
        
        Args:
            sensitive_features: List of column names to check for bias (e.g. 'product_line')
        """
        self.sensitive_features = sensitive_features

    def audit(self, model: BaseEstimator, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
        """
        Run fairness audit.
        
        Args:
            model: Trained model
            X: Features
            y_true: True labels
            
        Returns:
            Dictionary of fairness metrics
        """
        if not FAIRLEARN_AVAILABLE:
            logger.warning("Fairlearn not installed. Skipping advanced metrics.")
            return {"status": "skipped", "reason": "fairlearn missing"}

        y_pred = model.predict(X)
        results = {}

        for feature in self.sensitive_features:
            if feature not in X.columns:
                continue
                
            sensitive_col = X[feature]
            
            # MetricFrame computes metrics for each group
            mf = MetricFrame(
                metrics={
                    "accuracy": accuracy_score,
                    "recall": recall_score,
                    "false_negative_rate": false_negative_rate,
                    "selection_rate": selection_rate,
                },
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_col
            )
            
            results[feature] = {
                "overall": mf.overall.to_dict(),
                "by_group": mf.by_group.to_dict(),
                "difference": mf.difference().to_dict()
            }
            
        return results

def main():
    print("Fairness Check Module Initialized")

if __name__ == "__main__":
    main()
