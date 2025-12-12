"""
Model Training Pipeline with MLflow Integration.
=================================================

This module provides the training orchestration for acoustic anomaly
detection models at BSH, with full MLflow experiment tracking for
reproducibility and EU AI Act compliance.

Features:
- Scikit-learn model training with hyperparameter tuning
- MLflow experiment tracking (parameters, metrics, artifacts)
- Model versioning and registry integration
- Cross-validation with stratified splits
- Threshold optimization for False Negative Rate minimization

Author: BSH MLOps Architecture Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class TrainingError(Exception):
    """Base exception for training pipeline errors."""
    pass


class ModelValidationError(TrainingError):
    """Raised when model fails validation criteria."""
    pass


class MLflowError(TrainingError):
    """Raised when MLflow operations fail."""
    pass


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Experiment settings
    experiment_name: str = "bsh-acoustic-anomaly-detection"
    run_name: Optional[str] = None
    
    # Data settings
    test_size: float = 0.2
    validation_size: float = 0.15
    random_state: int = 42
    
    # Model settings
    model_type: str = "random_forest"  # random_forest, gradient_boosting, logistic
    
    # Training settings
    cv_folds: int = 5
    optimize_threshold: bool = True
    target_fnr: float = 0.001  # Target False Negative Rate < 0.1%
    
    # MLflow settings
    tracking_uri: Optional[str] = None
    register_model: bool = True
    model_name: str = "bsh-acoustic-anomaly-detector"
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("models"))
    
    def __post_init__(self) -> None:
        """Initialize paths and validate config."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        if self.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{self.model_type}_{timestamp}"


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    fnr: float  # False Negative Rate
    fpr: float  # False Positive Rate
    confusion_matrix: np.ndarray
    classification_report: str
    optimal_threshold: float = 0.5
    cv_scores: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for MLflow logging."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1,
            "roc_auc": self.roc_auc,
            "false_negative_rate": self.fnr,
            "false_positive_rate": self.fpr,
            "optimal_threshold": self.optimal_threshold,
            "cv_mean_score": np.mean(self.cv_scores) if self.cv_scores else None,
            "cv_std_score": np.std(self.cv_scores) if self.cv_scores else None,
        }


@dataclass
class TrainingResult:
    """Result container for training run."""
    
    model: BaseEstimator
    metrics: TrainingMetrics
    run_id: str
    artifact_uri: str
    model_version: Optional[str] = None
    feature_importance: Optional[Dict[str, float]] = None


# =============================================================================
# MODEL FACTORY
# =============================================================================

class ModelFactory:
    """Factory for creating ML models with default configurations."""
    
    SUPPORTED_MODELS = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression,
        "isolation_forest": IsolationForest,
    }
    
    DEFAULT_PARAMS = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42,
        },
        "logistic_regression": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42,
        },
        "isolation_forest": {
            "n_estimators": 100,
            "contamination": 0.1,
            "random_state": 42,
            "n_jobs": -1,
        },
    }
    
    HYPERPARAMETER_GRIDS = {
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
        },
        "gradient_boosting": {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        },
        "logistic_regression": {
            "C": [0.1, 1.0, 10.0],
        },
    }
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> BaseEstimator:
        """
        Create a model instance with default or custom parameters.
        
        Args:
            model_type: Type of model to create.
            custom_params: Optional custom parameters to override defaults.
            
        Returns:
            Configured model instance.
            
        Raises:
            ValueError: If model_type is not supported.
        """
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported: {list(cls.SUPPORTED_MODELS.keys())}"
            )
        
        model_class = cls.SUPPORTED_MODELS[model_type]
        params = cls.DEFAULT_PARAMS.get(model_type, {}).copy()
        
        if custom_params:
            params.update(custom_params)
        
        logger.info(f"Creating {model_type} model with params: {params}")
        return model_class(**params)
    
    @classmethod
    def get_param_grid(cls, model_type: str) -> Dict[str, List[Any]]:
        """Get hyperparameter grid for model type."""
        return cls.HYPERPARAMETER_GRIDS.get(model_type, {})


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class AcousticModelTrainer:
    """
    Main training pipeline for acoustic anomaly detection models.
    
    Orchestrates model training with MLflow tracking, hyperparameter
    optimization, and threshold tuning for FNR minimization.
    
    Example:
        >>> config = TrainingConfig(model_type="random_forest")
        >>> trainer = AcousticModelTrainer(config)
        >>> result = trainer.train(X_train, y_train, X_test, y_test)
        >>> print(f"FNR: {result.metrics.fnr:.4f}")
    """
    
    def __init__(self, config: TrainingConfig) -> None:
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration object.
        """
        self.config = config
        self._setup_mlflow()
        
        logger.info(f"Trainer initialized for experiment: {config.experiment_name}")
    
    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking."""
        try:
            if self.config.tracking_uri:
                mlflow.set_tracking_uri(self.config.tracking_uri)
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.config.experiment_name,
                    tags={"domain": "acoustic_anomaly_detection", "org": "BSH"}
                )
                logger.info(f"Created MLflow experiment: {self.config.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(self.config.experiment_name)
            
        except Exception as e:
            logger.warning(f"MLflow setup warning: {e}. Continuing with local tracking.")
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            feature_names: Optional list of feature names.
            
        Returns:
            Tuple of (X_array, y_array, feature_names).
        """
        if feature_names is None:
            feature_names = list(X.columns) if hasattr(X, 'columns') else \
                           [f"feature_{i}" for i in range(X.shape[1])]
        
        X_array = X.values if hasattr(X, 'values') else np.array(X)
        y_array = y.values if hasattr(y, 'values') else np.array(y)
        
        # Log class distribution
        unique, counts = np.unique(y_array, return_counts=True)
        for cls, count in zip(unique, counts):
            logger.info(f"Class {cls}: {count} samples ({count/len(y_array)*100:.1f}%)")
        
        return X_array, y_array, feature_names
    
    def find_optimal_threshold(
        self,
        model: BaseEstimator,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """
        Find optimal classification threshold to minimize FNR.
        
        Args:
            model: Trained model with predict_proba method.
            X_val: Validation features.
            y_val: Validation labels.
            
        Returns:
            Optimal threshold value.
        """
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model doesn't support predict_proba. Using default threshold.")
            return 0.5
        
        y_proba = model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
        
        # Find threshold that achieves target FNR (FNR = 1 - Recall)
        target_recall = 1 - self.config.target_fnr
        
        # Find highest threshold that achieves target recall
        valid_idx = recalls[:-1] >= target_recall
        if valid_idx.any():
            optimal_threshold = thresholds[valid_idx][-1]
        else:
            # If target not achievable, use threshold with best recall
            optimal_threshold = thresholds[np.argmax(recalls[:-1])]
        
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        return float(optimal_threshold)
    
    def compute_metrics(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5
    ) -> TrainingMetrics:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            model: Trained model.
            X: Feature array.
            y: True labels.
            threshold: Classification threshold.
            
        Returns:
            TrainingMetrics object.
        """
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X)
            y_proba = y_pred  # For models without probability
        
        # Compute confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Calculate FNR and FPR
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Calculate other metrics
        metrics = TrainingMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, zero_division=0),
            recall=recall_score(y, y_pred, zero_division=0),
            f1=f1_score(y, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
            fnr=fnr,
            fpr=fpr,
            confusion_matrix=cm,
            classification_report=classification_report(y, y_pred),
            optimal_threshold=threshold,
        )
        
        return metrics
    
    def train_with_cv(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[BaseEstimator, List[float]]:
        """
        Train model with cross-validation.
        
        Args:
            model: Model to train.
            X: Feature array.
            y: Target array.
            
        Returns:
            Tuple of (trained_model, cv_scores).
        """
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        # Get CV scores
        cv_scores = cross_val_score(
            model, X, y, cv=cv, scoring='recall'  # Optimize for recall (minimize FNR)
        )
        
        logger.info(f"CV Recall: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Fit on full training data
        model.fit(X, y)
        
        return model, cv_scores.tolist()
    
    def tune_hyperparameters(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Perform hyperparameter tuning with GridSearchCV.
        
        Args:
            model: Base model to tune.
            X: Feature array.
            y: Target array.
            
        Returns:
            Tuple of (best_model, best_params).
        """
        param_grid = ModelFactory.get_param_grid(self.config.model_type)
        
        if not param_grid:
            logger.info("No parameter grid defined. Skipping hyperparameter tuning.")
            model.fit(X, y)
            return model, {}
        
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring='recall',  # Optimize for recall
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def extract_feature_importance(
        self,
        model: BaseEstimator,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Extract feature importance from model.
        
        Args:
            model: Trained model.
            feature_names: List of feature names.
            
        Returns:
            Dictionary of feature importances.
        """
        importance = {}
        
        if hasattr(model, 'feature_importances_'):
            for name, imp in zip(feature_names, model.feature_importances_):
                importance[name] = float(imp)
        elif hasattr(model, 'coef_'):
            coef = model.coef_.flatten()
            for name, imp in zip(feature_names, np.abs(coef)):
                importance[name] = float(imp)
        
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None,
        tune_hyperparams: bool = False
    ) -> TrainingResult:
        """
        Execute full training pipeline with MLflow tracking.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            feature_names: Optional list of feature names.
            tune_hyperparams: Whether to perform hyperparameter tuning.
            
        Returns:
            TrainingResult with model, metrics, and MLflow info.
            
        Raises:
            TrainingError: If training fails.
        """
        logger.info("=" * 60)
        logger.info(f"Starting training: {self.config.run_name}")
        logger.info("=" * 60)
        
        # Prepare data
        X_train_arr, y_train_arr, feature_names = self.prepare_data(
            X_train, y_train, feature_names
        )
        X_test_arr, y_test_arr, _ = self.prepare_data(X_test, y_test, feature_names)
        
        # Start MLflow run
        with mlflow.start_run(run_name=self.config.run_name) as run:
            try:
                # Log configuration
                mlflow.log_params({
                    "model_type": self.config.model_type,
                    "cv_folds": self.config.cv_folds,
                    "target_fnr": self.config.target_fnr,
                    "test_size": self.config.test_size,
                    "n_features": len(feature_names),
                    "n_train_samples": len(X_train_arr),
                    "n_test_samples": len(X_test_arr),
                })
                
                # Create model
                model = ModelFactory.create_model(self.config.model_type)
                
                # Log model parameters
                model_params = model.get_params()
                mlflow.log_params({f"model_{k}": v for k, v in model_params.items()
                                   if v is not None and not callable(v)})
                
                # Hyperparameter tuning or cross-validation
                if tune_hyperparams:
                    model, best_params = self.tune_hyperparameters(
                        model, X_train_arr, y_train_arr
                    )
                    mlflow.log_params({f"tuned_{k}": v for k, v in best_params.items()})
                else:
                    model, cv_scores = self.train_with_cv(
                        model, X_train_arr, y_train_arr
                    )
                
                # Find optimal threshold
                if self.config.optimize_threshold:
                    # Split some training data for threshold optimization
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X_train_arr, y_train_arr,
                        test_size=self.config.validation_size,
                        random_state=self.config.random_state,
                        stratify=y_train_arr
                    )
                    model.fit(X_tr, y_tr)
                    optimal_threshold = self.find_optimal_threshold(model, X_val, y_val)
                    
                    # Refit on full training data
                    model.fit(X_train_arr, y_train_arr)
                else:
                    optimal_threshold = 0.5
                
                # Compute metrics
                metrics = self.compute_metrics(
                    model, X_test_arr, y_test_arr, optimal_threshold
                )
                metrics.cv_scores = cv_scores if not tune_hyperparams else None
                
                # Log metrics
                mlflow.log_metrics(metrics.to_dict())
                
                # Extract feature importance
                feature_importance = self.extract_feature_importance(model, feature_names)
                
                # Log feature importance as artifact
                importance_path = self.config.output_dir / "feature_importance.json"
                self.config.output_dir.mkdir(parents=True, exist_ok=True)
                with open(importance_path, "w") as f:
                    json.dump(feature_importance, f, indent=2)
                mlflow.log_artifact(str(importance_path))
                
                # Log classification report
                report_path = self.config.output_dir / "classification_report.txt"
                with open(report_path, "w") as f:
                    f.write(metrics.classification_report)
                mlflow.log_artifact(str(report_path))
                
                # Log model
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=self.config.model_name if self.config.register_model else None
                )
                
                # Log additional metadata
                mlflow.set_tags({
                    "domain": "acoustic_anomaly_detection",
                    "org": "BSH",
                    "compliance": "EU_AI_Act",
                    "fnr_target_met": str(metrics.fnr <= self.config.target_fnr),
                })
                
                # Create result
                result = TrainingResult(
                    model=model,
                    metrics=metrics,
                    run_id=run.info.run_id,
                    artifact_uri=run.info.artifact_uri,
                    feature_importance=feature_importance,
                )
                
                # Log summary
                logger.info("=" * 60)
                logger.info("Training Complete!")
                logger.info(f"Run ID: {result.run_id}")
                logger.info(f"Accuracy: {metrics.accuracy:.4f}")
                logger.info(f"F1 Score: {metrics.f1:.4f}")
                logger.info(f"ROC-AUC: {metrics.roc_auc:.4f}")
                logger.info(f"False Negative Rate: {metrics.fnr:.4f}")
                logger.info(f"Target FNR: {self.config.target_fnr}")
                logger.info(f"FNR Target Met: {metrics.fnr <= self.config.target_fnr}")
                logger.info("=" * 60)
                
                return result
                
            except Exception as e:
                logger.error(f"Training failed: {e}")
                mlflow.set_tag("error", str(e))
                raise TrainingError(f"Training failed: {e}") from e


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> None:
    """CLI entry point for training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="BSH Acoustic Anomaly Detection - Model Training"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to processed features file"
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="is_anomaly",
        help="Name of target column"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["random_forest", "gradient_boosting", "logistic_regression"],
        default="random_forest",
        help="Model type to train"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="bsh-acoustic-anomaly-detection",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--tune-hyperparams",
        action="store_true",
        help="Perform hyperparameter tuning"
    )
    parser.add_argument(
        "--target-fnr",
        type=float,
        default=0.001,
        help="Target False Negative Rate"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    if args.data_path.suffix == ".parquet":
        df = pd.read_parquet(args.data_path)
    else:
        df = pd.read_csv(args.data_path)
    
    # Split features and target
    X = df.drop(columns=[args.target_column])
    y = df[args.target_column]
    
    # Preprocessing: Drop identifiers and encode categoricals
    drop_cols = ["device_id", "timestamp"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    
    # Handle categorical variables (e.g. production_line)
    X = pd.get_dummies(X, drop_first=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Configure and run training
    config = TrainingConfig(
        experiment_name=args.experiment_name,
        model_type=args.model_type,
        target_fnr=args.target_fnr,
    )
    
    trainer = AcousticModelTrainer(config)
    result = trainer.train(
        X_train, y_train,
        X_test, y_test,
        feature_names=list(X.columns),
        tune_hyperparams=args.tune_hyperparams
    )
    
    print(f"\n✅ Training complete! Run ID: {result.run_id}")
    print(f"📊 FNR: {result.metrics.fnr:.4f} (target: {args.target_fnr})")


if __name__ == "__main__":
    main()
