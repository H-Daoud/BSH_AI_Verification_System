"""
ETL Pipeline for Acoustic Data Processing.
==========================================

This module handles the extraction, transformation, and loading of
acoustic sensor data from dishwasher production lines at BSH.

Features:
- Raw acoustic data ingestion from multiple sources
- FFT (Fast Fourier Transform) for frequency domain analysis
- Data validation and quality checks
- Configurable pipeline stages

Author: BSH MLOps Architecture Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.fft import fft, fftfreq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class ETLError(Exception):
    """Base exception for ETL pipeline errors."""
    pass


class ExtractionError(ETLError):
    """Raised when data extraction fails."""
    pass


class TransformationError(ETLError):
    """Raised when data transformation fails."""
    pass


class ValidationError(ETLError):
    """Raised when data validation fails."""
    pass


class LoadError(ETLError):
    """Raised when data loading fails."""
    pass


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class DataSource(Enum):
    """Supported data source types."""
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    DATABASE = "database"


class OutputFormat(Enum):
    """Supported output formats."""
    CSV = "csv"
    PARQUET = "parquet"
    FEATHER = "feather"


@dataclass
class PipelineConfig:
    """Configuration for the ETL pipeline."""
    
    # Source configuration
    source_path: Path
    source_type: DataSource = DataSource.CSV
    
    # FFT configuration
    sampling_rate: float = 44100.0  # Hz (audio sampling rate)
    fft_window_size: int = 2048
    overlap_ratio: float = 0.5
    
    # Output configuration
    output_path: Path = field(default_factory=lambda: Path("data/processed"))
    output_format: OutputFormat = OutputFormat.PARQUET
    
    # Validation thresholds
    min_vibration_level: float = 0.0
    max_vibration_level: float = 100.0
    min_signal_length: int = 1000  # minimum samples
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if isinstance(self.source_path, str):
            self.source_path = Path(self.source_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)


@dataclass
class FFTResult:
    """Container for FFT analysis results."""
    
    frequencies: np.ndarray
    magnitudes: np.ndarray
    phase: np.ndarray
    dominant_frequency: float
    spectral_centroid: float
    spectral_bandwidth: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert FFT result to dictionary for DataFrame storage."""
        return {
            "dominant_frequency": self.dominant_frequency,
            "spectral_centroid": self.spectral_centroid,
            "spectral_bandwidth": self.spectral_bandwidth,
            "frequency_bins": len(self.frequencies),
            "max_magnitude": float(np.max(self.magnitudes)),
            "mean_magnitude": float(np.mean(self.magnitudes)),
        }


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""
    
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    records_extracted: int = 0
    records_transformed: int = 0
    records_loaded: int = 0
    records_failed: int = 0
    validation_errors: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        """Calculate pipeline duration in seconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of transformation."""
        total = self.records_transformed + self.records_failed
        if total == 0:
            return 0.0
        return self.records_transformed / total


# =============================================================================
# FFT PROCESSOR
# =============================================================================

class FFTProcessor:
    """
    Fast Fourier Transform processor for acoustic signal analysis.
    
    Transforms time-domain acoustic signals into frequency domain
    for anomaly detection feature extraction.
    """
    
    def __init__(
        self,
        sampling_rate: float = 44100.0,
        window_size: int = 2048,
        overlap_ratio: float = 0.5
    ) -> None:
        """
        Initialize FFT processor.
        
        Args:
            sampling_rate: Sampling rate in Hz.
            window_size: Number of samples per FFT window.
            overlap_ratio: Overlap between consecutive windows (0.0 to 0.99).
        
        Raises:
            ValueError: If parameters are out of valid range.
        """
        if sampling_rate <= 0:
            raise ValueError(f"Sampling rate must be positive, got {sampling_rate}")
        if window_size <= 0 or (window_size & (window_size - 1)) != 0:
            raise ValueError(f"Window size must be a positive power of 2, got {window_size}")
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError(f"Overlap ratio must be in [0, 1), got {overlap_ratio}")
        
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.hop_size = int(window_size * (1 - overlap_ratio))
        
        logger.info(
            f"FFT Processor initialized: rate={sampling_rate}Hz, "
            f"window={window_size}, hop={self.hop_size}"
        )
    
    def apply_window(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply Hanning window to reduce spectral leakage.
        
        Args:
            signal: Input signal array.
            
        Returns:
            Windowed signal.
        """
        window = np.hanning(len(signal))
        return signal * window
    
    def compute_fft(self, signal: np.ndarray) -> FFTResult:
        """
        Compute FFT of a signal segment.
        
        Args:
            signal: Time-domain signal array.
            
        Returns:
            FFTResult containing frequency analysis data.
            
        Raises:
            TransformationError: If FFT computation fails.
        """
        try:
            # Validate input
            if len(signal) == 0:
                raise TransformationError("Cannot compute FFT of empty signal")
            
            # Pad signal to window size if necessary
            if len(signal) < self.window_size:
                signal = np.pad(signal, (0, self.window_size - len(signal)))
            elif len(signal) > self.window_size:
                signal = signal[:self.window_size]
            
            # Apply window function
            windowed_signal = self.apply_window(signal)
            
            # Compute FFT
            fft_result = fft(windowed_signal)
            frequencies = fftfreq(len(windowed_signal), 1 / self.sampling_rate)
            
            # Take positive frequencies only
            positive_mask = frequencies >= 0
            frequencies = frequencies[positive_mask]
            fft_positive = fft_result[positive_mask]
            
            # Compute magnitude and phase
            magnitudes = np.abs(fft_positive)
            phase = np.angle(fft_positive)
            
            # Compute spectral features
            dominant_freq = frequencies[np.argmax(magnitudes)]
            spectral_centroid = self._compute_spectral_centroid(frequencies, magnitudes)
            spectral_bandwidth = self._compute_spectral_bandwidth(
                frequencies, magnitudes, spectral_centroid
            )
            
            return FFTResult(
                frequencies=frequencies,
                magnitudes=magnitudes,
                phase=phase,
                dominant_frequency=float(dominant_freq),
                spectral_centroid=spectral_centroid,
                spectral_bandwidth=spectral_bandwidth,
            )
            
        except Exception as e:
            logger.error(f"FFT computation failed: {e}")
            raise TransformationError(f"FFT computation failed: {e}") from e
    
    def _compute_spectral_centroid(
        self,
        frequencies: np.ndarray,
        magnitudes: np.ndarray
    ) -> float:
        """
        Compute spectral centroid (center of mass of spectrum).
        
        Args:
            frequencies: Frequency bins.
            magnitudes: Magnitude values.
            
        Returns:
            Spectral centroid in Hz.
        """
        magnitude_sum = np.sum(magnitudes)
        if magnitude_sum == 0:
            return 0.0
        return float(np.sum(frequencies * magnitudes) / magnitude_sum)
    
    def _compute_spectral_bandwidth(
        self,
        frequencies: np.ndarray,
        magnitudes: np.ndarray,
        centroid: float
    ) -> float:
        """
        Compute spectral bandwidth (spread around centroid).
        
        Args:
            frequencies: Frequency bins.
            magnitudes: Magnitude values.
            centroid: Spectral centroid value.
            
        Returns:
            Spectral bandwidth in Hz.
        """
        magnitude_sum = np.sum(magnitudes)
        if magnitude_sum == 0:
            return 0.0
        variance = np.sum(magnitudes * (frequencies - centroid) ** 2) / magnitude_sum
        return float(np.sqrt(variance))
    
    def process_signal(
        self,
        signal: np.ndarray,
        aggregate: bool = True
    ) -> Union[FFTResult, List[FFTResult]]:
        """
        Process entire signal with sliding window FFT.
        
        Args:
            signal: Full time-domain signal.
            aggregate: If True, return aggregated result; else return per-window results.
            
        Returns:
            FFTResult or list of FFTResult objects.
        """
        if len(signal) < self.window_size:
            logger.warning(
                f"Signal length ({len(signal)}) < window size ({self.window_size}). "
                "Processing entire signal as single window."
            )
            return self.compute_fft(signal)
        
        results: List[FFTResult] = []
        
        for start in range(0, len(signal) - self.window_size + 1, self.hop_size):
            window = signal[start:start + self.window_size]
            result = self.compute_fft(window)
            results.append(result)
        
        if not aggregate:
            return results
        
        # Aggregate results
        return self._aggregate_results(results)
    
    def _aggregate_results(self, results: List[FFTResult]) -> FFTResult:
        """
        Aggregate multiple FFT results into single summary.
        
        Args:
            results: List of FFT results from windowed analysis.
            
        Returns:
            Aggregated FFTResult.
        """
        if not results:
            raise TransformationError("No FFT results to aggregate")
        
        # Use first result as template for frequencies
        avg_magnitudes = np.mean([r.magnitudes for r in results], axis=0)
        avg_phase = np.mean([r.phase for r in results], axis=0)
        
        return FFTResult(
            frequencies=results[0].frequencies,
            magnitudes=avg_magnitudes,
            phase=avg_phase,
            dominant_frequency=float(np.mean([r.dominant_frequency for r in results])),
            spectral_centroid=float(np.mean([r.spectral_centroid for r in results])),
            spectral_bandwidth=float(np.mean([r.spectral_bandwidth for r in results])),
        )


# =============================================================================
# ETL PIPELINE
# =============================================================================

class AcousticETLPipeline:
    """
    Main ETL pipeline for acoustic data processing.
    
    Orchestrates the extraction, transformation (including FFT),
    and loading of acoustic sensor data for anomaly detection.
    
    Example:
        >>> config = PipelineConfig(source_path=Path("data/raw/acoustics.csv"))
        >>> pipeline = AcousticETLPipeline(config)
        >>> metrics = pipeline.run()
        >>> print(f"Processed {metrics.records_loaded} records")
    """
    
    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize the ETL pipeline.
        
        Args:
            config: Pipeline configuration object.
        """
        self.config = config
        self.fft_processor = FFTProcessor(
            sampling_rate=config.sampling_rate,
            window_size=config.fft_window_size,
            overlap_ratio=config.overlap_ratio,
        )
        self.metrics = PipelineMetrics()
        
        logger.info(f"ETL Pipeline initialized with config: {config}")
    
    def extract(self) -> pd.DataFrame:
        """
        Extract acoustic data from source.
        
        Returns:
            DataFrame containing raw acoustic data.
            
        Raises:
            ExtractionError: If extraction fails.
        """
        logger.info(f"Extracting data from {self.config.source_path}")
        
        try:
            if not self.config.source_path.exists():
                raise ExtractionError(
                    f"Source file not found: {self.config.source_path}"
                )
            
            # Load based on source type
            if self.config.source_type == DataSource.CSV:
                df = pd.read_csv(self.config.source_path)
            elif self.config.source_type == DataSource.PARQUET:
                df = pd.read_parquet(self.config.source_path)
            elif self.config.source_type == DataSource.JSON:
                df = pd.read_json(self.config.source_path)
            else:
                raise ExtractionError(
                    f"Unsupported source type: {self.config.source_type}"
                )
            
            self.metrics.records_extracted = len(df)
            logger.info(f"Extracted {len(df)} records from source")
            
            return df
            
        except pd.errors.ParserError as e:
            logger.error(f"Failed to parse source file: {e}")
            raise ExtractionError(f"Parse error: {e}") from e
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise ExtractionError(f"Extraction failed: {e}") from e
    
    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate extracted data against quality rules.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Tuple of (valid DataFrame, list of validation errors).
        """
        logger.info("Running data validation...")
        errors: List[str] = []
        valid_mask = pd.Series([True] * len(df), index=df.index)
        
        # Check required columns
        required_columns = ["device_id", "timestamp", "signal_data"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Validate vibration_level if present
        if "vibration_level" in df.columns:
            invalid_vibration = (
                (df["vibration_level"] < self.config.min_vibration_level) |
                (df["vibration_level"] > self.config.max_vibration_level)
            )
            if invalid_vibration.any():
                count = invalid_vibration.sum()
                errors.append(
                    f"{count} records have vibration_level outside "
                    f"[{self.config.min_vibration_level}, {self.config.max_vibration_level}]"
                )
                valid_mask &= ~invalid_vibration
        
        # Validate signal data length
        if "signal_data" in df.columns:
            def check_signal_length(signal: Any) -> bool:
                if isinstance(signal, str):
                    try:
                        signal = json.loads(signal)
                    except (ValueError, TypeError):
                        return False
                if isinstance(signal, (list, np.ndarray)):
                    return len(signal) >= self.config.min_signal_length
                return False
            
            valid_signal = df["signal_data"].apply(check_signal_length)
            if not valid_signal.all():
                count = (~valid_signal).sum()
                errors.append(
                    f"{count} records have signal_data shorter than "
                    f"{self.config.min_signal_length} samples"
                )
                valid_mask &= valid_signal
        
        # Check for null values in critical columns
        if "device_id" in df.columns:
            null_device_ids = df["device_id"].isna()
            if null_device_ids.any():
                count = null_device_ids.sum()
                errors.append(f"{count} records have null device_id")
                valid_mask &= ~null_device_ids
        
        self.metrics.validation_errors = errors
        valid_df = df[valid_mask].copy()
        
        logger.info(
            f"Validation complete: {len(valid_df)}/{len(df)} records passed, "
            f"{len(errors)} error types found"
        )
        
        return valid_df, errors
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data including FFT feature extraction.
        
        Args:
            df: Validated input DataFrame.
            
        Returns:
            Transformed DataFrame with FFT features.
            
        Raises:
            TransformationError: If transformation fails.
        """
        logger.info(f"Transforming {len(df)} records...")
        
        try:
            transformed_records: List[Dict[str, Any]] = []
            
            for idx, row in df.iterrows():
                try:
                    # Parse signal data
                    signal = self._parse_signal(row.get("signal_data"))
                    
                    if signal is None or len(signal) == 0:
                        logger.warning(f"Skipping record {idx}: empty signal data")
                        self.metrics.records_failed += 1
                        continue
                    
                    # Compute FFT features
                    fft_result = self.fft_processor.process_signal(signal, aggregate=True)
                    
                    # Build transformed record
                    record = {
                        "device_id": row.get("device_id"),
                        "timestamp": row.get("timestamp"),
                        "original_signal_length": len(signal),
                        **fft_result.to_dict(),
                    }
                    
                    # Include additional columns if present
                    for col in ["vibration_level", "temperature", "production_line", "is_anomaly"]:
                        if col in row:
                            record[col] = row[col]
                    
                    transformed_records.append(record)
                    self.metrics.records_transformed += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to transform record {idx}: {e}")
                    self.metrics.records_failed += 1
            
            if not transformed_records:
                raise TransformationError("No records successfully transformed")
            
            result_df = pd.DataFrame(transformed_records)
            
            logger.info(
                f"Transformation complete: {len(result_df)} records, "
                f"{self.metrics.records_failed} failed"
            )
            
            return result_df
            
        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            raise TransformationError(f"Transformation failed: {e}") from e
    
    def _parse_signal(self, signal_data: Any) -> Optional[np.ndarray]:
        """
        Parse signal data from various formats to numpy array.
        
        Args:
            signal_data: Signal data in various formats.
            
        Returns:
            Numpy array of signal values, or None if parsing fails.
        """
        if signal_data is None:
            return None
        
        try:
            if isinstance(signal_data, np.ndarray):
                return signal_data.astype(np.float64)
            elif isinstance(signal_data, list):
                return np.array(signal_data, dtype=np.float64)
            elif isinstance(signal_data, str):
                # Try parsing as JSON array
                parsed = json.loads(signal_data)
                return np.array(parsed, dtype=np.float64)
            else:
                logger.warning(f"Unknown signal data type: {type(signal_data)}")
                return None
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse signal data: {e}")
            return None
    
    def load(self, df: pd.DataFrame) -> Path:
        """
        Load transformed data to destination.
        
        Args:
            df: Transformed DataFrame to save.
            
        Returns:
            Path to the saved file.
            
        Raises:
            LoadError: If loading fails.
        """
        logger.info(f"Loading {len(df)} records to {self.config.output_path}")
        
        try:
            # Create output directory if needed
            self.config.output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"acoustic_features_{timestamp}"
            
            # Save based on output format
            if self.config.output_format == OutputFormat.PARQUET:
                output_file = self.config.output_path / f"{filename}.parquet"
                df.to_parquet(output_file, index=False)
            elif self.config.output_format == OutputFormat.CSV:
                output_file = self.config.output_path / f"{filename}.csv"
                df.to_csv(output_file, index=False)
            elif self.config.output_format == OutputFormat.FEATHER:
                output_file = self.config.output_path / f"{filename}.feather"
                df.to_feather(output_file)
            else:
                raise LoadError(f"Unsupported output format: {self.config.output_format}")
            
            self.metrics.records_loaded = len(df)
            logger.info(f"Successfully saved {len(df)} records to {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Load failed: {e}")
            raise LoadError(f"Load failed: {e}") from e
    
    def run(self) -> PipelineMetrics:
        """
        Execute the complete ETL pipeline.
        
        Returns:
            PipelineMetrics with execution statistics.
            
        Raises:
            ETLError: If any pipeline stage fails.
        """
        logger.info("=" * 60)
        logger.info("Starting Acoustic ETL Pipeline")
        logger.info("=" * 60)
        
        self.metrics = PipelineMetrics()
        
        try:
            # Extract
            raw_df = self.extract()
            
            # Validate
            valid_df, errors = self.validate(raw_df)
            if errors:
                logger.warning(f"Validation found {len(errors)} error types:")
                for err in errors:
                    logger.warning(f" - {err}")
            
            # Transform
            transformed_df = self.transform(valid_df)
            
            # Load
            output_path = self.load(transformed_df)
            
            self.metrics.end_time = datetime.now()
            
            logger.info("=" * 60)
            logger.info("Pipeline completed successfully!")
            logger.info(f"Duration: {self.metrics.duration_seconds:.2f}s")
            logger.info(f"Records: {self.metrics.records_loaded} loaded")
            logger.info(f"Success Rate: {self.metrics.success_rate:.1%}")
            logger.info(f"Output: {output_path}")
            logger.info("=" * 60)
            
            return self.metrics
            
        except ETLError:
            self.metrics.end_time = datetime.now()
            raise
        except Exception as e:
            self.metrics.end_time = datetime.now()
            logger.error(f"Pipeline failed with unexpected error: {e}")
            raise ETLError(f"Pipeline failed: {e}") from e


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> None:
    """CLI entry point for running the ETL pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="BSH Acoustic Data ETL Pipeline"
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to source data file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Output directory path"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet", "feather"],
        default="parquet",
        help="Output format"
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=44100.0,
        help="Audio sampling rate in Hz"
    )
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        source_path=args.source,
        output_path=args.output,
        output_format=OutputFormat(args.format),
        sampling_rate=args.sampling_rate,
    )
    
    pipeline = AcousticETLPipeline(config)
    metrics = pipeline.run()
    
    print(f"\nPipeline completed with {metrics.success_rate:.1%} success rate")


if __name__ == "__main__":
    main()
