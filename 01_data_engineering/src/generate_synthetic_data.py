"""
Synthetic Data Generator for Project Antigravity.
=================================================

Generates realistic acoustic sensor data for dishwashers to enable
pipeline testing and verification without real production data.
"""

import pandas as pd
import numpy as np
import random
import json
from pathlib import Path

def generate_data(num_samples=1000):
    np.random.seed(42)
    random.seed(42)
    
    print(f"Generating {num_samples} samples...")
    
    data = []
    start_time = pd.Timestamp.now()
    
    for i in range(num_samples):
        is_anomaly = np.random.random() < 0.05  # 5% defect rate
        
        # Simulate sensor readings
        # Normal temp: 60 +/- 5. Anomaly: 85 +/- 10
        temp = np.random.normal(60, 5) if not is_anomaly else np.random.normal(85, 10)
        
        # Normal vibration: 50 +/- 10. Anomaly: 110 +/- 15
        vibration = np.random.normal(50, 10) if not is_anomaly else np.random.normal(110, 15)
        
        # Frequency
        dominant_freq = np.random.normal(1200, 50) if not is_anomaly else np.random.normal(800, 100)
        
        # Production Line
        prod_line = np.random.choice(["Line_A", "Line_B", "Line_C"])
        
        # Signal Data (2048 samples)
        # Using json.dumps to ensure valid JSON array in CSV string
        signal = list(np.random.normal(0, 1, 2048))
        signal_str = json.dumps(signal)
        
        row = {
            "device_id": f"DW-2024-{i:06d}",
            "timestamp": start_time - pd.Timedelta(minutes=i),
            "production_line": prod_line,
            "temperature": temp,
            "vibration_level": vibration,
            "frequency_mean": dominant_freq, 
            "signal_data": signal_str,
            "is_anomaly": is_anomaly
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Save
    output_path = Path("data/raw/sensor_data.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} samples at {output_path}")
    return df

if __name__ == "__main__":
    generate_data(1000)
