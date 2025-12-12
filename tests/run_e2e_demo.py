"""
End-to-End System Test Runner.
==============================

Executes the full Project Antigravity pipeline and logs results to file.

Steps:
1. Data Generation
2. ETL Processing
3. Model Training
4. API Verification (Integration Test)

Output: tests/system_test.log
"""

import logging
import subprocess
import sys
import time
import requests
import signal
import os
from pathlib import Path

# Configure Logging
LOG_FILE = Path("tests/system_test.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SystemTest")

def run_command(cmd, step_name):
    logger.info(f"▶️ STARTING: {step_name}")
    logger.info(f"Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            executable="/bin/bash" 
        )
        logger.info(f"Standard Output:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Standard Error:\n{result.stderr}")
        logger.info(f"✅ COMPLETED: {step_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FAILED: {step_name}")
        logger.error(f"Error Output:\n{e.stderr}")
        return False

def test_api():
    logger.info("▶️ STARTING: API Integration Test")
    
    import socket
    
    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    if is_port_in_use(8000):
        logger.info("ℹ️ API appears to be already running on port 8000. Skipping startup.")
        proc = None
    else:
        # Start API in background
        api_cmd = ["venv/bin/uvicorn", "05_backend_system.api_service.app.main:app", "--port", "8000"]
        
        # Using file path for uvicorn app import needs care.
        # Let's assume running from root.
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        
        proc = subprocess.Popen(
            api_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
            env=env
        )
        
        logger.info(f"API Interface started with PID {proc.pid}. Waiting 5s for startup...")
        time.sleep(5)
    
    try:
        # PING Health
        resp = requests.get("http://localhost:8000/health")
        logger.info(f"Health Check: {resp.status_code} - {resp.json()}")
        
        # Test Verification
        payload = {
           "device_id": "TEST-SYS-01",
           "timestamp": "2025-12-11T12:00:00",
           "product_line": "EcoLine",
           "vibration_val": 95.5,
           "audio_freq_hz": 1200.0,
           "temperature": 45.0
        }
        resp = requests.post("http://localhost:8000/verify", json=payload)
        logger.info(f"Verify Endpoint: {resp.status_code}")
        logger.info(f"Response: {resp.json()}")
        
        if resp.status_code == 200:
            logger.info("✅ API Test Passed")
        else:
            logger.error("❌ API Test Failed")
            
    except Exception as e:
        logger.error(f"❌ API Connection Failed: {e}")
    finally:
        if proc:
            logger.info("Stopping API...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            logger.info("ℹ️ API was running externally, leaving it active.")

def main():
    logger.info("=" * 60)
    logger.info("🚀 BSH PROJECT ANTIGRAVITY - SYSTEM TEST")
    logger.info("=" * 60)
    
    # Ensure venv (or correct python usage)
    python_exe = sys.executable
    logger.info(f"Using Python: {python_exe}")

    # 1. Data Gen
    if not run_command(f"{python_exe} 01_data_engineering/src/generate_synthetic_data.py", "Data Generation"):
        return

    # 2. ETL
    if not run_command(f"{python_exe} 01_data_engineering/src/etl_pipeline.py --source data/raw/sensor_data.csv --output data/processed --format csv", "ETL Pipeline"):
        return

    # 3. Training
    # Find latest file
    try:
        files = sorted(Path("data/processed").glob("*.csv"), key=os.path.getmtime)
        latest_file = files[-1]
        logger.info(f"Found latest feature file: {latest_file}")
    except IndexError:
        logger.error("No processed data found!")
        return

    if not run_command(f"{python_exe} 03_ml_engineering/ml_pipeline/training.py --data-path {latest_file}", "Model Training"):
        return

    # 4. API
    test_api()

    logger.info("=" * 60)
    logger.info("🏁 SYSTEM TEST COMPLETE. Check tests/system_test.log for details.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
