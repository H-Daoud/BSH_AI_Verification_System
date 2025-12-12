#!/bin/bash
set -e

# Define project root
PROJECT_NAME="bsh-verification-antigravity"

echo "Creating project structure for: $PROJECT_NAME"

# Create root directory
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Create directories
echo "Creating directories..."
mkdir -p .azure-pipelines
mkdir -p 00_business_context
mkdir -p 01_data_engineering/configs/great_expectations
mkdir -p 01_data_engineering/src
mkdir -p 01_data_engineering/tests
mkdir -p 02_data_science_research/notebooks
mkdir -p 02_data_science_research/reports
mkdir -p 02_data_science_research/sandbox_data
mkdir -p 03_ml_engineering/ml_pipeline
mkdir -p 03_ml_engineering/tracking
mkdir -p 03_ml_engineering/model_registry
mkdir -p 04_governance_compliance/eu_ai_act
mkdir -p 04_governance_compliance/ethics
mkdir -p 04_governance_compliance/model_cards
mkdir -p 04_governance_compliance/audit_logs
mkdir -p 05_backend_system/api_service/app
mkdir -p 05_backend_system/infrastructure
mkdir -p 06_frontend_interactive/lib/screens
mkdir -p 06_frontend_interactive/lib/models
mkdir -p data

# Create empty files
echo "Creating empty files..."
touch .azure-pipelines/build-and-test.yaml .azure-pipelines/security-scan.yaml .azure-pipelines/deploy-prod.yaml
touch 00_business_context/problem_definition.md 00_business_context/roi_calculator.xlsx 00_business_context/poc_criteria.md 00_business_context/stakeholders_map.md
touch 01_data_engineering/configs/drift_config.yaml 01_data_engineering/src/etl_pipeline.py 01_data_engineering/src/quality_gates.py
touch 02_data_science_research/notebooks/01_eda_acoustics.ipynb 02_data_science_research/notebooks/02_feature_selection.ipynb 02_data_science_research/notebooks/03_model_selection_poc.ipynb 02_data_science_research/reports/research_summary.pdf
touch 03_ml_engineering/ml_pipeline/training.py 03_ml_engineering/ml_pipeline/evaluation.py 03_ml_engineering/ml_pipeline/explainability.py 03_ml_engineering/tracking/mlflow_project
touch 04_governance_compliance/eu_ai_act/risk_assessment.md 04_governance_compliance/eu_ai_act/technical_docs.pdf 04_governance_compliance/ethics/fairness_check.py 04_governance_compliance/model_cards/v1_production.md
touch 05_backend_system/api_service/app/main.py 05_backend_system/api_service/app/router.py 05_backend_system/api_service/app/schemas.py 05_backend_system/api_service/Dockerfile
touch 06_frontend_interactive/lib/screens/dashboard_roi.dart 06_frontend_interactive/lib/screens/verify_detail.dart 06_frontend_interactive/pubspec.yaml
touch data/raw.dvc data/processed.dvc data/features.dvc
touch dvc.yaml

# Populate basic files
echo "Populating basic configuration files..."

# .gitignore
cat << EOF > .gitignore
# Python
__pycache__/
*.py[cod]
*\$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.pytest_cache/
.coverage
htmlcov/
.tox/
.nox/

# Virtual Environments
venv/
env/
.env
.venv

# Jupyter
.ipynb_checkpoints

# VS Code
.vscode/

# MacOS
.DS_Store

# DVC
/data/
EOF

# README.md
cat << EOF > README.md
# bsh-verification-antigravity

Project structure for BSH verification antigravity.

## Directory Structure

- \`.azure-pipelines/\`: CI/CD Definitions
- \`00_business_context/\`: Business Objectives & ROI
- \`01_data_engineering/\`: ETL & Data Monitoring
- \`02_data_science_research/\`: Research & EDA
- \`03_ml_engineering/\`: Model Analysis & Training Pipeline
- \`04_governance_compliance/\`: Governance, Ethics & Compliance
- \`05_backend_system/\`: Backend System
- \`06_frontend_interactive/\`: Frontend App
- \`data/\`: Data storage
EOF

# pyproject.toml
cat << EOF > pyproject.toml
[project]
name = "bsh-verification-antigravity"
version = "0.1.0"
description = "BSH verification antigravity project"
readme = "README.md"
requires-python = ">=3.9"
dependencies = []
EOF

# .pre-commit-config.yaml
cat << EOF > .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
EOF

echo "Setup complete! Project created at $(pwd)"
