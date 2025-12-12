# EU AI Act - High-Risk AI System Risk Assessment

**Project:** BSH Antigravity (Acoustic Anomaly Detection)  
**Date:** 2025-12-11  
**Status:** DRAFT

---

## 1. System Overview
**Purpose:** Automated quality verification of dishwashers using acoustic sensor data.  
**AI System Type:** Machine Learning (Supervised Classification).  
**Deployment Context:** Factory production line, edge deployment.

## 2. Risk Classification
**Classification:** High-Risk (Annex III, Critical Infrastructure/Safety Component)  
*Justification:* The system is a safety component of a product covered by Union harmonisation legislation (Machinery Directive), or acts as a safety function. A failure (False Negative) results in defective appliances reaching consumers, potentially causing safety hazards (leakage, electrical) or significant economic loss.

## 3. Risks & Mitigation Strategies

| ID | Risk Description | Severity | Likelihood | Mitigation Strategy | Residual Risk |
|----|------------------|----------|------------|---------------------|---------------|
| R01 | **Model Bias / Discrimination**<br>Model performs worse on specific models (e.g., EcoLine vs Premium) due to unbalanced training data. | High | Medium | • Stratified sampling by product line<br>• `FairnessAuditor` checks in pipeline<br>• Reweighting techniques | Low |
| R02 | **Concept Drift**<br>Acoustic signatures change due to new parts supplier or wear in assembly tools. | Medium | High | • Drift detection (Great Expectations)<br>• Monthly retraining with human-in-the-loop<br>• Monitoring dashboard | Low |
| R03 | **False Negatives (Safety)**<br>System marks defective unit as 'OK'. | Critical | Low | • Optimize threshold for Recall > 99.9%<br>• Human audit of 5% of 'OK' units<br>• Failsafe mechanism (if unsure, flag as NOK) | Low |
| R04 | **Adversarial Attacks**<br>Manipulation of audio to hide defects. | Low | Low | • Sensor tampering detection<br>• Secure data pipeline (TLS/Encryption) | Low |
| R05 | **Lack of Explainability**<br>Operator cannot understand why a unit was rejected. | Medium | High | • SHAP integration (`explainability.py`)<br>• User-friendly dashboard | Low |

## 4. Technical Documentation (Article 11)
- [x] **System Architecture:** See `README.md` and codebase structure.
- [ ] **Data Governance:** Training data metadata, data provenance (DVC).
- [x] **Monitoring:** Logging in place (MLflow).
- [ ] **Cybersecurity:** Trivy scans implemented.

## 5. Human Oversight (Article 14)
- **Mode:** Human-in-the-loop for borderline cases.
- **Training:** QA Engineers trained to interpret SHAP plots.
- **Stop Button:** Operators can override AI decision manually via the Flutter App.

---
**Sign-off:**
_________________________  
Lead AI Architect
