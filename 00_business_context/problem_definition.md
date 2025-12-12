# Problem Definition: Acoustic Anomaly Detection in Dishwasher Production

**Document ID:** BSH-PA-BUS-001  
**Version:** 1.0.0  
**Last Updated:** 2024-01-15  
**Owner:** BSH Quality Assurance & MLOps Team  
**Classification:** Internal

---

## Executive Summary

BSH Hausgeräte Group seeks to implement an AI-powered acoustic verification system ("Project Antigravity") to detect production anomalies in dishwashers before shipment. This initiative directly addresses quality control gaps that result in warranty claims, customer dissatisfaction, and brand reputation risk.

---

## 1. Business Problem Statement

### 1.1 Current State

| Metric | Current Value | Impact |
|--------|---------------|--------|
| **False Negative Rate** | ~2.5% | Defective units shipped to customers |
| **Annual Warranty Claims** | €4.2M | Direct cost of repairs/replacements |
| **Manual Inspection Time** | 45 sec/unit | Production bottleneck |
| **Customer Returns** | 1.8% | Brand reputation damage |

### 1.2 Root Cause Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUALITY CONTROL GAP                          │
├─────────────────────────────────────────────────────────────────┤
│  Human Inspection Limitations:                                  │
│  • Auditory fatigue after 2-3 hours                            │
│  • Subjective interpretation of "normal" sounds                 │
│  • Inability to detect sub-audible frequency anomalies         │
│  • Inconsistency across shifts and inspectors                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Target State

Implement an AI-based acoustic anomaly detection system that:
1. Achieves **False Negative Rate < 0.1%** (25x improvement)
2. Reduces inspection time by **60%**
3. Provides **explainable predictions** for quality engineers
4. Maintains **full EU AI Act compliance**

---

## 2. Business Objectives & KPIs

### 2.1 Primary Objective

> **Reduce False Negative Rate to below 0.1%** while maintaining False Positive Rate below 2% to avoid unnecessary rework costs.

### 2.2 Key Performance Indicators

| KPI | Baseline | Target | Measurement |
|-----|----------|--------|-------------|
| False Negative Rate (FNR) | 2.5% | **< 0.1%** | Weekly validation batch |
| False Positive Rate (FPR) | 1.2% | < 2.0% | Production monitoring |
| Inspection Throughput | 45 sec/unit | 18 sec/unit | Time-motion study |
| Warranty Claim Rate | 1.8% | < 0.5% | Monthly finance report |
| Model Explainability Score | N/A | > 85% | SHAP coverage metric |

### 2.3 Success Criteria for PoC

- [ ] FNR < 0.5% on holdout test set (minimum viable)
- [ ] FNR < 0.1% on holdout test set (target)
- [ ] Inference latency < 100ms per unit
- [ ] Model interpretability via SHAP for 100% of predictions
- [ ] Integration with existing MES (Manufacturing Execution System)

---

## 3. ROI Analysis

### 3.1 Cost-Benefit Model

#### Investment Costs (Year 1)

| Category | Cost (EUR) |
|----------|------------|
| ML Development & Data Engineering | €180,000 |
| Infrastructure (Azure ML, Storage) | €48,000 |
| Sensor Hardware (10 production lines) | €75,000 |
| Integration & Testing | €45,000 |
| Training & Change Management | €22,000 |
| **Total Investment** | **€370,000** |

#### Annual Operating Costs (Year 2+)

| Category | Cost (EUR) |
|----------|------------|
| Cloud Infrastructure | €36,000 |
| Model Maintenance & Retraining | €60,000 |
| Support & Monitoring | €24,000 |
| **Total Annual OpEx** | **€120,000** |

### 3.2 Warranty Savings Formula

The core ROI calculation is based on **warranty cost avoidance**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROI CALCULATION FORMULA                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Annual Warranty Savings = Units × (FNR_old - FNR_new) × C_w    │
│                                                                  │
│  Where:                                                          │
│    Units    = Annual production volume (units/year)             │
│    FNR_old  = Current False Negative Rate (0.025)               │
│    FNR_new  = Target False Negative Rate (0.001)                │
│    C_w      = Average warranty claim cost per unit (EUR)        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Calculation Example

```python
# BSH Project Antigravity - ROI Calculator

# Input Parameters
annual_production_units = 850_000      # Units per year
fnr_current = 0.025                    # 2.5% current false negative rate
fnr_target = 0.001                     # 0.1% target false negative rate
avg_warranty_cost = 185.00             # EUR per warranty claim

# Warranty Savings Calculation
defects_avoided = annual_production_units * (fnr_current - fnr_target)
annual_warranty_savings = defects_avoided * avg_warranty_cost

# Results
print(f"Defects Avoided: {defects_avoided:,.0f} units/year")
print(f"Annual Warranty Savings: €{annual_warranty_savings:,.0f}")

# Output:
# Defects Avoided: 20,400 units/year
# Annual Warranty Savings: €3,774,000
```

### 3.3 ROI Summary

| Metric | Value |
|--------|-------|
| **Year 1 Net Benefit** | €3,774,000 - €370,000 = **€3,404,000** |
| **Annual Net Benefit (Year 2+)** | €3,774,000 - €120,000 = **€3,654,000** |
| **Payback Period** | **< 2 months** |
| **3-Year ROI** | **2,789%** |

### 3.4 Additional Benefits (Not Quantified)

- **Brand Reputation**: Reduced negative reviews and returns
- **Customer Satisfaction**: Higher NPS scores
- **Operational Efficiency**: Freed inspection capacity for other tasks
- **Regulatory Compliance**: Proactive EU AI Act alignment
- **Data Asset**: Acoustic dataset for future ML applications

---

## 4. Risk Assessment

### 4.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Insufficient training data | Medium | High | Partner with 3 production sites |
| Model drift over time | High | Medium | Automated retraining pipeline |
| Integration complexity | Medium | Medium | Phased rollout approach |
| Latency exceeds requirements | Low | High | Edge deployment option |

### 4.2 Business Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Resistance from QA staff | Medium | Medium | Change management program |
| ROI assumptions incorrect | Low | High | Conservative baseline estimates |
| Regulatory non-compliance | Low | High | EU AI Act review at each phase |

---

## 5. Stakeholder Alignment

### 5.1 Decision Makers

| Role | Name | Responsibility |
|------|------|----------------|
| Executive Sponsor | [TBD] | Budget approval, strategic alignment |
| Project Owner | [TBD] | Delivery accountability |
| Technical Lead | [TBD] | Architecture decisions |

### 5.2 Approval Gates

1. **PoC Approval**: After successful lab validation
2. **Pilot Approval**: After single-line deployment success
3. **Full Rollout**: After multi-line validation

---

## 6. Next Steps

1. ✅ Define problem and ROI model (this document)
2. ⏳ Data collection from pilot production line
3. ⏳ EDA and feature engineering
4. ⏳ Model development and validation
5. ⏳ PoC review with stakeholders
6. ⏳ Pilot deployment planning

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **FNR** | False Negative Rate - proportion of defective units incorrectly classified as normal |
| **FPR** | False Positive Rate - proportion of normal units incorrectly classified as defective |
| **FFT** | Fast Fourier Transform - algorithm for frequency analysis |
| **SHAP** | SHapley Additive exPlanations - model interpretability method |
| **MES** | Manufacturing Execution System |

---

## Appendix B: References

1. BSH Quality Management Policy QM-2024-001
2. EU AI Act - Article 6 (High-Risk AI Systems)
3. ISO 22400 - Manufacturing Operations Management KPIs
4. Internal: Warranty Claims Analysis Report Q3-2023

---

*Document approved for internal distribution. Contact MLOps team for questions.*
