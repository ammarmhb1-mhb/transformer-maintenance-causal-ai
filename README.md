# A Multi-Method Causal AI Framework for Robust Remote Maintenance

## Overview
This repository contains the complete Python code for the paper:
"A Multi-Method Causal AI Framework for Robust Remote Maintenance in Energy Systems with Trust Calibration".

## File Description
- `transformer_maintenance_ai.py`: The main script containing the entire analysis pipeline, including:
  - Isolation Forest (Anomaly Detection)
  - Causal Inference (DoWhy + DR-Learner)
  - Causal Forest (Heterogeneous Effects)
  - Adversarial Defense (Label Poisoning Simulation)
  - Human Study Analysis (Trust Calibration)

## Requirements
To run this code, install the following dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn dowhy econml statsmodels scipy
