# AI-Based Cloud Failure Prediction, Risk Assessment & Automated Self-Healing System

An AI-driven DevOps system for predicting cloud/task failures, detecting anomalous behaviour, estimating operational risk, recommending recovery actions, and supporting automated self-healing workflows.

## Project Overview

Modern cloud environments generate large volumes of telemetry such as CPU usage, memory usage, task activity, resource requests, and other operational signals. Detecting failures only after they occur can increase downtime and recovery cost.

This project develops an AI-based failure management pipeline that:

1. Processes cloud workload telemetry.
2. Predicts the probability of task/resource failure.
3. Detects anomalous workload behaviour.
4. Combines failure probability and anomaly information into a unified risk score.
5. Maps risk levels to recovery actions.
6. Provides confidence and model-based explanations for decisions.
7. Provides temporal load forecasting as a separate proactive analysis.
8. Provides a cloud-healing adapter architecture for future AWS execution.

The current ML pipeline is developed and evaluated using Borg workload traces. AWS telemetry integration and real AWS healing execution are separate deployment-stage components.

---

## System Architecture

```text
                    Borg Workload Traces
                            |
                            v
                  Data Preprocessing
                            |
                            v
                   Feature Engineering
                            |
              +-------------+-------------+
              |                           |
              v                           v
       Failure Prediction          Anomaly Detection
       Random Forest               Isolation Forest
              |                           |
              |                           v
              |                  Anomaly Calibration
              |                           |
              +-------------+-------------+
                            |
                            v
                       Risk Scoring
                            |
                            v
                    Policy / Decision
                            |
            +---------------+----------------+
            |               |                |
            v               v                v
        Normal          Scale/Migrate    Restart Task
                            |
                            v
                    Healing Adapter
                            |
                            v
                  Cloud Environment
                  (AWS integration)
```
Risk Assessment

The system combines two signals:

Failure probability from the Random Forest classifier.
Calibrated anomaly signal from the Isolation Forest.

The resulting risk score is bounded between 0 and 1.

The current policy uses the following risk bands:

Risk Score	Decision
< 0.30	Normal
0.30 – < 0.50	Scale Resources
0.50 – < 0.65	Migrate VM
>= 0.65	Restart Task

These thresholds are policy decisions and are not modified simply to obtain desired demonstration outputs.

Machine Learning Models
Failure Prediction

A Random Forest classifier is used to estimate the probability of failure from engineered workload features.

Current evaluation results:

Metric	Result
Accuracy	0.997
Precision	0.9947
Recall	0.9921
F1 Score	0.9934
ROC-AUC	0.9999
PR-AUC	0.9998
Brier Score	0.00276
Anomaly Detection

Isolation Forest is used to identify workload behaviour that differs from normal workload patterns.

The raw anomaly output is calibrated into a [0, 1] signal before being combined with the failure probability.

Explainability / RCA

The system provides a model-based explanation for individual decisions.

The explanation identifies features that are most influential to the model's prediction and reports:

Feature value
Relative contribution/sensitivity
Direction of influence
Raw anomaly information
Calibrated anomaly signal
Failure probability
Final risk decision

The explanation is model-based and should not be interpreted as causal proof of the actual infrastructure root cause.

The detailed explanation path is kept separate from the normal low-latency prediction path.

Load Forecasting

A temporal load forecasting module is included for proactive analysis.

It uses ordered workload timestamps to forecast future:

CPU load
Memory load

The forecasting evaluation uses a chronological holdout rather than a random split.

A persistence baseline is also evaluated to provide a meaningful comparison.

Forecasting is intentionally independent of the risk and recovery decision pipeline. Forecast output does not directly modify the current risk score or recovery action.

Performance

The measured prediction + anomaly + risk + decision path has the following latency:

Metric	Latency
Mean	60.89 ms
P50	60.68 ms
P95	62.30 ms
P99	62.53 ms

P50, P95 and P99 represent the latency below which approximately 50%, 95%, and 99% of requests complete respectively.

The detailed model-based explanation/RCA analysis is substantially slower and is therefore treated as a separate analysis path.

Dataset

The project uses Borg workload traces for model development and evaluation.

The raw dataset is intentionally excluded from Git tracking because of its size and distribution considerations.

Expected raw dataset location:

data/raw/borg_traces_data.csv

Generated/processed analysis files may be stored under:

data/processed/
Project Structure
ai-cloud-failure-self-healing/
│
├── app.py
├── main.py
├── config.py
│
├── src/
│   ├── anomaly_detection.py
│   ├── boundary_analysis.py
│   ├── confidence.py
│   ├── decision_engine.py
│   ├── diagnosis.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── forecasting.py
│   ├── healing.py
│   ├── incident_memory.py
│   ├── learning.py
│   ├── outcome.py
│   ├── pipeline.py
│   ├── policy.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── profile_analysis.py
│   ├── profile_prediction.py
│   ├── risk_score.py
│   ├── single_prediction.py
│   ├── train_model.py
│   └── utils.py
│
├── models/
│   ├── failure_model.pkl
│   ├── random_forest.pkl
│   ├── anomaly_model.pkl
│   ├── anomaly_calibrator.pkl
│   ├── feature_medians.pkl
│   └── scaler.pkl
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── eda.ipynb
│   └── feature_testing.ipynb
│
├── results/
│   ├── figures/
│   ├── forecasting/
│   ├── latency/
│   └── metrics/
│
├── tests/
│
├── DECISIONS.md
├── EXECUTION_FLOW.md
├── README_PROJECT_STATUS.md
└── .gitignore
Installation
1. Clone the repository
git clone https://github.com/adityasarin12/ai-cloud-failure-self-healing.git
cd ai-cloud-failure-self-healing
2. Create a virtual environment

Windows:

python -m venv venv
venv\Scripts\activate

Linux/macOS:

python3 -m venv venv
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt

If requirements.txt is not present in the repository, install the project dependencies according to the development environment used for the project.

Running the Application

Start the Streamlit application using:

streamlit run app.py

The application provides functionality for:

Single workload prediction
Risk assessment
Recovery decision
Model-based explanation
Performance information
Load forecasting
Workload profile analysis
Running Tests

Run the test suite using:

pytest

The test suite covers major components including:

Risk scoring
Failure prediction
Anomaly calibration
Decision engine
Profile prediction
Boundary analysis
Integration
Scientific extensions
UI behaviour
Results and Figures

Generated evaluation results are available under:

results/

Important outputs include:

results/metrics/
results/figures/
results/latency/
results/forecasting/

These include classifier metrics, confusion matrices, ROC/PR curves, calibration curves, risk distributions, action distributions, latency measurements, and forecasting results.

AWS / Cloud Healing Integration

The project contains a cloud-healing adapter architecture designed to connect the decision engine with a real cloud environment.

Current status:

AI/ML prediction pipeline: Implemented
Anomaly detection: Implemented
Risk assessment: Implemented
Decision engine: Implemented
Model-based explanation: Implemented
Load forecasting: Implemented
Cloud healing adapter architecture: Implemented
Real AWS telemetry integration: Pending
Real AWS healing execution: Pending
Before/after AWS recovery validation: Pending

The AWS component should be evaluated using real telemetry and real recovery outcomes before claiming complete automated self-healing in a production cloud environment.

Research and Engineering Decisions

Important engineering and research decisions are documented in:

DECISIONS.md

The execution flow and system boundaries are documented in:

EXECUTION_FLOW.md

Current project status is documented in:

README_PROJECT_STATUS.md
Scientific Boundaries

The project follows several important evaluation principles:

Training and runtime environments are kept conceptually separate.
Raw model metrics are reported without post-hoc manipulation.
Real workload profiles are used for realistic prediction analysis.
Synthetic/manual telemetry is treated separately from real workload profiles.
Model-based explanations are not presented as causal root-cause proof.
Forecasting is evaluated independently from the risk decision.
A persistence baseline is included for forecasting comparison.
Normal prediction latency is measured separately from detailed explanation latency.
AWS self-healing claims require real execution and before/after validation.
Current Project Status
Completed
Borg data preprocessing
Feature engineering
Failure prediction
Anomaly detection
Anomaly calibration
Risk scoring
Policy-based decision engine
Confidence layer
Real profile-based prediction UI
Manual telemetry analysis
Model-based explanation
Load forecasting
Performance benchmarking
Evaluation metrics
Research documentation
Automated tests
Streamlit interface
In Progress / Pending
AWS telemetry integration
Real cloud healing execution
Recovery outcome collection
Before/after recovery validation
Final AWS experimental results
Final paper figures based on the complete experimental system
