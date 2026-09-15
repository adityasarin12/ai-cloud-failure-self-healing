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