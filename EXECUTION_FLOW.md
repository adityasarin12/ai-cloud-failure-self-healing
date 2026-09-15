# System Execution Flow

This document explains exactly how the AI-Based Cloud Failure Prediction, Risk Assessment & Automated Healing system operates at runtime. It is designed to be understandable to team members, supervisors, future developers, and paper reviewers.

---

## 1. Input Sources

The system accepts telemetry data through two modes:

### A. Realistic Training Profile (Primary)

**Status**: ✓ IMPLEMENTED

Uses actual preprocessed task records from the Google Borg training dataset:
- Source: `data/raw/borg_traces_data.csv`
- Preprocessing: `src/preprocessing.py` / `preprocess_data()`
- Reference set: 5,000 real task profiles with measured CPU, memory, priority, and other attributes
- Feature preservation: All 26 model features intact; no synthetic or filled values
- User interaction: Select a profile from the dropdown; optionally edit individual telemetry fields (direct task observables only)

**Example workflow**:
```
1. User selects "Task ID 12345" from profile list
2. Streamlit loads the complete 26-feature row for that task
3. User may adjust specific fields: cpu_mean, assigned_memory, priority, etc.
4. System validates edits are within EDITABLE_TELEMETRY_FEATURES
5. Features passed to model in original trained feature order
```

### B. Manual Telemetry (Secondary, Exploratory)

**Status**: ✓ IMPLEMENTED

Users can manually enter telemetry values for interactive testing:
- Input fields: priority, cpu_mean, assigned_memory, cycles_per_instruction, memory_accesses_per_instruction, sample_rate, req_cpu, req_memory, avg_cpu, avg_memory, max_cpu, max_memory, sample_cpu, tail_cpu_mean, page_cache_memory
- Bounds: P01–P99 quantiles from reference profiles
- Limitations: Cannot naturally reach "Normal" action (minimum risk ≈ 0.424 due to input-space characteristics)
- Documentation: Clearly labeled as exploratory; results shown separately from profile mode

**Important**: Both input modes use the **same trained models** and **same decision logic**. The difference is purely in how feature vectors are constructed.

### C. AWS Real-Time Input (Future)

**Status**: ⏳ PENDING (AWS teammate responsibility)

Intended architecture:
```
AWS Cloud Telemetry (runtime)
         ↓
Collector Service
         ↓
Preprocessor (align to model features)
         ↓
Prediction Service (this AI system)
         ↓
Decision + Action Execution
```

**Not yet active**: Awaiting AWS teammate to implement telemetry collection and integrate with this service.

---

## 2. Feature Construction

Once telemetry input is received (profile-based or manual), it must be aligned to the trained model's feature order.

```
Input Profile / Telemetry
         ↓
Feature Extraction
         ↓
Validation (all 26 model features present)
         ↓
Feature Re-alignment
         ↓
Model Feature Order
(matching model.feature_names_in_)
         ↓
Feature Vector [26 dimensions]
```

### Implementation Details

**Module**: `src/profile_prediction.py` / `build_profile_prediction_features()`

```python
# Ensures the feature row matches the model's learned feature order exactly
input_data = pd.DataFrame([raw_profile])
for feature in model_features:
    if feature not in input_data.columns:
        raise ValueError(f"Missing feature: {feature}")

prediction_row = input_data[list(model_features)]  # Re-align to model order
```

**Critical**: The feature order is determined by `model.feature_names_in_` (set at training time). If order is wrong, predictions are meaningless.

**Scaling**: The active prediction path **does NOT apply additional scaling** because Random Forest and Isolation Forest are scale-invariant. Features are used as-is from the model's training distribution.

---

## 3. Failure Prediction

The Random Forest classifier predicts the probability of task failure.

```
Feature Vector [26 dimensions]
         ↓
Random Forest Model
         ↓
Decision Tree Ensemble
         ↓
Class Probabilities [P(class=0), P(class=1)]
         ↓
failure_probability = P(class=1)  ∈ [0, 1]
```

### Implementation Details

**Module**: `src/predict.py` / `predict_failure()`

- **Model artifact**: `models/failure_predictor.pkl`
- **Inference**: `model.predict_proba(feature_row)[:, 1]` extracts class-1 (failure) probability
- **Latency**: ~46.71 ms (measured mean; P99: 49.44 ms)
- **Output**: Single float in [0, 1]

### Borg Training Evaluation (offline)

| Metric | Value |
|--------|-------|
| Accuracy | 0.997 |
| Precision | 0.9947 |
| Recall | 0.9921 |
| F1 Score | 0.9934 |
| ROC-AUC | 0.9999 |
| PR-AUC | 0.9998 |
| Brier Score | 0.0028 |

**Important**: These metrics are from **Borg evaluation set**, not real AWS production. Online validation required.

---

## 4. Anomaly Detection

The Isolation Forest detects anomalous patterns in the feature space, independent of failure labels.

```
Feature Vector [26 dimensions]
         ↓
Isolation Forest Model
         ↓
Sparse Decision Trees
(path length to leaf)
         ↓
Raw Anomaly Score (unbounded; can be negative)
         ↓
Anomaly Calibrator
         ↓
Calibrated Anomaly ∈ [0, 1]
```

### Implementation Details

**Module**: `src/anomaly_detection.py` / `detect_anomaly()`

- **Model artifact**: `models/anomaly_detector.pkl`
- **Inference**: `model.decision_function(feature_row)` produces raw score
- **Latency**: ~11.18 ms (measured mean; P99: 12.52 ms)
- **Raw score range**: Unbounded; can be negative, zero, or positive
  - Negative score → anomalous (deeper in tree; more isolated)
  - Positive score → normal (shallow in tree; less isolated)

### Calibration Step

**Module**: `src/risk_score.py` / `_normalize_anomaly_score()`

Raw IF scores are not suitable for direct use in risk calculation because:
1. Unbounded (can be <-1 or >1)
2. Not comparable to [0,1] failure probability without transformation
3. Direction not intuitive (negative = anomalous, but negative ≠ probability)

**Calibration pipeline**:
```
Raw Anomaly Score: s ∈ (-∞, +∞)
         ↓
Lookup in Reference Distribution
(where was raw score in training distribution?)
         ↓
Percentile Rank: r ∈ [0, 1]
         ↓
Flip (anomalous scores are lower percentile):
anomaly_signal = 1.0 - r
         ↓
Clip: max(0.0, min(1.0, anomaly_signal))
         ↓
Calibrated Anomaly ∈ [0, 1]
```

**Calibrator artifact**: `models/anomaly_calibrator.pkl` (fitted on training set decision scores)

**Latency**: ~0.18 ms (measured mean; negligible)

### Output Interpretation

- **Calibrated anomaly = 0.0**: Input is deep in normal distribution (typical example)
- **Calibrated anomaly = 0.5**: Input is at median of training distribution (borderline)
- **Calibrated anomaly = 1.0**: Input is extremely anomalous (tail of training distribution)

The calibrated value is the **user-facing anomaly score**. The raw IF score is available in detailed explanation output but should not be presented to users.

---

## 5. Risk Calculation

Risk combines failure prediction and anomaly signals into a single [0, 1] score.

```
Failure Probability: f ∈ [0, 1]
         ↓
Calibrated Anomaly: a ∈ [0, 1]
         ↓
Risk Formula:
   risk = 0.7 * f + 0.3 * a
         ↓
Clip to [0, 1]:
   risk = min(max(risk, 0.0), 1.0)
         ↓
Risk Score ∈ [0, 1]
```

### Implementation Details

**Module**: `src/risk_score.py` / `calculate_risk()`

```python
def calculate_risk(failure_prob, anomaly_score, 
                   failure_weight=0.7, anomaly_weight=0.3):
    norm_anom = _normalize_anomaly_score(anomaly_score)
    risk = (failure_weight * failure_prob + 
            anomaly_weight * norm_anom)
    risk = np.clip(risk, 0.0, 1.0)
    return float(risk)
```

**Weights**:
- **0.7 on failure probability**: Supervised model has higher confidence (ROC-AUC 0.9999 on Borg)
- **0.3 on anomaly**: Complementary unsupervised signal; prevents over-reliance on one model

**Latency**: ~0.18 ms (measured mean; negligible)

**Bounded [0, 1]**: Risk always fits within policy bands (see section 6)

---

## 6. Risk Policy

Risk determines the base action through fixed thresholds.

```
Risk ∈ [0, 1]
         ↓
Apply Policy Bands
         ↓
    ┌─ risk < 0.30
    │    ↓
    │   "Normal"
    │
    ├─ 0.30 ≤ risk < 0.50
    │    ↓
    │   "Scale Resources"
    │
    ├─ 0.50 ≤ risk < 0.65
    │    ↓
    │   "Migrate VM"
    │
    └─ risk ≥ 0.65
         ↓
        "Restart Task"
```

### Implementation Details

**Module**: `src/policy.py` / `get_base_action(risk)`

```python
def get_base_action(risk):
    risk = float(risk)
    if risk < 0.30:
        return "Normal"
    if risk < 0.50:
        return "Scale Resources"
    if risk < 0.65:
        return "Migrate VM"
    return "Restart Task"
```

**Important**:
- Thresholds are **hardcoded and intentionally not configurable**
- Not changed to force action distribution
- Based on model confidence and action severity hierarchy

### Real-Profile Distribution Analysis

Evaluation on 5,000 real Borg profiles:

| Action | Count | Percentage |
|--------|-------|-----------|
| Normal | 3,824 | 76.48% |
| Scale Resources | 32 | 0.64% |
| Migrate VM | 6 | 0.12% |
| Restart Task | 1,138 | 22.76% |

**Key finding**: All four actions are naturally reachable when using real profiles. Action distribution reflects genuine risk distribution in the data.

---

## 7. Decision Engine

The decision engine transforms base action into a structured decision with policy validation and confidence assessment.

```
Risk Score
         ↓
Base Action (from policy)
         ↓
Policy Validation
(is this action allowed at this risk level?)
         ↓
Confidence Assessment
(do the failure & anomaly models agree?)
         ↓
Action Ranking
(compare alternatives within policy constraints)
         ↓
Execution Mode
(autonomous vs. supervised vs. alert-only)
         ↓
Structured Decision
{
  incident_id,
  recommended_action,
  risk_score,
  confidence,
  execution_mode,
  policy_status,
  alternatives,
  explanation
}
```

### Implementation Details

**Module**: `src/decision_engine.py` / `make_decision()`

**Policy check**: `src/policy.py` / `check_policy(action, risk, confidence_level, max_blast_radius)`

```python
def check_policy(action, risk, confidence_level, max_blast_radius):
    # Returns ActionPolicy(allowed, approval_required, blast_radius, rollback_required)
    # Validates:
    # 1. Action matches risk band
    # 2. Blast radius ≤ configured limit
    # 3. Confidence supports execution mode
```

**Confidence assessment**: `src/confidence.py` / `assess_confidence()`

```python
def assess_confidence(failure_probability, anomaly_score):
    # Measures agreement between RF and IF models
    anomaly_signal = _normalize_anomaly_score(anomaly_score)
    agreement = 1.0 - abs(failure_probability - anomaly_signal)
    
    if agreement >= 0.80:
        return HIGH confidence ("autonomous")
    if agreement >= 0.55:
        return MEDIUM confidence ("supervised")
    else:
        return LOW confidence ("alert_only")
```

**Critical**: Confidence does **NOT** modify risk or action selection. It determines execution autonomy, not decision content.

### Execution Modes

| Confidence | Mode | Meaning |
|-----------|------|---------|
| HIGH | autonomous | Execute action immediately (dry-run or real) |
| MEDIUM | supervised | Action selected; requires human approval before execution |
| LOW | alert_only | Action suggested; user decides whether to proceed |

### Structured Decision Output

```json
{
  "incident": "task-12345",
  "recommended_action": "Restart Task",
  "risk_score": 0.72,
  "risk_band": ">=0.65",
  "confidence": 0.85,
  "confidence_level": "HIGH",
  "execution_mode": "autonomous",
  "policy_status": "approved",
  "alternatives": [
    {
      "action": "Restart Task",
      "score": 0.42,
      "policy_status": "approved"
    }
  ],
  "explanation": "Failure probability 0.65 and calibrated anomaly 0.82 produced risk 0.72, which falls in the >=0.65 risk band. Selected action: Restart Task; the policy maps this risk band to that action. ..."
}
```

---

## 8. RCA / Reason Explanation

Model-based explanation identifies features associated with the predicted risk.

```
Trained Models
(RF + IF)
         ↓
Feature Replacement
(zero each feature one at a time)
         ↓
Sensitivity Analysis
(measure prediction change for each feature)
         ↓
Rank by Impact
         ↓
Top-K Contributing Factors
{
  feature_name,
  feature_value,
  impact,
  direction (increases/decreases risk)
}
```

### Implementation Details

**Module**: `src/diagnosis.py` / `explain_prediction()`

**Algorithm**:
1. Get current prediction: f(X)
2. For each feature i in 1..26:
   - Create modified row: X' = X with feature i set to 0
   - Get modified prediction: f(X')
   - Calculate impact: Δ = f(X) - f(X')
3. Sort by |Δ| (absolute impact)
4. Return top-K factors

**Latency**: ~0.89 seconds for single row (sample measured: 88,983 ms)

This is **NOT suitable for real-time path**. Must be asynchronous or optional.

### Output Structure

```json
{
  "primary_reason": "cpu_mean shows the strongest model-associated effect and increases_risk.",
  "contributing_factors": [
    {
      "feature": "cpu_mean",
      "value": 0.85,
      "impact": 0.12,
      "direction": "increases_risk",
      "failure_impact": 0.10,
      "anomaly_impact": 0.02
    },
    {
      "feature": "assigned_memory",
      "value": 0.72,
      "impact": 0.08,
      "direction": "increases_risk",
      "failure_impact": 0.06,
      "anomaly_impact": 0.02
    }
  ],
  "risk_score": 0.72,
  "recommended_action": "Restart Task",
  "disclaimer": "Model-based explanation — not causal proof."
}
```

### Important Scientific Boundary

**RCA output shows predictive association, not causal root cause.**

- ✓ Correct: "CPU-related telemetry is associated with increased risk"
- ✓ Correct: "The model uses CPU features to predict failure"
- ✗ Incorrect: "High CPU caused the failure"
- ✗ Incorrect: "CPU is the root cause"

Every RCA output includes the disclaimer: **"Model-based explanation — not causal proof."**

---

## 9. Load Forecasting (Independent Path)

Load forecasting predicts CPU and memory usage independent of failure prediction or risk calculation.

```
Ordered Borg Load Time Series
         ↓
Temporal Preprocessing
(extract cpu_load, memory_load by start_time)
         ↓
Train/Test Split
(chronological; no future leakage)
         ↓
Linear Regression
(fit on training window)
         ↓
Forecast Next-N Values
         ↓
Compare vs. Persistence Baseline
         ↓
Report Metrics (MAE, RMSE, MAPE)
```

### Implementation Details

**Module**: `src/forecasting.py`

- **Model**: `LinearRegression()` fitted on ordered time series
- **Evaluation**: Chronological train/test split (2,389 training, 598 test timestamps)
- **Baseline**: Persistence (last value = next value)
- **Latency**: Negligible (<1 ms per forecast)

### Results

| Resource | Linear MAE | Persistence MAE | Winner |
|----------|-----------|-----------------|--------|
| CPU | 0.01031 | Better | Persistence |
| Memory | 0.00558 | Better | Persistence |

**Important finding**: Persistence baseline **currently performs better** than linear regression on MAE for both resources.

**Paper statement**: "A short-horizon forecasting component was implemented and evaluated using a chronological holdout. However, the persistence baseline achieved lower MAE for both CPU and memory, indicating that the current forecasting component requires further improvement."

### Decoupling from Risk / Decision

**CRITICAL**: Load forecasting does NOT feed into:
- Failure probability
- Anomaly detection
- Risk calculation
- Action selection

It is a **separate analytical module** shown in the Streamlit UI but not connected to the decision pipeline. This allows:
- Independent evaluation and improvement
- Separate paper section or future work
- Future enhancement: when forecasting beats baseline, integrate into proactive actions

---

## 10. Output: Streamlit UI

The Streamlit interface displays predictions and recommendations to the user.

```
Streamlit Session
         ↓
Input Selection
(Profile or Manual Telemetry tab)
         ↓
Preprocessing & Feature Alignment
         ↓
Inference
(RF + IF + Calibration + Risk + Decision + RCA)
         ↓
Display Results
{
  Failure Probability,
  Anomaly Score,
  Risk Score,
  Recommended Action,
  Confidence,
  Why This Decision? (RCA),
  Performance / Latency,
  Load Forecast,
  Input Distribution Diagnostics
}
```

### UI Sections

1. **Input Selection**
   - Realistic Training Profile (primary)
   - Manual Telemetry (secondary)

2. **Results Display**
   - Failure Probability: [0, 1] gauge
   - Anomaly Score: [0, 1] calibrated signal (NOT raw IF score)
   - Risk Score: [0, 1] with band label
   - Recommended Action: string + policy status

3. **Explanation**
   - Confidence level + score
   - "Why this decision?" → RCA top-K factors (async-loaded or deferred)
   - Decision rationale from policy

4. **Performance**
   - Latency breakdown: RF, IF, Calibration, Risk, Decision, RCA
   - Mean / P50 / P95 / P99 for main components
   - Note on RCA latency (separate from real-time path)

5. **Load Forecast** (collapsed by default)
   - Current CPU / Memory
   - Forecasted next-3 values
   - Comparison vs. persistence

6. **Input Distribution Diagnostics** (collapsed by default)
   - Feature | Current | P01 | Median | P99 | Status
   - Status: IN RANGE / LOW OOD / HIGH OOD

---

## 11. AWS Execution Path (Pending)

The intended production architecture connects AI decision to AWS infrastructure changes.

```
Structured Decision
(from src/decision_engine.py)
         ↓
CloudHealingAdapter Interface
         ↓
         ├─ DryRunHealingAdapter (current)
         │  └─ Logs intent; returns {"status": "DRY_RUN", ...}
         │
         └─ Real AWS Adapter (pending AWS teammate)
            ├─ Authenticate to AWS
            ├─ Fetch before-action metrics (CPU, memory, error rate, latency)
            ├─ Execute action (scale ASG, migrate task, restart container)
            ├─ Monitor during execution
            ├─ Fetch after-action metrics
            ├─ Calculate recovery_time = (time to return to normal) or (predefined SLA)
            ├─ Determine success/failure
            ├─ Store outcome
            │
            └─ Return HealingOutcome
               {
                 action,
                 incident_id,
                 before_metrics { cpu, memory, error_rate, ... },
                 after_metrics { ... },
                 recovery_time,
                 success: bool,
                 cost_impact
               }
```

### Implementation Status

**Current** (AI side, implemented):
- `src/healing.py` / `CloudHealingAdapter` abstract interface
- `src/healing.py` / `DryRunHealingAdapter` mock implementation
- Structured decision export ready

**Pending** (AWS teammate):
- Real AWS authentication and service integration
- Before/after metrics collection
- Action execution (ASG scaling, task restart, VM migration)
- Outcome storage
- Success/failure determination logic

### Design Principles

1. **Clear separation**: AI decides; AWS executes
2. **Dry-run first**: Always validate with dry-run before real execution
3. **Outcome feedback**: Real results → offline retraining (future enhancement)
4. **Safety**: Policy checks (approval_required, rollback_required) enforced by adapter

---

## 12. Latency Path

System latency is measured at multiple stages to understand performance characteristics.

### Real-Time Decision Path (~60 ms)

```
Feature Vector Input
    ↓ (negligible: 0.0 ms)
Random Forest Prediction
    ↓ (46.71 ms mean)
Isolation Forest Anomaly
    ↓ (11.18 ms mean)
Anomaly Calibration
    ↓ (0.18 ms mean)
Risk Calculation
    ↓ (0.18 ms mean)
Decision Engine
    ↓ (0.15 ms mean)
Total: 60.89 ms mean
       62.53 ms P99
```

### Detailed Latency Measurements

| Component | Mean | P50 | P95 | P99 | Min | Max |
|-----------|------|-----|-----|-----|-----|-----|
| Preprocessing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Random Forest | 46.71 | 48.12 | 49.23 | 49.44 | 39.01 | 49.49 |
| Isolation Forest | 11.18 | 12.09 | 12.45 | 12.52 | 6.83 | 12.54 |
| Calibration | 0.18 | 0.19 | 0.21 | 0.22 | 0.07 | 0.22 |
| Risk | 0.18 | 0.19 | 0.22 | 0.23 | 0.08 | 0.23 |
| Decision | 0.15 | 0.17 | 0.19 | 0.19 | 0.02 | 0.20 |
| **Real-time Total** | **60.89** | **60.68** | **62.30** | **62.53** | **59.91** | **62.59** |

### Optional / Asynchronous RCA Path

```
Structured Decision
    ↓ (decision returned immediately)
Detailed RCA (asynchronous or deferred)
    ↓ (88,983 ms sample measured)
Feature Sensitivity Analysis
    ↓
Top-K Factors + Explanation
```

**Important**: RCA latency **must NOT be mixed** with real-time decision latency. Separate numbers:
- **Decision latency**: ~60 ms (suitable for real-time)
- **RCA latency**: ~89 s (not real-time; must be async)

### Optimization Opportunities

1. **RCA optimization**: Cache feature importances; batch explanations; incremental computation
2. **Model optimization**: Compile RF/IF with ONNX; GPU acceleration
3. **Parallelization**: Run RF and IF in parallel (currently sequential)

---

## 13. Complete End-to-End Execution Diagram

```mermaid
graph TD
    A["Input Source"] -->|Profile| B["Feature Construction"]
    A -->|Manual Telemetry| B
    A -->|AWS Telemetry| B
    
    B -->|26 Features| C["Feature Alignment<br/>to Model Order"]
    
    C -->|Feature Vector| D["Random Forest<br/>Failure Prediction"]
    C -->|Feature Vector| E["Isolation Forest<br/>Anomaly Detection"]
    
    D -->|failure_prob ∈ [0,1]| F["Anomaly Calibration<br/>Raw Score → [0,1]"]
    E -->|raw_anomaly_score| F
    
    F -->|calibrated_anomaly ∈ [0,1]| G["Risk Calculation<br/>0.7*fp + 0.3*ca"]
    D -->|failure_probability| G
    
    G -->|risk ∈ [0,1]| H["Policy Mapping<br/>Risk → Action Band"]
    
    H -->|base_action| I["Decision Engine<br/>Validate + Confidence"]
    D -->|fp| I
    F -->|ca| I
    
    I -->|structured_decision| J["Streamlit Output"]
    I -->|decision| K["Async RCA<br/>Feature Sensitivity"]
    
    K -->|explanation| J
    
    J -->|user_sees| L["UI Display<br/>Action + Reason + Confidence"]
    
    I -->|structured_decision| M["CloudHealingAdapter<br/>Dry-Run or AWS"]
    M -->|dry_run| N["Log Intent"]
    M -->|real_aws| O["Execute Action<br/>Before Metrics"]
    
    O -->|execute| P["Healing Action<br/>Scale/Migrate/Restart"]
    P -->|after_metrics| Q["Outcome Validation<br/>Success/Failure"]
    Q -->|outcome| R["Store Outcome"]
    R -->|feedback| S["Future Retraining<br/>src/learning.py"]
    
    C -->|26 Features| T["Load Forecasting<br/>INDEPENDENT"]
    T -->|forecast_cpu_memory| J
    
    L -->|metrics| U["Performance Logging"]
    U -->|latency_benchmark| V["results/latency/"]
    
    style D fill:#90EE90
    style E fill:#90EE90
    style G fill:#90EE90
    style H fill:#90EE90
    style I fill:#90EE90
    style T fill:#FFB6C1
    style M fill:#FFD700
    style O fill:#FFD700
    style S fill:#FFA07A
```

**Legend**:
- 🟢 Green: IMPLEMENTED core decision path
- 🔴 Pink: IMPLEMENTED but independent (forecasting)
- 🟡 Yellow: Adapter + AWS (pending real implementation)
- 🟠 Orange: Future enhancement (retraining)

---

## 14. Current vs. Future State

| Component | Current Status | Measured/Validated | Future Work |
|-----------|----------------|-------------------|-------------|
| **ML Models** |
| Random Forest | ✓ Trained | 0.9999 ROC-AUC on Borg | Retrain on real AWS outcomes |
| Isolation Forest | ✓ Trained | ~11 ms inference | N/A (unsupervised) |
| **Calibration** | ✓ Empirical CDF | Validation test suite | Recalibrate on new Borg/AWS data |
| **Risk Scoring** | ✓ 0.7/0.3 weighted | 5,000 profile evaluation | N/A (formula fixed) |
| **Decision Engine** | ✓ Policy + confidence | Dry-run validation | Real AWS policy validation |
| **RCA / Explanation** | ✓ Feature sensitivity | ~89 s latency measured | Optimize for <10 s async |
| **Forecasting** | ✓ Linear regression | Baseline comparison | Improve to beat persistence |
| **Latency** | ✓ Real-time measured | 60.89 ms mean | Parallelize RF/IF |
| **Streamlit UI** | ✓ Full feature-set | Manual + profile modes | Async RCA loading |
| **CloudHealingAdapter** | ✓ Dry-run mock | Intent logging | AWS implementation |
| **Real AWS Telemetry** | ⏳ Pending | Not yet available | AWS teammate to provide |
| **Real AWS Healing** | ⏳ Pending | Not yet executed | AWS teammate implementation |
| **Outcome Feedback** | ⏳ Schema ready | Dry-run outcomes only | Real AWS before/after metrics |
| **Retraining Loop** | ⏳ Ready to consume | Placeholder logic | Activate with real outcomes |

---

## 15. Important Scientific Boundaries

### Do NOT claim these (yet)

1. **High offline accuracy = production readiness**
   - Random Forest achieves 0.9999 ROC-AUC on Borg evaluation
   - Borg ≠ AWS; offline metrics ≠ production performance
   - Real AWS outcomes required for production claims

2. **Borg training data represents AWS workloads**
   - Borg is Google's historical cluster workloads
   - AWS production is different infrastructure, different workloads
   - Real AWS validation required; cannot assume transfer

3. **Model-based RCA is causal root-cause analysis**
   - RCA identifies predictive associations, not proven causes
   - Every explanation includes disclaimer: "not causal proof"
   - Avoid causal language in all outputs

4. **Forecasting outperforms baseline**
   - Persistence baseline currently wins on MAE for CPU and memory
   - Forecasting requires improvement before claiming superiority
   - Honest statement: "requires further enhancement"

5. **Detailed RCA is real-time**
   - Real-time path: ~60 ms (RF + IF + Risk + Decision)
   - Detailed RCA: ~89 s (feature sensitivity)
   - Must be kept separate; RCA must be async/optional

6. **System prevents failures**
   - Real AWS outcome data required to validate healing effectiveness
   - Dry-run does not prove real remediation works
   - No production effectiveness claims until real outcomes collected

7. **Manual telemetry mode is representative**
   - Manual input cannot reach "Normal" action (min risk ≈ 0.424)
   - This is input-space limitation, not system defect
   - Real profiles solve this by spanning the risk space

8. **Policy thresholds are optimal**
   - Thresholds are fixed and justified by model confidence hierarchy
   - No claim of optimality; designed for safety and gradualism
   - Subject to future refinement based on real outcomes

### What CAN be claimed

✓ Model achieves high offline accuracy on Borg evaluation data  
✓ Real-profile predictions naturally reach all four actions  
✓ Real-time decision path meets sub-100 ms latency budget  
✓ System provides interpretable model-based explanations  
✓ Anomaly detection is independent complementary signal  
✓ Policy ensures graduated response to increasing risk  
✓ Dry-run mode enables safe testing without infrastructure changes  
✓ Forecasting identifies temporal patterns (even if baseline beats model)  

---

## 16. Deployment Readiness Checklist

### Phase 1: ML Validation (✓ COMPLETE)
- [x] Model training on Borg data
- [x] Offline evaluation (high accuracy)
- [x] Feature engineering validated
- [x] Anomaly calibration implemented
- [x] Risk formula tested
- [x] Policy thresholds fixed and documented

### Phase 2: System Integration (✓ COMPLETE)
- [x] Prediction pipeline implemented
- [x] Anomaly detection integrated
- [x] Risk calculation operational
- [x] Decision engine functional
- [x] RCA/explanation module ready
- [x] Streamlit UI complete
- [x] Latency measured and acceptable
- [x] Test suite (90 tests, 0 failures)

### Phase 3: AWS Integration (⏳ IN PROGRESS)
- [ ] AWS teammate implements telemetry collection
- [ ] CloudHealingAdapter implemented for real AWS
- [ ] Dry-run mode tested with actual AWS SDK
- [ ] Before/after metrics collection validated
- [ ] Outcome storage operational
- [ ] Policy approval workflows configured

### Phase 4: Production Validation (⏳ PENDING)
- [ ] Real AWS incidents collected (N > 30 recommended)
- [ ] Action effectiveness measured (success rate, recovery time)
- [ ] No adverse effects observed
- [ ] Safety guardrails validated
- [ ] Monitoring/alerting configured
- [ ] Rollback procedure tested

### Phase 5: Continuous Improvement (⏳ FUTURE)
- [ ] Real outcomes collected and stored
- [ ] Retraining pipeline activated
- [ ] Model performance tracked over time
- [ ] Threshold adjustments based on production data
- [ ] Forecasting improved to beat baseline
- [ ] RCA latency optimized for real-time

---

## Appendix: Key Files & Modules

| Component | File(s) | Function |
|-----------|---------|----------|
| **Input** | `src/profile_prediction.py` | Load profiles, build feature vectors |
| | `src/preprocessing.py` | Prepare raw Borg data |
| **ML Models** | `models/failure_predictor.pkl` | Trained Random Forest |
| | `models/anomaly_detector.pkl` | Trained Isolation Forest |
| | `models/anomaly_calibrator.pkl` | Quantile calibrator |
| **Prediction** | `src/predict.py` | RF inference |
| | `src/anomaly_detection.py` | IF inference |
| **Risk** | `src/risk_score.py` | Risk calculation + calibration |
| | `src/policy.py` | Policy bands + action mapping |
| **Decision** | `src/decision_engine.py` | Structured decision output |
| | `src/confidence.py` | Confidence assessment |
| **Explanation** | `src/diagnosis.py` | Feature sensitivity analysis + RCA |
| **Forecasting** | `src/forecasting.py` | Load forecasting (independent) |
| **Healing** | `src/healing.py` | CloudHealingAdapter interface |
| | `src/outcome.py` | Outcome schema |
| **Testing** | `tests/` | 90 unit & integration tests |
| **UI** | `app.py` | Streamlit entry point |
| **Results** | `results/metrics/classifier_metrics.json` | Offline evaluation metrics |
| | `results/latency/latency_summary.csv` | Latency breakdown |
| | `results/forecasting/forecast_metrics.json` | Forecast baseline comparison |

---

**Document Version**: 1.0  
**Last Updated**: 2026-08-27  
**Audience**: Team, supervisor, paper reviewers, future developers  
**Status**: Living document reflecting current implementation state
