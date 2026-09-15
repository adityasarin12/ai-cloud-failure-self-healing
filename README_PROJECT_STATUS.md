# Final Year Project --- AI-Based Cloud Failure Prediction, Risk Assessment & Automated Healing

## 1. Project Overview

This project is a smart cloud reliability and self-healing system that
takes workload/telemetry data, predicts failure risk, detects anomalous
behaviour, calculates a bounded risk score, selects a safe remediation
action, explains the decision, and is designed to connect the decision
to an AWS healing layer.

The current ML/prototype pipeline is based on real Google Borg cluster
workload data for training/evaluation. The planned deployment path is
separate: the teammate's AWS layer will provide real cloud/runtime
telemetry and execute the selected action through a cloud-healing
adapter.

### Core architecture

``` text
Cloud / Borg Telemetry
        |
        v
Preprocessing + Feature Engineering
        |
        +-----------------------+
        |                       |
        v                       v
Random Forest             Isolation Forest
Failure Prediction        Anomaly Detection
        |                       |
        |                       v
        |                Anomaly Calibration
        |                       |
        +-----------+-----------+
                    |
                    v
               Risk Score
                    |
                    v
             Decision Engine
                    |
          +---------+---------+
          |                   |
          v                   v
       RCA/Reason        Action Selection
          |                   |
          +---------+---------+
                    |
                    v
           Structured Decision
                    |
          +---------+---------+
          |                   |
          v                   v
      Streamlit UI       AWS Healing Adapter
                              |
                              v
                       Actual AWS Action
                              |
                              v
                    Before/After Outcome
```

------------------------------------------------------------------------

# 2. What Is Already Implemented

## ML / AI

-   Random Forest failure prediction
-   Isolation Forest anomaly detection
-   Empirical anomaly calibration
-   Bounded risk scoring
-   Risk-band policy
-   Confidence levels
-   Decision engine
-   Model-based explanation / RCA evidence
-   Real-profile prediction using actual preprocessed Borg records
-   Manual telemetry prediction
-   OOD/reference-distribution diagnostics
-   Batch analysis
-   Load forecasting as a separate analytical path
-   Persistence baseline comparison
-   Latency measurement

## Risk Policy

``` text
Risk < 0.30
    -> Normal

0.30 <= Risk < 0.50
    -> Scale Resources

0.50 <= Risk < 0.65
    -> Migrate VM

Risk >= 0.65
    -> Restart Task
```

These thresholds are currently fixed and should NOT be changed just to
force action diversity.

------------------------------------------------------------------------

# 3. Current Model Results

The current Random Forest evaluation produced:

  Metric             Result
  ------------- -----------
  Accuracy            0.997
  Precision       0.9947438
  Recall          0.9921363
  F1 Score        0.9934383
  ROC-AUC         0.9999409
  PR-AUC          0.9998051
  Brier Score     0.0027576

These are results from the existing saved model/evaluation pipeline. The
model was not retrained during the latest scientific-extension work.

------------------------------------------------------------------------

# 4. Real-Profile Action Validation

A deterministic evaluation was performed on **5,000 real preprocessed
Borg profiles** through the same profile builder and prediction/decision
pipeline used by Single Prediction.

### Risk distribution

  Metric         Value
  --------- ----------
  Minimum     0.003656
  P1          0.021463
  P25         0.123891
  Median      0.203087
  P75         0.295314
  P99         0.982497
  Maximum     0.996414

### Action distribution

  Action              Count   Percentage
  ----------------- ------- ------------
  Normal              3,824       76.48%
  Scale Resources        32        0.64%
  Migrate VM              6        0.12%
  Restart Task        1,138       22.76%

### Important finding

All four actions are naturally reachable using real reference profiles.

Therefore:

**Do not manipulate thresholds, model outputs, or inputs simply to make
the UI show all four actions.**

The earlier manual six-slider interface could not naturally reach the
Normal band because its generated telemetry produced a minimum risk of
approximately `0.4238`. That was an input-space limitation, not a
decision-engine defect.

The current profile-backed interface solves this by using real
training/reference profiles instead of artificial synthetic feature
combinations.

------------------------------------------------------------------------

# 5. Manual Telemetry Result

A fresh audit of 503 manual telemetry cases produced:

  Metric                              Result
  --------------------- --------------------
  Risk minimum                     0.4237725
  Risk maximum                     0.9510525
  Mean risk                        0.5373017
  Median risk                      0.5365900
  Failure probability             0.18--0.99
  Calibrated anomaly      0.860175--0.999725

Actions:

  Action              Count   Percentage
  ----------------- ------- ------------
  Normal                  0        0.00%
  Scale Resources        95     18.8867%
  Migrate VM            404     80.3181%
  Restart Task            4      0.7952%

Normal is not reachable in this manual input space because even the
lowest observed manual case has:

``` text
Failure probability = 0.18
Calibrated anomaly  = 0.992575

Risk =
0.7 * 0.18 + 0.3 * 0.992575
= 0.4237725
```

This should be documented as a limitation of the manual telemetry input
space, not "fixed" by changing the model.

------------------------------------------------------------------------

# 6. Anomaly Calibration

The original Isolation Forest output can legitimately be negative. This
is a raw model score and is NOT the final anomaly value shown to the
user.

The calibrated anomaly signal is bounded:

``` text
0.0 <= calibrated anomaly <= 1.0
```

The current pipeline is:

``` text
Raw Isolation Forest Score
        |
        v
Anomaly Calibrator
        |
        v
Calibrated Anomaly Signal [0,1]
```

The UI should display the calibrated anomaly signal as the user-facing
anomaly score, while the raw Isolation Forest score can remain available
in the explanation/details section.

------------------------------------------------------------------------

# 7. Risk Calculation

The current risk formula is:

``` text
Risk =
0.7 * Failure Probability
+
0.3 * Calibrated Anomaly
```

The result is bounded to `[0,1]`.

The risk calculation and thresholds have been validated through
automated tests and should remain unchanged during the AWS integration
phase.

------------------------------------------------------------------------

# 8. Explainability / RCA

A model-based explanation layer has now been implemented.

It uses input-specific Random Forest / Isolation Forest sensitivity to
identify important model features and their direction/impact.

Important scientific wording:

> Model-based explanation --- not causal proof.

This means the system can provide evidence for why a prediction/risk
score is high, but we should not claim that the identified feature is a
proven causal root cause.

The intended UI output is:

``` text
Why this decision?

Failure probability: Elevated
Anomaly signal: High
Risk score: 0.53
Risk band: Migrate VM

Top contributing factors:
- CPU-related telemetry
- Memory-related telemetry
- Workload/anomaly-related features

Decision rationale:
Migrate VM is selected because the resulting risk
falls within the 0.50–0.65 policy band.
```

### Important current limitation

The detailed sensitivity-based explanation is expensive.

Measured explanation-enabled execution was approximately:

``` text
88,982.93 ms
```

This is not suitable as a synchronous real-time path.

The recommendation is:

``` text
Real-time path:
Prediction -> Anomaly -> Risk -> Decision
                    ~60 ms

Optional/asynchronous path:
Prediction -> Detailed RCA
```

RCA should therefore be optimized or run asynchronously before
production deployment.

------------------------------------------------------------------------

# 9. Latency Results

The prediction + anomaly + risk benchmark currently gives:

  Metric                      Value
  -------------------- ------------
  Mean                   60.8898 ms
  P50                    60.6837 ms
  P95                    62.3038 ms
  P99                    62.5313 ms
  Minimum                59.9083 ms
  Maximum                62.5882 ms
  Standard deviation      1.0708 ms

This is the useful real-time inference path.

The detailed explanation path is much slower and must not be mixed with
the normal inference latency when reporting system responsiveness.

------------------------------------------------------------------------

# 10. Load Forecasting

Load forecasting is implemented as a **separate analytical path**.

It does NOT currently modify:

-   failure probability
-   anomaly score
-   risk score
-   action selection

Real temporal structure was found in the Borg data:

-   2,987 unique ordered timestamps
-   2,389 training timestamps
-   598 chronological holdout timestamps
-   No future leakage detected

### Current linear forecast results

  Resource         MAE      RMSE
  ---------- --------- ---------
  CPU          0.01031   0.01737
  Memory       0.00558   0.01082

### Important limitation

The persistence baseline performed better on MAE for both CPU and
memory.

Therefore, the paper must NOT claim that the current linear forecasting
model is superior.

Correct interpretation:

> A short-horizon forecasting component was implemented and evaluated
> using a chronological holdout. However, the persistence baseline
> achieved lower MAE for both CPU and memory, indicating that the
> current forecasting component requires further improvement.

The forecasting component can still be presented as an independent
proactive-analysis module.

------------------------------------------------------------------------

# 11. Streamlit UI

The Streamlit interface currently supports:

-   Realistic Training Profile selection
-   Manual Telemetry
-   Prediction
-   Failure probability
-   Anomaly score
-   Risk score
-   Recommended action
-   Confidence level
-   Decision explanation
-   Performance/latency information
-   Load Forecast section
-   Input Distribution Diagnostics

The diagnostics UI was recently cleaned.

Instead of showing many raw diagnostic lines beside every slider, it now
uses:

``` text
Input Distribution Diagnostics
```

with:

  Feature     Current   P01   Median   P99 Status
  --------- --------- ----- -------- ----- -------------------------------
  ...             ...   ...      ...   ... IN RANGE / LOW OOD / HIGH OOD

This section is collapsed by default.

------------------------------------------------------------------------

# 12. Testing Status

Latest full automated suite:

``` text
90 passed
0 failed
```

There are existing non-breaking warnings from:

-   pandas compatibility/deprecation
-   sklearn feature-name validation
-   Altair/Streamlit compatibility

These warnings do not currently cause test failures.

------------------------------------------------------------------------

# 13. What Is Still Pending

## A. AWS Integration --- MAIN TEAMMATE TASK

The AWS/cloud teammate needs to complete the real deployment/integration
boundary.

### Required work

1.  Dockerize the application/service as required.
2.  Deploy the runtime on AWS.
3.  Collect real runtime/cloud telemetry.
4.  Implement `CloudHealingAdapter`.
5.  Receive the structured decision from our AI layer.
6.  Execute the selected action safely.
7.  Capture before-action metrics.
8.  Execute action.
9.  Capture after-action metrics.
10. Calculate:
    -   recovery time
    -   success/failure
    -   latency
    -   resource impact
    -   cost if measurable
11. Store the outcome.
12. Validate the outcome through the existing outcome-validation path.
13. Start with dry-run/sandbox validation before real destructive
    actions.

### Expected interface

``` text
AI Decision
    |
    v
CloudHealingAdapter
    |
    v
AWS Action
    |
    +---- Before Metrics
    |
    +---- Execute
    |
    +---- After Metrics
    |
    v
Healing Outcome
```

The AWS layer should NOT duplicate the ML/risk logic.

It should consume the structured decision produced by the AI system.

------------------------------------------------------------------------

# 14. What I Recommend We Do Next

### Priority 1 --- Optimize RCA

Reduce the current approximately 89-second explanation cost.

Do not sacrifice the normal \~60 ms prediction path.

Possible architecture:

``` text
Prediction/Decision
       |
       +---- immediate result
       |
       +---- optional RCA calculation
```

### Priority 2 --- AWS integration

The teammate should connect the structured decision to the real cloud
environment.

### Priority 3 --- Real outcome evaluation

Once AWS execution exists, collect:

``` text
Action
Before metrics
After metrics
Recovery time
Success/failure
Cost
Healing latency
```

This will provide the strongest evidence that the system is useful
beyond offline prediction.

### Priority 4 --- Paper results

After AWS/outcome data is available, finalize:

-   model results
-   risk/action distributions
-   latency
-   RCA
-   forecasting
-   healing effectiveness
-   recovery time
-   cost
-   comparison against simple threshold-based methods

------------------------------------------------------------------------

# 15. Paper Graphs

**Graphs are intentionally NOT being finalized yet.**

The underlying metrics and result artifacts are already being generated.

After the AWS/outcome experiments are complete, generate the final paper
figures from the final frozen result files.

Planned figures:

1.  Confusion Matrix
2.  ROC Curve
3.  Precision-Recall Curve
4.  Calibration Curve
5.  Failure Probability Distribution
6.  Raw Isolation Forest Distribution
7.  Calibrated Anomaly Distribution
8.  Risk Score Distribution
9.  Risk Band Distribution
10. Action Distribution
11. Failure Probability vs Risk
12. Anomaly vs Risk
13. Feature Importance
14. RCA Top Features
15. Inference Latency Distribution
16. Component Latency Breakdown
17. Actual vs Forecast CPU
18. Actual vs Forecast Memory
19. Forecast vs Persistence Baseline
20. AWS healing/recovery-time graphs once real outcomes exist

Do NOT create paper graphs from invented values.

------------------------------------------------------------------------

# 16. Final Project Status

``` text
ML Failure Prediction          DONE
Anomaly Detection              DONE
Anomaly Calibration            DONE
Risk Engine                    DONE
Decision Engine                DONE
Confidence                     DONE
Real Profile Prediction        DONE
Manual Telemetry               DONE
OOD Diagnostics                DONE
RCA / Explanation              IMPLEMENTED
Load Forecasting               IMPLEMENTED + EVALUATED
Latency Measurement            DONE
Paper Metrics                  DONE
Automated Testing              DONE
Streamlit UI                   DONE
AWS Real Execution             PENDING
Real Healing Outcomes          PENDING
Final Paper Graphs             PENDING
Final Paper Results             PENDING
```

## Bottom line

The AI/ML prototype is now in a strong evaluation-ready state.

The biggest remaining contribution is **real cloud execution and outcome
feedback**.

The AWS teammate should focus on the cloud
adapter/deployment/healing/outcome loop. The ML side should remain
stable while those real outcomes are collected.

Once real outcomes exist, we can evaluate not only:

> "Did the model predict risk?"

but also:

> "Did the selected action actually recover the workload, how quickly,
> and at what cost?"

That is the evidence needed to move the project from an offline ML
prototype toward a complete intelligent self-healing cloud system.

------------------------------------------------------------------------

## Important Development Rule

From this point onward:

**Do not change the model, risk formula, thresholds, or policy just to
improve screenshots or force desired actions.**

Any change to the ML/policy layer should be justified by an experiment
and reflected in the evaluation results.

Graphs will be generated later from the final frozen experimental
results.
