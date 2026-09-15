# Engineering & Research Decisions

This document logs key engineering and research decisions made during the development of the AI-Based Cloud Failure Prediction, Risk Assessment & Automated Healing system. Each decision is tied to the current implementation state and research implications.

---

## D-001 — Use Google Borg Workload Data for ML Training/Evaluation

**Date**: Project initialization  
**Status**: IMPLEMENTED  
**Decision**: Use historical Google Borg cluster task records as the primary training and evaluation dataset for ML models.

### Context / Problem
The system needs labeled failure data with high fidelity to train supervised and unsupervised anomaly detection models. Proprietary production cloud data was not available during development.

### Reason for choosing it
- Borg data is publicly available through Google Cloud publications and includes real-world workload characteristics (CPU, memory, task metadata, failure outcomes)
- Provides sufficient scale (~5,000+ cleaned task records) for meaningful model training and offline evaluation
- Allows reproducible, peer-reviewable research without proprietary data constraints

### Alternatives considered
- Synthetic workload generators (would be unrealistic)
- Public cloud trace datasets from other sources (limited availability of labeled failure data)
- Custom simulation (high effort, low realism)

### Why alternatives were rejected
Synthetic data cannot capture real failure patterns; real public datasets (Borg) provide legitimate, labeled ground truth.

### Impact on architecture
- ML training pipeline (`src/train_model.py`, `src/preprocessing.py`) is built around Borg data schema
- Feature engineering (`src/feature_engineering.py`) aligns directly to Borg task attributes (priority, CPU, memory, cycles per instruction, etc.)
- Model assumes 26-dimensional feature space from preprocessed Borg records
- Data preprocessing pipeline is specific to Borg CSV format

### Impact on experiments/paper
- Paper can demonstrate ML model performance on well-understood public data
- Limitations must be explicitly stated: offline evaluation ≠ production readiness
- Future real AWS data will serve a different purpose (online validation) than Borg training data

### Current implementation status
✓ IMPLEMENTED: Borg data loading, preprocessing, and model training are fully functional.

---

## D-002 — Separate Training Data from Real AWS Runtime Data

**Date**: Architecture design phase  
**Status**: PARTIALLY IMPLEMENTED (AI side only)  
**Decision**: Keep the training/evaluation pipeline (Borg-based) completely separate from the intended real AWS runtime telemetry stream.

### Context / Problem
The system needs two distinct data flows:
1. **Offline**: Borg data → training → evaluation → model artifacts
2. **Online**: AWS cloud telemetry → real-time prediction → decision → action

Confusing these would violate fundamental ML practices (data leakage, test set contamination) and make results uninterpretable.

### Reason for choosing it
- Standard ML practice: training distribution ≠ deployment distribution
- Borg represents historical reference workloads; AWS represents live runtime reality
- Keeps model development reproducible and independent of production data
- Enables honest evaluation of offline vs. online performance gaps

### Alternatives considered
- Retrain on real AWS data as it arrives (would create feedback loops and data leakage)
- Use Borg data for everything (would never validate on real infrastructure)

### Why alternatives were rejected
Retraining immediately on production outcomes creates circular dependencies and makes causality assessment impossible; Borg-only validation ignores production heterogeneity.

### Impact on architecture
- **Current**: `src/preprocessing.py` and `src/train_model.py` consume Borg CSV data
- **Current**: `src/predict.py` and `src/single_prediction.py` provide inference interface ready for AWS telemetry
- **Pending**: AWS teammate to implement real telemetry ingestion and outcome collection
- **Design**: `src/outcome.py` and `src/learning.py` are structured to receive real AWS outcomes for future offline analysis

### Impact on experiments/paper
- Paper must clearly state: offline metrics are on Borg evaluation data; production metrics will be on real AWS outcomes
- Cannot claim production readiness based solely on offline accuracy (0.997 on Borg evaluation)
- Forecasting baseline comparison (Borg data) is separate from real load prediction evaluation

### Current implementation status
✓ AI inference pipeline ready for AWS input  
⏳ PENDING: AWS teammate to provide real cloud telemetry and implement `CloudHealingAdapter`

---

## D-003 — Use Random Forest for Failure Prediction

**Date**: Model selection phase  
**Status**: IMPLEMENTED  
**Decision**: Train a Random Forest classifier to predict binary failure outcomes (task failure vs. normal completion).

### Context / Problem
System requires supervised failure prediction with:
- High accuracy (few false positives/negatives)
- Interpretability (feature importance available)
- Robustness to feature scale and missing patterns
- Fast inference (<50 ms per prediction)

### Reason for choosing it
- Random Forest provides strong baseline performance (ROC-AUC 0.9999 on Borg evaluation data)
- Feature importances are built-in and efficient for RCA
- No feature scaling required; handles mixed numeric types naturally
- Inference is fast: ~46 ms per prediction (measured)
- Well-established for infrastructure/cloud reliability problems

### Alternatives considered
- Logistic Regression (simpler but lower accuracy)
- Gradient Boosting/XGBoost (higher latency, similar results)
- Deep Learning (black-box, high latency, not justified by problem complexity)

### Why alternatives were rejected
LR too simple for this feature space; GB/XGBoost offer marginal gains for 2–3× inference cost; NN overkill and harder to explain.

### Impact on architecture
- Model artifact stored in `models/failure_predictor.pkl`
- Feature ordering tied to `model.feature_names_in_` (must preserve order)
- Inference in `src/predict.py` extracts class-1 probability
- RCA in `src/diagnosis.py` uses feature importances

### Impact on experiments/paper
- Paper will report Borg offline metrics:
  - Accuracy: 0.997
  - Precision: 0.9947
  - Recall: 0.9921
  - F1: 0.9934
  - ROC-AUC: 0.9999
  - PR-AUC: 0.9998
  - Brier Score: 0.0028
- Must note: these are Borg evaluation results, not production validation
- Latency benchmark (~46 ms) supports real-time decision making

### Current implementation status
✓ IMPLEMENTED: Model trained, evaluated, and integrated into prediction pipeline.

---

## D-004 — Use Isolation Forest for Anomaly Detection

**Date**: Model selection phase  
**Status**: IMPLEMENTED  
**Decision**: Use Isolation Forest to detect anomalous task behavior patterns independent of supervised failure labels.

### Context / Problem
Supervised failure prediction alone may miss novel failure modes not well-represented in training data. System needs a complementary unsupervised signal to detect out-of-distribution or unusual telemetry patterns.

### Reason for choosing it
- Isolation Forest is efficient (linear time), fast inference (~11 ms), and doesn't require labeled anomaly data
- Produces decision scores (not just binary labels), suitable for probability-like calibration
- Robust to high-dimensional data; no feature scaling needed
- Complements failure prediction: FF predicts label-based risk; IF detects data-space anomalies

### Alternatives considered
- Local Outlier Factor (LOF) (quadratic complexity, too slow)
- Autoencoders (high latency, complex training)
- One-Class SVM (slower than IF; similar conceptual role)

### Why alternatives were rejected
LOF and autoencoders violate latency budget (~11 ms IF vs. 100+ ms for alternatives); One-Class SVM is slower and no clearer conceptually.

### Impact on architecture
- Model artifact: `models/anomaly_detector.pkl` (Isolation Forest trained on Borg training set)
- Raw decision scores can be negative (valid model output, not an error)
- Scores are calibrated into [0, 1] anomaly signal before use (see D-005)
- Inference in `src/anomaly_detection.py`

### Impact on experiments/paper
- Paper will present IF as independent analytical path, not a replacement for supervised learning
- Must clarify: high anomaly score ≠ guaranteed failure, only suggests unusual pattern
- Baseline for comparison: what fraction of Borg evaluation tasks are flagged as anomalies?

### Current implementation status
✓ IMPLEMENTED: Model trained on Borg training set; decision scores generated and calibrated.

---

## D-005 — Calibrate Raw Isolation Forest Score

**Date**: Risk integration phase  
**Status**: IMPLEMENTED  
**Decision**: Transform raw Isolation Forest decision scores into a bounded [0, 1] anomaly signal using empirical quantile calibration.

### Context / Problem
Isolation Forest produces unbounded decision scores (can be negative or > 1). These scores cannot be directly averaged with failure probability (which is naturally bounded to [0, 1]) in a risk formula.

Directly treating negative scores as probabilities is scientifically incorrect and breaks the risk calculation.

### Reason for choosing it
- Empirical CDF / quantile-based mapping is non-parametric and data-driven
- Preserves monotonicity: lower (more anomalous) raw scores → higher calibrated anomaly value
- Fitted strictly on training data decision scores; no information leakage
- Simple, interpretable, fast (<1 ms overhead)

### Alternatives considered
- Sigmoid squashing (assumes normal distribution; not validated)
- Min-max scaling (sensitive to outliers in training set)
- Fixed reference bounds (arbitrary, not data-justified)

### Why alternatives were rejected
Sigmoid assumes distribution shape not verified; min-max brittle; fixed bounds ignore actual data characteristics.

### Impact on architecture
- Calibrator artifact: `models/anomaly_calibrator.pkl`
- Loaded lazily in `src/risk_score.py`
- Raw IF score → percentile rank in training distribution → 1.0 − percentile → clipped to [0, 1]
- Fallback empirical bounds if calibrator unavailable

### Impact on experiments/paper
- Paper must explain calibration step clearly: raw IF score is not user-facing anomaly
- Emphasize: calibration is dataset-specific; retraining on new Borg samples or AWS data requires recalibration
- User-facing UI should show calibrated anomaly ∈ [0, 1], not raw IF decision score

### Current implementation status
✓ IMPLEMENTED: Calibrator trained on Borg training set; integrated into risk pipeline.

---

## D-006 — Risk Score Combines Failure Probability and Anomaly Signal

**Date**: Risk formula design  
**Status**: IMPLEMENTED  
**Decision**: Compute risk as a weighted combination: 0.7 × failure_probability + 0.3 × calibrated_anomaly, bounded to [0, 1].

### Context / Problem
System must integrate two independent signals (failure prediction and anomaly detection) into a single risk score that drives action selection.

### Reason for choosing it
- **Weight 0.7 on failure probability**: Supervised prediction has higher model confidence (0.9999 ROC-AUC on Borg data)
- **Weight 0.3 on anomaly**: Unsupervised signal provides additional protective signal for novel patterns
- **Weighting philosophy**: Majority weight on well-calibrated supervised model; minority weight for additional coverage
- Bounded [0, 1] ensures risk fits naturally into [0.00, 1.00] policy bands

### Alternatives considered
- Equal weighting 0.5/0.5 (ignores confidence gap between models)
- ML-driven weight optimization (overfitting risk; not interpretable)
- Anomaly-only with failure as veto (loses failure prediction signal)

### Why alternatives were rejected
Equal weighting wastes the superior RF calibration; optimized weights overfit; anomaly-only loses primary signal.

### Impact on architecture
- Risk calculation: `src/risk_score.py` / `calculate_risk()`
- Formula: `risk = 0.7 * failure_prob + 0.3 * calibrated_anomaly`
- Result clipped to [0, 1]
- Risk drives action selection through policy bands (see D-007)

### Impact on experiments/paper
- Paper must justify the 0.7/0.3 weighting with the model confidence argument
- Sensitivity analysis recommended: vary weights ±0.1 and show impact on action distribution
- Document that risk formula is fixed; not changed to force desired UI outputs (see D-017)

### Current implementation status
✓ IMPLEMENTED: Formula integrated, tested, and validated across all prediction modes.

---

## D-007 — Fixed Risk Bands and Controlled Actions

**Date**: Policy design  
**Status**: IMPLEMENTED  
**Decision**: Use fixed risk bands to determine action:
- Risk < 0.30 → Normal
- 0.30 ≤ Risk < 0.50 → Scale Resources
- 0.50 ≤ Risk < 0.65 → Migrate VM
- Risk ≥ 0.65 → Restart Task

### Context / Problem
System must map continuous risk [0, 1] to discrete actions. Thresholds must be:
1. Justified by underlying model confidence and action severity
2. Fixed and not changed merely to create desired UI screenshots

### Reason for choosing it
- **0.30 (first intervention threshold)**: Risk below this is model-predicted "safe" state; prevents over-remediation
- **0.50 (moderate intervention)**: Non-destructive action (Scale Resources) below this
- **0.65 (severe intervention threshold)**: High-risk boundary; Restart Task (most disruptive) only above this
- **No adaptive/dynamic thresholds**: Keeps decision logic deterministic and auditable

### Alternatives considered
- Automatically adjust thresholds to force equal action distribution (scientific misconduct)
- Use only 2 actions (too coarse; loses graduated response)
- Continuous action selection (ill-defined; cloudy decision boundary)

### Why alternatives were rejected
Threshold tuning to achieve desired outputs violates research integrity; coarse actions reduce safety; continuous is ambiguous.

### Impact on architecture
- Policy defined in `src/policy.py`: `get_base_action(risk)` returns action string
- Thresholds are hardcoded; intentionally not configurable
- All decisions must satisfy policy (`check_policy()`)

### Impact on experiments/paper
- Real-profile evaluation on 5,000 Borg tasks shows:
  - Normal: 3,824 (76.48%)
  - Scale Resources: 32 (0.64%)
  - Migrate VM: 6 (0.12%)
  - Restart Task: 1,138 (22.76%)
- **All four actions are naturally reachable** using real reference profiles
- Manual telemetry (synthetic input) cannot reach Normal (minimum risk ≈ 0.424) — this is an **input-space limitation**, not a policy defect
- Paper must document this clearly: do NOT claim the policy is "broken" if it doesn't force equal action distribution

### Current implementation status
✓ IMPLEMENTED: Thresholds fixed in code; evaluated across 5,000 real profiles; all actions naturally reachable.

---

## D-008 — Replace Synthetic Six-Slider Prediction with Real-Profile-Backed Prediction

**Date**: UI/UX design refinement  
**Status**: IMPLEMENTED  
**Decision**: Replace the original synthetic six-slider input interface with a real-profile selection + optional edits interface.

### Context / Problem
The original manual input UI (priority, cpu_mean, assigned_memory, cycles_per_instruction, memory_accesses_per_instruction, sample_rate) could not naturally generate low-risk scenarios (minimum risk ≈ 0.424).

This created a false appearance that the policy never produces "Normal" actions, when actually the policy works fine on real data.

### Reason for choosing it
- Real Borg profiles are valid training examples; using them respects the model's native feature distribution
- Users can select a real profile and then make targeted edits to individual telemetry fields
- Guarantees the system can reach all policy bands (including Normal) because real reference data spans the risk space
- Preserves interpretability: all displayed features correspond to actual Borg task attributes

### Alternatives considered
- Keep synthetic six-slider but retune sliders to artificially cover risk space (input distortion; not realistic)
- Retrain model on synthetic data (loses fidelity to real Borg workloads)
- Remove the low-risk option from the UI (hides model capability)

### Why alternatives were rejected
Artificial slider retuning distorts the model's native feature space; synthetic retraining wastes real data; hiding options is dishonest.

### Impact on architecture
- UI profiles loaded from: `src/profile_prediction.py` / `load_reference_profiles()`
- Profile selection + optional edits (telemetry-only fields) via: `build_profile_prediction_features()`
- Streamlit UI uses real 5,000-profile set for user selection
- Legacy synthetic six-slider logic remains in `src/single_prediction.py` for historical boundary-analysis tests only

### Impact on experiments/paper
- Paper demonstrates that the system's action diversity is natural, not forced
- Real-profile action distribution is evidence of model reliability
- Manual telemetry remains available for exploratory analysis but is now clearly labeled as a separate input mode with known limitations

### Current implementation status
✓ IMPLEMENTED: Profile-backed prediction is primary UI mode; manual telemetry available as secondary mode.

---

## D-009 — Keep Manual Telemetry as a Separate Input Mode

**Date**: UI feature development  
**Status**: IMPLEMENTED  
**Decision**: Retain the ability to manually enter telemetry values (priority, CPU, memory, etc.) alongside the main profile-based prediction.

### Context / Problem
While real profiles are the primary input, interactive testing and "what-if" analysis require a manual override mode. However, this mode has different characteristics and limitations than the real data.

### Reason for choosing it
- Enables interactive exploration: users can understand how individual features affect risk
- Useful for debugging and validation
- Provides a comparison point: manual vs. profile-based predictions show the importance of realistic feature combinations
- Documented as a testing/exploratory feature, not the primary evaluation mode

### Alternatives considered
- Remove manual mode entirely (loses flexibility for interactive testing)
- Make it the primary mode (returns to input-space limitation problem)

### Why alternatives were rejected
Removing it eliminates useful exploratory capability; promoting it reintroduces the risk-space limitation issue.

### Impact on architecture
- Manual input specs built from reference quantiles: `src/profile_prediction.py` / `manual_telemetry_specs()`
- Validates that all edited fields are in EDITABLE_TELEMETRY_FEATURES (direct task-level observables)
- Separate code path in Streamlit UI: Manual Telemetry tab
- Results explicitly labeled with source (Manual vs. Profile)

### Impact on experiments/paper
- Paper clearly separates two input modes with a table:
  | Mode | Status | Risk Range | Normal Reachable | Use Case |
  | --- | --- | --- | --- | --- |
  | Real Profile | Primary | 0.0036–0.9964 | Yes | Evaluation, production-readiness |
  | Manual Telemetry | Secondary | 0.4238–0.9511 | No | Interactive testing, what-if analysis |

- Limitations are explicit in documentation; not hidden.

### Current implementation status
✓ IMPLEMENTED: Manual mode available; clearly documented as separate from primary profile-based evaluation.

---

## D-010 — Preserve Independent Telemetry Fields

**Date**: Feature engineering  
**Status**: IMPLEMENTED  
**Decision**: Do not artificially couple independent telemetry fields (e.g., CPU and memory) even if they might visually appear related.

### Context / Problem
Borg training data shows that CPU and memory usage are largely independent attributes of tasks. The temptation exists to couple them (e.g., "high CPU → always high memory") for simplicity, but doing so falsifies the model's native feature space.

### Reason for choosing it
- Model was trained on independent CPU and memory dimensions
- Real tasks often have uncorrelated CPU-memory profiles (e.g., memory-bound vs. compute-bound)
- Coupling would introduce artificial constraints and reduce prediction accuracy
- Preserves model fidelity to its training distribution

### Alternatives considered
- Enforce CPU ≈ Memory (simpler UI, but scientifically wrong)
- Use PCA to reduce dimensions (loses interpretability)

### Why alternatives were rejected
Coupling falsifies the model; PCA hides which features drive decisions.

### Impact on architecture
- Feature edits in profile mode are independent: each field editable separately
- No automatic constraint enforcement (e.g., cpu_mean ≠ assigned_memory)
- Preprocessing pipeline preserves all 26 original features

### Impact on experiments/paper
- Paper must document the 26-feature model architecture
- Demonstrate that CPU and memory are independently predictive
- RCA output will sometimes show CPU-only or memory-only impacts, reflecting real independence

### Current implementation status
✓ IMPLEMENTED: All 26 features preserved; no artificial coupling enforced.

---

## D-011 — Use P01/P99 Reference Ranges for OOD Diagnostics

**Date**: Diagnostics UI development  
**Status**: IMPLEMENTED  
**Decision**: Flag input features as out-of-distribution (OOD) if they fall outside the P01–P99 reference range from training data.

### Context / Problem
Users need feedback on whether their input is typical (in-distribution) or unusual (out-of-distribution). A clear, data-driven definition helps them understand prediction uncertainty.

### Reason for choosing it
- P01–P99 provides a robust, quantile-based in-distribution boundary
- Protects against extreme outliers while allowing 2% data naturally beyond bounds
- Fast to compute; stored in reference statistics
- Does not change prediction logic; only provides diagnostic feedback

### Alternatives considered
- Use 3-sigma bounds (assumes normality; not validated)
- Use 0–100% bounds (too permissive; would not flag any OOD cases)

### Why alternatives were rejected
Gaussian assumption unjustified; 0–100% gives no useful diagnostic signal.

### Impact on architecture
- Reference quantiles computed during preprocessing
- Diagnostic check in Streamlit UI: compare current input vs. P01/P99
- Labels: "IN RANGE", "LOW OOD", "HIGH OOD"
- Diagnostics section collapsed by default in UI

### Impact on experiments/paper
- OOD diagnostics are informational only; do not affect prediction or risk
- Paper documents: "OOD flags alert users to unusual inputs but do not modify model inference"
- Important: real AWS telemetry may naturally fall outside Borg P01–P99 (expected; does not indicate model failure)

### Current implementation status
✓ IMPLEMENTED: P01–P99 diagnostics integrated into Streamlit UI; collapsed by default.

---

## D-012 — Separate Real-Time Decision Path from Detailed RCA

**Date**: Latency optimization  
**Status**: IMPLEMENTED  
**Decision**: Split prediction workflow into two paths:
1. **Real-time path** (~60 ms): Input → RF → IF → Calibration → Risk → Decision
2. **Detailed RCA path** (~89 s sample): Prediction → Feature sensitivity → RCA explanation

### Context / Problem
Real-time remediation requires fast decisions (<100 ms). Detailed root-cause analysis using feature replacement is expensive (~89 seconds observed).

Mixing them makes system appear unresponsive or forces RCA to be dropped from real-time flow.

### Reason for choosing it
- Separates concerns: decision-making vs. explanation
- Real-time path meets latency budget: mean 60.89 ms, P99 62.53 ms
- Detailed RCA can run asynchronously (post-decision) or be skipped in time-critical scenarios
- Allows paper to report accurate latency numbers without hiding expensive operations

### Alternatives considered
- Always include detailed RCA (violates latency budget; real-time unusable)
- Skip RCA entirely (loses explainability)
- Optimize RCA in-place (high effort; still fundamentally O(features × predictions))

### Why alternatives were rejected
Always-include makes system too slow for production; skipping hides important diagnostics; optimization helps but doesn't solve the O(n) complexity.

### Impact on architecture
- Real-time components: `src/predict.py`, `src/anomaly_detection.py`, `src/risk_score.py`, `src/decision_engine.py`
- Detailed RCA: `src/diagnosis.py` / `explain_prediction()` (called separately, optional)
- Streamlit UI can show quick decision immediately, then load RCA details asynchronously

### Impact on experiments/paper
- Paper must report two separate latency numbers:
  | Path | Component | Mean | P50 | P95 | P99 |
  | --- | --- | --- | --- | --- | --- |
  | Real-time | RF+IF+Calibration+Risk+Decision | 60.89 ms | 60.68 ms | 62.30 ms | 62.53 ms |
  | Detailed RCA (sample) | Feature sensitivity (1 row) | ~88,983 ms | — | — | — |
- Critical: do NOT mix these numbers into a single "latency" figure
- Paper must recommend: "Detailed RCA should be optimized or run asynchronously before production deployment"

### Current implementation status
✓ IMPLEMENTED: Both paths implemented; latency measured separately; Streamlit UI ready for async RCA loading.

---

## D-013 — RCA Is Model-Based Explanation, Not Causal Proof

**Date**: Explainability design  
**Status**: IMPLEMENTED  
**Decision**: Clearly label all RCA output as "model-based explanation — not causal proof" and avoid causal language.

### Context / Problem
Identifying that feature X has high importance in the model is NOT proof that X caused the failure. Users might misinterpret RCA as definitive root-cause diagnosis.

### Reason for choosing it
- Essential scientific integrity: correlation ≠ causation
- Machine learning explanations describe model behavior, not ground truth
- Prevents users from taking incorrect remedial actions based on false causal beliefs

### Alternatives considered
- Market RCA as "root cause analysis" (scientifically misleading)
- Don't explain predictions (loses valuable signal)

### Why alternatives were rejected
Causal claim is false; omitting explanations wastes valuable model insights.

### Impact on architecture
- Every RCA output includes: `"disclaimer": "Model-based explanation — not causal proof."`
- Feature impacts labeled: "increases_risk" / "decreases_risk" (not "causes")
- Documentation emphasizes predictive association, not causality
- `src/diagnosis.py` has built-in disclaimer in all outputs

### Impact on experiments/paper
- Paper must use correct terminology:
  - ✓ "Features associated with increased risk"
  - ✓ "Model drivers"
  - ✓ "Predictive factors"
  - ✗ "Root cause"
  - ✗ "Causes the failure"
- Acknowledge limitation clearly: "Model-based RCA provides evidence, not proof, of contributing factors"

### Current implementation status
✓ IMPLEMENTED: Disclaimer in all RCA outputs; terminology consistently non-causal.

---

## D-014 — Load Forecasting Is an Independent Analytical Path

**Date**: Feature development  
**Status**: IMPLEMENTED  
**Decision**: Implement load forecasting (CPU and memory prediction) as a **separate analytical module** that does NOT currently feed into failure probability, anomaly detection, risk, or action selection.

### Context / Problem
Predicting future load is valuable for proactive provisioning but is conceptually distinct from failure prediction. Including it in the risk pipeline without validation risks clouding causality.

### Reason for choosing it
- Separates concerns: reactive failure mitigation vs. proactive load management
- Allows independent evaluation and improvement of forecasting
- Forecasting can be published separately; failure prediction independently evaluated
- Enables future work: once validated, forecasting could inform preventive actions

### Alternatives considered
- Integrate forecasting into risk calculation immediately (premature; may degrade performance)
- Omit forecasting entirely (loses valuable analysis)

### Why alternatives were rejected
Immediate integration risks introducing confounds; omitting it hides a useful capability.

### Impact on architecture
- Forecasting pipeline: `src/forecasting.py` / `load_ordered_load_series()`, `evaluate_temporal_forecast()`, `forecast_cpu_memory()`
- Separate Streamlit section: "Load Forecast" (shown but not used in decision logic)
- No connection to failure prediction, anomaly detection, or risk calculation
- Results stored separately in `results/forecasting/forecast_metrics.json`

### Impact on experiments/paper
- Paper presents forecasting as independent exploratory analysis
- Clearly states: "Forecasting does NOT currently modify prediction, risk, or action selection"
- Forecasting can be discussed in future work or as supplementary analysis

### Current implementation status
✓ IMPLEMENTED: Forecasting module fully functional but intentionally decoupled from risk/decision paths.

---

## D-015 — Compare Forecasting Against Persistence Baseline

**Date**: Forecasting evaluation  
**Status**: IMPLEMENTED  
**Decision**: Evaluate linear forecasting model against a persistence baseline (last observed value = next prediction) on a chronological holdout.

### Context / Problem
Forecasting performance is only meaningful relative to baseline. Persistence (no-change model) is the strongest weak baseline for time-series data.

### Reason for choosing it
- Persistence is the standard forecasting baseline
- If the model cannot beat persistence, it has learned nothing from the temporal structure
- Chronological holdout ensures no future leakage
- Enables honest comparison

### Alternatives considered
- No baseline (impossible to judge model quality)
- Naive averages (weaker than persistence)

### Why alternatives were rejected
Baselines are essential for credible evaluation; naive methods are too simple to be useful.

### Impact on architecture
- Evaluation: `src/forecasting.py` / `evaluate_temporal_forecast()` compares linear vs. persistence
- Results: `results/forecasting/forecast_metrics.json` shows both MAE, RMSE, MAPE
- Chronological split: 2,389 training, 598 test timestamps (no leakage verified)

### Impact on experiments/paper
- **CRITICAL**: Measured results show persistence baseline performs better on MAE:
  | Resource | Linear MAE | Persistence MAE | Winner |
  | --- | --- | --- | --- |
  | CPU | 0.01031 | (better) | Persistence |
  | Memory | 0.00558 | (better) | Persistence |

- **Paper must NOT claim forecasting superiority**
- Correct statement: "A linear forecasting component was implemented and evaluated using a chronological holdout. However, the persistence baseline achieved lower MAE for both CPU and memory, indicating that the current forecasting component requires further improvement."
- This is honest and sets up future work: improve forecasting to beat baseline

### Current implementation status
✓ IMPLEMENTED: Linear model trained; persistence baseline evaluated; honest comparison published.

---

## D-016 — Measure Actual Latency Instead of Claiming Real-Time Performance

**Date**: Performance benchmarking  
**Status**: IMPLEMENTED  
**Decision**: Measure actual wall-clock latency of prediction, anomaly, risk, and decision components separately; report measured values, not theoretical claims.

### Context / Problem
"Real-time" is vague. System must demonstrate actual inference latency with measured percentiles.

### Reason for choosing it
- Enables external validation and reproducibility
- Forces honest assessment of deployment feasibility
- Allows comparison with actual AWS response requirements

### Alternatives considered
- Claim "real-time" without measurement (unsubstantiated)
- Report only mean latency (hides variance; P95/P99 matter for SLA)

### Why alternatives were rejected
Unmeasured claims are unreliable; mean-only reporting is incomplete.

### Impact on architecture
- Latency instrumentation: `src/pipeline.py` or main inference loop times components
- Results stored: `results/latency/latency_benchmark.csv` (component-level breakdown)
- Streamlit UI shows latency summary to users

### Impact on experiments/paper
- Paper reports measured latency from `results/latency/latency_summary.csv`:
  | Component | Mean | P50 | P95 | P99 |
  | --- | --- | --- | --- | --- |
  | Random Forest | 46.71 ms | 48.12 ms | 49.23 ms | 49.44 ms |
  | Isolation Forest | 11.18 ms | 12.09 ms | 12.45 ms | 12.52 ms |
  | Calibration | 0.18 ms | 0.19 ms | 0.21 ms | 0.22 ms |
  | Risk | 0.18 ms | 0.19 ms | 0.22 ms | 0.23 ms |
  | Decision | 0.15 ms | 0.17 ms | 0.19 ms | 0.19 ms |
  | **Total (real-time)** | **60.89 ms** | **60.68 ms** | **62.30 ms** | **62.53 ms** |
  | RCA (sample) | ~88,983 ms | — | — | — |

- Conclusion: "System meets sub-100 ms latency requirement for real-time path; detailed RCA must be asynchronous."

### Current implementation status
✓ IMPLEMENTED: Latency measured on all components; results stored and visualized.

---

## D-017 — Do Not Retrain/Change Models Merely to Force Desired UI Actions

**Date**: Research integrity  
**Status**: IMPLEMENTED  
**Decision**: Keep models, thresholds, and feature engineering fixed. Do not retrain or adjust hyperparameters to create a desired UI screenshot or action distribution.

### Context / Problem
If the system doesn't show all four actions, the temptation is to "tune" the model until it does. This violates research integrity and makes results unreproducible.

### Reason for choosing it
- Scientific validity: results must be honest, not engineered to impress
- Reproducibility: locking models enables peer review
- Integrity: avoids p-hacking and selective reporting
- The real problem was the input space (D-008), not the model — this was fixed correctly

### Alternatives considered
- Retrain model to force action distribution (research misconduct)
- Use hand-tuned thresholds to create desired outputs (also misconduct)

### Why alternatives were rejected
Both violate fundamental research integrity and make the paper unreviewable.

### Impact on architecture
- Models in `models/` are frozen (do not retrain without explicit justification)
- Thresholds in `src/policy.py` are hardcoded and documented in decisions
- Feature engineering pipeline is locked to Borg schema
- All changes must pass through the decision review process (this document)

### Impact on experiments/paper
- Paper documents: "Model architecture, thresholds, and feature engineering were fixed prior to final evaluation to ensure result integrity"
- Any deviations must be explicitly noted and justified
- Real-profile evaluation shows all four actions naturally reach; this is the honest answer

### Current implementation status
✓ IMPLEMENTED: Models frozen; thresholds documented; no post-hoc tuning for screenshots.

---

## D-018 — Keep AWS Healing Behind CloudHealingAdapter

**Date**: Architecture design  
**Status**: IMPLEMENTED (Interface), PENDING (AWS Implementation)  
**Decision**: Define a clear interface boundary: AI system produces structured decisions; AWS teammate implements the `CloudHealingAdapter` to execute actions on real infrastructure.

### Context / Problem
AI decision logic must be decoupled from cloud-specific execution. Different clouds (AWS, Azure, GCP) have different APIs and safety mechanisms.

### Reason for choosing it
- Separation of concerns: decision ≠ execution
- Allows parallel development: AI team finalizes decisions; AWS team builds adapter independently
- Enables testing without real AWS access (mock/dry-run implementations)
- Clear contract: AI produces `StructuredDecision`; AWS consumes it and produces `HealingOutcome`

### Alternatives considered
- Embed AWS SDK calls directly in AI code (tight coupling; blocking)
- Have AI team implement AWS integration (resource mismatch; wrong expertise)

### Why alternatives were rejected
Tight coupling makes parallel development impossible; AWS integration is the teammate's responsibility.

### Impact on architecture
- Adapter interface: `src/healing.py` / `CloudHealingAdapter` (abstract base)
- Concrete implementation: `DryRunHealingAdapter` (safe local mock)
- AI decision export: `src/decision_engine.py` / `make_decision()` returns structured dict
- AWS teammate: implement real adapter and outcome validation

### Impact on experiments/paper
- Paper section: "AWS Integration (Pending)"
- Clearly state: "The structured decision is production-ready; AWS integration is in progress by the team's infrastructure specialist"
- Do NOT claim AWS healing works until the adapter is implemented and tested

### Current implementation status
✓ Interface defined and tested with dry-run mock  
⏳ PENDING: AWS teammate to implement real `CloudHealingAdapter` and outcome collection

---

## D-019 — Use Dry-Run / Safe Execution Before Real Healing

**Date**: Safety design  
**Status**: IMPLEMENTED (Dry-run), PENDING (Real AWS)  
**Decision**: Before executing real infrastructure changes, always validate through dry-run / sandbox mode.

### Context / Problem
Remediation actions (restart task, migrate VM) can impact user workloads. The system must be proven safe before enabling real execution.

### Reason for choosing it
- Prevents accidental infrastructure damage during testing/validation
- Allows A/B testing: dry-run decisions vs. real decisions
- Builds confidence in decision quality
- Standard DevOps practice: CI pipeline → staging → production

### Alternatives considered
- Jump directly to real execution (irresponsible; high risk)
- Skip execution testing entirely (untested; production unknown)

### Why alternatives were rejected
Real execution without testing is unsafe; skipping testing guarantees issues in production.

### Impact on architecture
- `DryRunHealingAdapter` in `src/healing.py` logs intent without executing
- Streamlit UI shows recommended action; user initiates dry-run or real execution
- Dry-run results: `{"status": "DRY_RUN", "action": "...", "incident": "..."}`
- Real execution: AWS adapter (pending) performs actual remediation

### Impact on experiments/paper
- Paper: "System evaluated in dry-run mode; real AWS integration with safety guardrails in progress"
- Do NOT report metrics from real execution unless it has been completed and validated

### Current implementation status
✓ IMPLEMENTED: Dry-run fully functional; safe for testing  
⏳ PENDING: Real AWS execution and before/after outcome collection

---

## D-020 — Real AWS Outcome Feedback Is Required for Final Healing Evaluation

**Date**: Evaluation design  
**Status**: PENDING  
**Decision**: Final validation of healing effectiveness requires real before/after metrics:
1. Detect incident (before metrics)
2. AI decides action
3. Execute action (AWS)
4. Measure recovery (after metrics)
5. Compare: success/failure, recovery time, resource impact
6. Store outcome
7. Analyze: did the action prevent/mitigate failure?

### Context / Problem
Offline evaluation on Borg data proves failure prediction; online evaluation on real AWS proves that remediation actually works in production.

### Reason for choosing it
- Answers the question: does the AI system actually help production reliability?
- Requires real data, not simulated outcomes
- Enables feedback loop: real outcomes → offline retraining → better models

### Alternatives considered
- Use simulated outcomes (not credible; defeats the purpose)
- Skip outcome validation (proceed to production untested; irresponsible)

### Why alternatives were rejected
Simulated outcomes are not evidence; skipping validation is unsafe.

### Impact on architecture
- Outcome schema: `src/outcome.py` / `HealingOutcome` (before metrics, action, after metrics, success/failure, recovery_time)
- Storage: OutcomeStore (pending AWS implementation)
- Retraining loop: `src/learning.py` (ready to consume outcomes)

### Impact on experiments/paper
- **CRITICAL**: Real AWS outcome collection is a gate for deployment approval
- Paper must include: "Healing effectiveness validated through [N] real AWS incidents; observed success rate X%, mean recovery time Y seconds"
- Do NOT claim production readiness without real outcome data

### Current implementation status
⏳ PENDING: AWS teammate to collect real before/after metrics and outcomes
⏳ PENDING: Offline retraining on real outcomes (future enhancement)

---

## Decision Principles

The project adheres to the following engineering and research principles:

1. **Do not manipulate model outputs to create desired screenshots.**
   - Models and thresholds are fixed; input-space changes are the correct approach.

2. **Keep ML prediction separate from cloud execution.**
   - AI decides; AWS adapter executes; outcomes feed back to learning.

3. **Prefer real/reference workload profiles over unrealistic synthetic feature combinations.**
   - Real profiles respect the model's native feature distribution and training data characteristics.

4. **Measure actual latency.**
   - Real numbers enable honest assessment; missing hidden operations is scientifically dishonest.

5. **Clearly distinguish prediction, anomaly detection, RCA, forecasting, and healing.**
   - Each has separate architectural components and evaluation criteria.

6. **Do not claim causality from model-based explanation.**
   - "Associated with" ≠ "caused by"; preserve this distinction in all documentation.

7. **Do not claim forecasting superiority when the baseline performs better.**
   - Honest evaluation: if persistence beats the model, say so and plan improvements.

8. **Validate real healing using actual before/after outcomes.**
   - Simulated outcomes are not evidence; real AWS data is required for production claims.

9. **Maintain research integrity throughout development and deployment.**
   - Reproducibility, transparency, and honest reporting take precedence over flashy demos.

10. **Separate offline evaluation (Borg data) from online validation (AWS production).**
    - Different purposes; cannot be conflated; both required for complete picture.

---

## Summary Table

| Decision | Status | Impact |
| --- | --- | --- |
| D-001: Borg data | ✓ Implemented | ML training pipeline established |
| D-002: Separate Borg/AWS | ✓ AI Ready / ⏳ AWS Pending | Clear boundary for integration |
| D-003: Random Forest | ✓ Implemented | 0.9999 ROC-AUC; 46 ms inference |
| D-004: Isolation Forest | ✓ Implemented | 11 ms anomaly detection |
| D-005: Anomaly calibration | ✓ Implemented | Bounded [0,1] anomaly signal |
| D-006: Risk formula (0.7/0.3) | ✓ Implemented | Integrated failure + anomaly signals |
| D-007: Fixed risk bands | ✓ Implemented | All four actions naturally reachable |
| D-008: Profile-backed UI | ✓ Implemented | Real profiles; Normal action reachable |
| D-009: Manual telemetry mode | ✓ Implemented | Secondary exploratory input mode |
| D-010: Independent fields | ✓ Implemented | 26-feature model preserved |
| D-011: P01/P99 OOD diagnostics | ✓ Implemented | Reference-based input assessment |
| D-012: Separate RCA path | ✓ Implemented | 60 ms real-time; 89s async RCA |
| D-013: RCA ≠ causality | ✓ Implemented | Disclaimer in all outputs |
| D-014: Forecasting independent | ✓ Implemented | Separate analytical module |
| D-015: Forecast vs. persistence | ✓ Implemented | Persistence baseline better (honest) |
| D-016: Measured latency | ✓ Implemented | Sub-100 ms real-time path |
| D-017: No post-hoc tuning | ✓ Implemented | Models/thresholds frozen |
| D-018: CloudHealingAdapter | ✓ Interface / ⏳ AWS Impl | Clear AI↔AWS boundary |
| D-019: Dry-run before real | ✓ Dry-run / ⏳ Real AWS | Safe testing → production path |
| D-020: Real outcome validation | ⏳ Pending | Required gate for deployment |

---

**Document Version**: 1.0  
**Last Updated**: 2026-08-27  
**Audience**: Team, supervisor, paper reviewers  
**Status**: Living document; decisions are final unless explicitly superseded with rationale.
