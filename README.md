
# WESAD-based Output-Side Privacy Leakage Study

This repository contains the WESAD-based experimental code for studying output-side privacy leakage in physiological recommender systems.

### The Central Question
> Even if raw physiological signals are hidden, can an observer still infer a user’s sensitive state (e.g., stress) from the recommendation list alone?

To study this question, we build a controlled recommendation benchmark on top of WESAD physiological state trajectories and evaluate whether different serving-time recommendation policies leak sensitive state information through observable Top-K recommendation logs.

---

## 1. Project Overview

This codebase implements the WESAD extension described in the paper:

- **WESAD Preprocessing**
  - Loading subject .pkl files and questionnaire .csv.
  - Parsing condition intervals from quest.csv.
  - Sliding-window segmentation.
  - Handcrafted feature extraction.

- **Physiological State Classification**
  - Subject-level train/validation/test split.
  - Multiclass state classifier.
  - GT-state (Ground Truth) and Predicted-state evaluation settings.

- **Recommendation Simulation**
  - Baseline state-adaptive ranking.
  - Fixed filtering.
  - Noisy ranking.
  - Risk-aware reranking.
  - Smoothing / serving-time mitigation variants.

- **Exposure-aware Privacy Modeling**
  - Estimating item exposure statistics from recommendation logs.
  - Exposure-mode ablation.

- **State-inference Attacks**
  - Logistic Regression, Random Forest, MLP, BiLSTM, XGBoost.

- **Evaluation and Visualization**
  - Relevance, risk, sensitive exposure.
  - Temporal stability / diversity.
  - Attacker AUC.
  - Parameter sweeps and trade-off plots.

---

## 2. Repository Structure
```text
physio-rec-privacy/
├── README.md
├── requirements.txt
├── scripts/
│   └── run_wesad_pipeline.py
└── src/
    ├── __init__.py
    ├── config.py
    ├── utils.py
    ├── data_wesad.py
    ├── state_model.py
    ├── catalog.py
    ├── exposure.py
    ├── policies.py
    ├── attackers.py
    ├── evaluation.py
    ├── plotting.py
    └── experiments.py
```

---

## 3. Installation
We recommend Python 3.10 or newer.

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Manual installation:**
```bash
pip install numpy scipy scikit-learn pandas matplotlib torch xgboost
```

---

## 4. Dataset Preparation

This project uses the WESAD dataset.

**Expected directory layout:**
```text
data/WESAD/
├── S2/
│   ├── S2.pkl
│   └── S2_quest.csv
├── S3/
│   ├── S3.pkl
│   └── S3_quest.csv
...
├── S17/
│   ├── S17.pkl
│   └── S17_quest.csv
```

*Note: Subject S12 is excluded following the current implementation.*

**Set the dataset path:**
- **Option 1:** Environment variable
```bash
export WESAD_DATA_DIR=./data/WESAD
```
- **Option 2:** Edit the default path in src/config.py.

---

## 5. What the Benchmark Contains

### 5.1 Real Component
Derived from WESAD physiological trajectories and labels:
- States: baseline, stress, amusement, meditation.
- Extracted features from: ECG, EDA, Respiration, Temperature, ACC, BVP.
- Default windowing: 60s window length, 30s step size.

### 5.2 Instantiated Recommendation Environment
Since WESAD lacks recommendation logs, we instantiate:
- A 12-item wellness micro-catalog.
- State-dependent utility scores.
- Item sensitivity scores.
- Comfort penalties.
- Exposure-aware privacy penalties.

---

## 6. Main Experimental Logic
1. **Build** a window-level WESAD feature dataset.
2. **Train** a multiclass state classifier with a subject-level split.
3. **Construct** two state sources:
   - GT-States: Ground-truth WESAD labels.
   - Pred-States: Classifier-predicted states.
4. **Generate** recommendation logs under different serving policies.
5. **Learn** exposure statistics from baseline recommendation logs.
6. **Evaluate** whether logs reveal stress states using Attackers.
7. **Run** paper-aligned ablations and parameter sweeps.

---

## 7. Recommendation Policies
| Policy Name | Description |
| :--- | :--- |
| **Baseline-R** | Direct state-adaptive ranking using only relevance. |
| **Fixed-Filter** | Hard filtering of risky items before ranking. |
| **DP-Noisy** | Adds Laplace noise to ranking scores. |
| **RiskOnly-ItemSens** | Penalizes only item sensitivity. |
| **RA (Risk-Aware)** | Reranking using relevance minus privacy risk. |
| **RA + Mix / Mitigation** | Risk-aware reranking + temporal smoothing/mixing. |
| **Sensitive-Cap** | Controls cumulative exposure ratio of sensitive items. |
| **Popular-Mix** | Injects state-independent popularity bias into ranking. |

---

## 8. Attack Models
We measure inference risk from observable recommendation logs alone:
- **Static Attackers:** Logistic Regression, Random Forest, MLP.
- **Sequential Attackers:** BiLSTM, XGBoost.
- **Embedding-level Attacker:** Upper bound attack using physiological embeddings.

---

## 9. Evaluation Metrics
- **MeanRel:** Average recommendation relevance.
- **MeanRisk:** Average risk score of exposed items.
- **HighSensRatio:** Fraction of highly sensitive items recommended.
- **MeanJaccard:** Overlap between adjacent Top-K lists.
- **CategoryDiversity:** Diversity of item categories within a list.
- **TemporalDiversity:** Change across adjacent recommendation lists.
- **AttackAUC:** strongest attacker AUC for inferring stress from logs.

> **Interpretation:** Values near 0.5 indicate near-chance detectability (high privacy); higher values mean stronger leakage.

---

## 10. Running the Full Pipeline
```bash
python scripts/run_wesad_pipeline.py
```

**The script will:**
1. Build WESAD features.
2. Train the state classifier.
3. Evaluate embedding-level leakage.
4. Run recommendation leakage experiments for both GT-States and Pred-States.
5. Save CSV outputs and generate plots.

---

## 11. Main Outputs
**Result Tables (.csv):**
- wesad_baseline_multi_seed_gtstates.csv
- wesad_lambda_sweep_gtstates.csv
- wesad_alpha_tv_auc_gtstates.csv

**Generated Plots:**
- Lambda trade-off curves.
- Lambda sweep with AUC confidence intervals.
- Attack AUC vs. Drop probability.
- Alpha-TV-AUC analyses.

---

## 12. Mapping Paper Sections to Code
| Paper Section | Main Functionality | Code Location |
| :--- | :--- | :--- |
| **Section 3.3: Detectability** | Attack formulation | src/attackers.py, src/evaluation.py |
| **Section 4.1: serving Rule** | Risk scoring & reranking | src/catalog.py, src/policies.py |
| **Section 4.2: Exposure Risk** | Exposure calibration | src/exposure.py |
| **Section 4.3: Mitigation** | Smoothing / Mixing | src/policies.py |
| **Section 5.3: WESAD extension** | Preprocessing & Experiments | src/data_wesad.py, src/experiments.py |
| **Figures / Sweeps** | Plotting & Reporting | src/plotting.py |

---

## 13. Notes on Interpretation
- **Controlled Benchmark:** This is a WESAD-grounded benchmark, not a production recommender. It isolates the mechanism of output-side leakage using real physiological trajectories.
- **GT-States vs Pred-States:** Separates leakage mechanism from upstream state-estimation noise.
- **About RA+Mix:** Mitigation through risk-aware reranking + temporal/history-aware smoothing.

---

## 14. Reproducibility Notes
- Randomness is controlled via set_global_seed(...).
- **Subject-level splitting** prevents data leakage.
- Multi-seed evaluation is supported.
- **Confidence Intervals:** Computed via non-parametric bootstrap.

---

## 15. Typical Workflow
1. Prepare WESAD files under data/WESAD.
2. Run the pipeline: python scripts/run_wesad_pipeline.py.
3. Inspect generated CSVs for results.
4. View saved Figures in the output directory.

---

## 16. Known Limitations
- Exact table numbering may require additional wrappers.
- Emphasizes WESAD experiments over live AR/XR deployment.
- Settings configured in src/config.py.
- Mitigation combines reranking and smoothing.
```

---

