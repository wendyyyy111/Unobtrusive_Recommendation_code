This repository contains the WESAD-based experimental code for studying **output-side privacy leakage in physiological recommender systems**.

The central question is:

> Even if raw physiological signals are hidden, can an observer still infer a user’s sensitive state (e.g., stress) from the recommendation list alone?

To study this question, we build a controlled recommendation benchmark on top of WESAD physiological state trajectories and evaluate whether different serving-time recommendation policies leak sensitive state information through observable Top-\(K\) recommendation logs.

---

1. Project Overview

This codebase implements the WESAD extension described in the paper:

- **WESAD preprocessing**
  - loading subject `.pkl` files and questionnaire `.csv`
  - parsing condition intervals from `quest.csv`
  - sliding-window segmentation
  - handcrafted feature extraction
- **Physiological state classification**
  - subject-level train/validation/test split
  - multiclass state classifier
  - GT-state and predicted-state evaluation settings
- **Recommendation simulation**
  - baseline state-adaptive ranking
  - fixed filtering
  - noisy ranking
  - risk-aware reranking
  - smoothing / serving-time mitigation variants
- **Exposure-aware privacy modeling**
  - estimating item exposure statistics from recommendation logs
  - exposure-mode ablation
- **State-inference attacks**
  - Logistic Regression
  - Random Forest
  - MLP
  - BiLSTM
  - XGBoost
- **Evaluation and visualization**
  - relevance, risk, sensitive exposure
  - temporal stability / diversity
  - attacker AUC
  - parameter sweeps and trade-off plots

---

2. Repository Structure

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

3. Installation

We recommend Python 3.10 or newer.

Install dependencies with:

```bash
pip install -r requirements.txt

If needed, install manually:

bash
pip install numpy scipy scikit-learn pandas matplotlib torch xgboost

4. Dataset Preparation
This project uses the WESAD dataset.

Expected directory layout:

text
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

Subject S12 is excluded, following the current implementation.

Set the dataset path
Option 1: environment variable

bash
export WESAD_DATA_DIR=./data/WESAD

Option 2: edit the default path in src/config.py.

5. What the Benchmark Contains
5.1 Real component
The real component comes from WESAD physiological trajectories and labels:

baseline
stress
amusement
meditation
The code extracts window-level features from:

ECG
EDA
Respiration
Temperature
ACC
BVP
Default windowing:

window length: 60 seconds
step size: 30 seconds
5.2 Instantiated recommendation environment
Because WESAD does not contain real recommendation logs, we instantiate:

a 12-item wellness micro-catalog
state-dependent utility scores
item sensitivity scores
comfort penalties
exposure-aware privacy penalties
This creates a transparent and reproducible recommendation environment for measuring output-side sensitive-state leakage.

6. Main Experimental Logic
The pipeline follows these steps:

Build a window-level WESAD feature dataset
Train a multiclass state classifier with a subject-level split
Construct two state sources:
GT-States: ground-truth WESAD labels
Pred-States: classifier-predicted states
Generate recommendation logs under different serving policies
Learn exposure statistics from baseline recommendation logs
Evaluate whether recommendation logs reveal stress state
Run paper-aligned ablations and parameter sweeps
7. Recommendation Policies
The repository includes several serving-time policies.

Baseline-R
Direct state-adaptive ranking using only relevance.

Fixed-Filter
Hard filtering of certain risky items before ranking.

DP-Noisy
Adds Laplace noise to ranking scores.

RiskOnly-ItemSens
Penalizes only item sensitivity.

RA
Risk-aware reranking using relevance minus privacy risk.

RA + Smooth / serving-time mitigation
A serving-time mitigation variant that reduces state-coded output shifts through history-aware smoothing and privacy penalties.

Sensitive-Cap
Controls the cumulative exposure ratio of highly sensitive items.

Popular-Mix
Injects state-independent popularity bias into ranking.

8. Attack Models
We measure whether the sensitive state can be inferred only from observable recommendation logs.

Static attackers
Logistic Regression
Random Forest
MLP
Stronger sequential attackers
BiLSTM
XGBoost on flattened sequence features
Embedding-level attacker
We also include a GERAI-style embedding-level attack that predicts whether a WESAD window belongs to a sensitive state directly from physiological feature embeddings.

9. Evaluation Metrics
We report:

MeanRel: average recommendation relevance
MeanRisk: average risk score of exposed items
HighSensRatio: fraction of highly sensitive recommended items
MeanJaccard: overlap between adjacent Top-
𝐾
K lists
CategoryDiversity: diversity of item categories within a list
TemporalDiversity: change across adjacent recommendation lists
AttackAUC: strongest attacker AUC for inferring stress from recommendation logs
Interpretation:

lower AttackAUC means lower sensitive-state leakage
values near 0.5 indicate near-chance detectability
10. Running the Full Pipeline
Run the main WESAD pipeline:

bash
python scripts/run_wesad_pipeline.py

The script will:

build WESAD features
train the state classifier
evaluate embedding-level leakage
run recommendation-log leakage experiments for:
GT-States
Pred-States
save CSV outputs for major analyses
11. Main Outputs
The pipeline produces result tables such as:

wesad_baseline_multi_seed_gtstates.csv
wesad_baseline_multi_seed_predstates.csv
wesad_lambda_sweep_gtstates.csv
wesad_lambda_sweep_predstates.csv
wesad_alpha_tv_auc_gtstates.csv
It also generates plots, such as:

lambda trade-off curves
lambda sweep with AUC confidence intervals
attack AUC vs. drop probability
alpha-TV-AUC analyses
12. Mapping from Paper Sections to Code
Paper Section	Main Functionality	Code Location
Section 2: Framework / detectability	log-based attack formulation	src/attackers.py, src/evaluation.py
Section 3.1: Risk-aware serving rule	risk scoring and reranking	src/catalog.py, src/policies.py
Section 3.2: Exposure risk	exposure calibration	src/exposure.py
Section 3.3: Serving-time mitigation	smoothing / mixing-style post-processing	src/policies.py
Section 4.3: WESAD extension	preprocessing, subject split, experiments	src/data_wesad.py, src/state_model.py, src/experiments.py
Figures / sweeps	plotting and reporting	src/plotting.py, src/experiments.py
13. Notes on Interpretation
Controlled benchmark, not a production recommender
This code implements a controlled WESAD-grounded benchmark, not a natural recommendation-log dataset. It combines:

real physiological state trajectories
instantiated recommendation utilities and privacy annotations
Its purpose is to isolate and study the mechanism of output-side sensitive-state leakage.

GT-States vs Pred-States
We report both:

GT-States to study the mechanism under clean labels
Pred-States to study the effect under upstream state-estimation noise
About “RA+Mix”
The paper discusses state-independent mixing as a mechanism for reducing distinguishability. In the current code release, the main serving-time mitigation variant is implemented using privacy-aware reranking together with temporal/history-aware smoothing. This is conceptually aligned with serving-time leakage reduction, but may not be identical to every idealized abstraction discussed in the paper.

14. Reproducibility Notes
Randomness is controlled through set_global_seed(...)
Subject-level splitting is used for state classification
Multi-seed evaluation is supported for key sweeps
Attack AUC confidence intervals are computed via bootstrap over seed-level means
15. Typical Workflow for Reproducing Core Results
A minimal reproduction flow is:

Prepare WESAD files under data/WESAD
Run:
bash
python scripts/run_wesad_pipeline.py

Inspect:
baseline comparison CSVs
lambda sweep CSVs
alpha-TV-AUC CSVs
saved figures
16. Known Limitations of the Current Code Release
Some paper-level analyses may still need additional wrappers if you want exact table numbering from the manuscript
The code currently emphasizes WESAD experiments rather than the live AR/XR deployment stack
Command-line argument parsing is minimal; most settings are currently configured in src/config.py
The current mitigation implementation is closer to risk-aware reranking + smoothing than a fully separate neutral-list mixer module
