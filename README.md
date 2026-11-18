# Credit-Card Customer Segment Classification

[한국어](README.ko.md) · [Case study](PORTFOLIO.md)

Classifies credit-card customers into five segments (A–E) from 857 monthly
customer, credit, sales, billing, balance, channel, and marketing columns, many
of them largely empty, for the DACON credit-card customer segment
classification competition.

**Why it matters.** Segments decide which card products and campaigns a
customer is offered, but the classes are extremely uneven: in the 2,400,000
training rows, segment E is 80.1% and segments A and B together are under
0.05%. A model can look accurate while never finding the rare segments, and
the missing values differ in cause from column to column, so how they are
handled is itself a modelling decision.

**Result.** On a 20,000-row evaluation sample drawn from the training data, a
CatBoost + logistic-regression + MLP stack reached weighted F1 0.8936 and
CatBoost alone 0.8893, against 0.7026 for always predicting the majority
segment. The experiment summary records a private leaderboard score of 0.6251
(58th, top 25%). The two are different evaluation settings and should not be
compared directly.

Raw data, submission files, and a leaderboard export are not stored here; the
leaderboard values are as recorded in `reports/실험요약.md`.

## Problem and data boundary

The notebook outputs record 2,400,000 training rows and 600,000 test rows of
monthly customer records, with 857 columns after the monthly domain tables are
merged, and a multiclass `Segment` target. Segment counts in the training rows
are E 1,922,052, D 349,242, C 127,590, A 972, and B 144. The source files are
available from the competition page linked in `data/data` and are excluded
from the repository.

## Analysis flow

```mermaid
flowchart LR
    A[Monthly source tables<br/>not included] --> B[Merge on month and ID]
    B --> C[Missingness analysis<br/>and rule-based preprocessing]
    C --> D[Encoding and scaling]
    D --> E[Sampled model comparison]
    E --> F[XGBoost, CatBoost, NN,<br/>TabNet, and stacking]
    F --> G[Weighted F1 validation]
    G --> H[Competition submission<br/>not included]
```

## Methodology recorded in the repository

- **Data assembly:** `src/baseline_xgb.py` loads monthly tables, concatenates
  months by domain, then left-merges domains on month and `ID`.
- **Missingness work:** `src/missing_mechanism_analysis.py` calculates missing
  rates and includes rule-based filling and a multi-output random-forest
  imputation experiment.
- **Feature preparation:** the report records removal of columns with at least
  50% missingness, model-based treatment for 20–50% missingness, and an
  unknown category treatment below 20%. It also records log transformation and
  standardization.
- **Model comparison:** the 20k sampled experiments compare CatBoost,
  logistic regression, XGBoost, random forest, DNN/MLP/CNN, TabNet, and
  stacking configurations.

The decisions behind these steps and the alternatives rejected are in
[PORTFOLIO.md](PORTFOLIO.md#decisions).

## Retained results

| Experiment | F1 | Evaluation context |
|---|---:|---|
| Always predict segment E (reference) | 0.7026 | weighted F1, 20k evaluation sample (computed from its class counts) |
| Baseline XGBoost | 0.607 | public leaderboard |
| Missingness-aware XGBoost | 0.625 | public leaderboard |
| TabNet | 0.8285 | weighted F1, 20k evaluation sample |
| CatBoost | 0.8893 | weighted F1, 20k evaluation sample |
| CatBoost + Logistic Regression + MLP stacking | **0.8936** | weighted F1, 20k evaluation sample |

The summary additionally records a public score of **0.64636** (75th) and a
private score of **0.6251** (58th; top 25%). Those values are reported as
competition outcomes in the project artifact, but cannot be independently
verified from this checkout. CatBoost and TabNet values appear in the saved
notebook outputs; the stacking value is recorded in the experiment summary and
the chart below. The summary labels the TabNet value "test" and the others
"validation", but all of them were scored on the same 20,000-row sample.

![Validation F1 comparison across retained model experiments](images/model_f1_comparison.png)

*Figure. Weighted F1 on the 20k evaluation sample for the retained sampled
experiments. The axis starts at 0.82, so differences look larger than they
are; the majority-segment reference (0.7026) is off the left edge. It is not a
rerun on the full dataset.*

## What the results do and do not show

- The 20k samples are drawn from monthly rows, not grouped by customer, so the
  same customer can appear in both the training and the evaluation sample in
  different months.
- Weighted F1 is dominated by segment E. The evaluation sample contains 11
  segment-A rows and no segment-B rows, so it says almost nothing about the
  rare segments.
- The leaderboard metric definition is not recorded in this repository, which
  is one reason the sampled F1 (about 0.89) and the leaderboard scores (about
  0.6) should not be read as the same quantity.
- Per-class F1 and confusion matrices were not retained.

## How to run

The raw data is not included and original package versions were not recorded,
so the retained results cannot be reproduced from this checkout. The portable
scripts document their inputs without data:

```powershell
pip install -r requirements.txt
python src\preprocess_missing_features.py --help
python src\missing_mechanism_analysis.py --help
python src\preprocess_overview.py --help
```

With the competition files placed as described in
[research/RUN_MANIFEST.md](research/RUN_MANIFEST.md), `python
src\preprocess_missing_features.py` builds the missingness-aware baseline CSV
(columns at or above `--drop-threshold`, default 0.50, are dropped). The 20k
model comparison is preserved in `notebooks/train_eval_20k.ipynb` with its
outputs; the exported `src/train_eval_20k.py` still contains notebook shell
commands and does not run as a script.

## Repository layout

```text
.
├── src/                         # Exported Colab preprocessing/training scripts
├── notebooks/                   # Notebooks with saved outputs (source of the results)
├── reports/실험요약.md           # Retained experiment summary and scores
├── images/model_f1_comparison.png
└── data/                        # Placeholder with the data-source link
```

## Documentation

- [Portfolio case study](PORTFOLIO.md)
- [Project review](docs/PROJECT_REVIEW.md)
- [Architecture](docs/ARCHITECTURE.md)
- [CV bullets](docs/CV_BULLETS.md)
- [Run manifest](research/RUN_MANIFEST.md)
