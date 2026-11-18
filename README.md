# Credit-Card Customer Segment Classification

[한국어](README.ko.md) · [Case study](PORTFOLIO.md)

Classifies credit-card customers into five segments (A–E) from 857 monthly
customer, credit, sales, billing, balance, channel, and marketing columns, many
of them largely empty, for the DACON (Korean data-science competition platform)
credit-card customer segment classification competition.

**Why it matters.** Segments decide which card products and campaigns a
customer is offered, but classes are extremely uneven: in 2,400,000 training
rows, segment E is 80.1% and segments A and B together are under 0.05%. Missing
values also differ in cause from column to column, so their handling is itself
a modelling decision.

**Result.** On a 20,000-row evaluation sample drawn from the training data, a
CatBoost + logistic-regression + MLP (multilayer perceptron) stack reaches
weighted F1 0.8936 and CatBoost alone 0.8893, against 0.7026 for always
predicting the majority segment. Private leaderboard score: 0.6251 (58th, top
25%), a different evaluation setting.

## Data

2,400,000 training and 600,000 test rows of monthly customer records; 857
columns after merging the monthly domain tables; multiclass `Segment` target.
Training segment counts: E 1,922,052, D 349,242, C 127,590, A 972, B 144. Raw
data not included; see `data/data` for the competition source link.

## Approach

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

- **Data assembly:** `src/baseline_xgb.py` concatenates monthly tables by
  domain, then left-merges domains on month and `ID`.
- **Missingness:** `src/missing_mechanism_analysis.py` computes missing rates
  and tests rule-based filling and multi-output random-forest imputation.
- **Feature preparation:** drops columns with at least 50% missingness, applies
  model-based treatment for 20–50%, and an `Unknown` category below 20%; log
  transformation and standardization.
- **Model comparison:** on 20k samples, compares CatBoost, logistic
  regression, XGBoost, random forest, DNN/MLP/CNN (deep, multilayer, and
  convolutional neural networks), TabNet, and stacking configurations.

Design choices and rejected alternatives: [PORTFOLIO.md](PORTFOLIO.md#decisions).

## Results

| Experiment | F1 | Evaluation context |
|---|---:|---|
| Always predict segment E (reference) | 0.7026 | weighted F1, 20k evaluation sample (computed from its class counts) |
| Baseline XGBoost | 0.607 | public leaderboard |
| Missingness-aware XGBoost | 0.625 | public leaderboard |
| TabNet | 0.8285 | weighted F1, 20k evaluation sample |
| CatBoost | 0.8893 | weighted F1, 20k evaluation sample |
| CatBoost + Logistic Regression + MLP stacking | **0.8936** | weighted F1, 20k evaluation sample |

Competition scores from the experiment summary (`reports/실험요약.md`): public
**0.64636** (75th), private **0.6251** (58th; top 25%). CatBoost and TabNet
values are in saved notebook outputs; the stacking value is in the experiment
summary and the chart below. All sampled F1 values come from the same
20,000-row sample (the summary labels TabNet "test" and the others
"validation").

![Validation F1 comparison across retained model experiments](images/model_f1_comparison.png)

*Figure. Weighted F1 on the 20k evaluation sample. The axis starts at 0.82, so
differences look larger than they are; the majority-segment reference (0.7026)
is off the left edge.*

## How to run

The portable scripts document their inputs without data:

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
model comparison is in `notebooks/train_eval_20k.ipynb` with its outputs.

## Limitations

- Sampled F1 (about 0.89) and leaderboard scores (about 0.6) are different settings; the leaderboard metric definition is not recorded.
- Leaderboard scores come from the experiment summary; submission files and a leaderboard export are not included.
- Samples are drawn by monthly row, not by customer, so a customer can appear in both training and evaluation samples.
- Weighted F1 is dominated by segment E; the evaluation sample has 11 segment-A rows and no segment-B rows. Per-class F1 and confusion matrices were not retained.
- Sampled F1 values come from 20k samples, not a full-dataset rerun.
- Original package versions were not recorded, so results cannot be reproduced exactly.
- The exported `src/train_eval_20k.py` still contains notebook shell commands and does not run as a script.

## Repository layout

```text
.
├── src/                         # Exported Colab preprocessing/training scripts
├── notebooks/                   # Notebooks with saved outputs (source of the results)
├── reports/실험요약.md           # Experiment summary and scores
├── images/model_f1_comparison.png
└── data/                        # Placeholder with the data-source link
```

## Documentation

- [Portfolio case study](PORTFOLIO.md)
- [Project review](docs/PROJECT_REVIEW.md)
- [Architecture](docs/ARCHITECTURE.md)
- [CV bullets](docs/CV_BULLETS.md)
- [Run manifest](research/RUN_MANIFEST.md)
