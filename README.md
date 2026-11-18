# Credit-Card Customer Segment Classification

Multiclass model that assigns credit-card customers to five segments (A–E) from
857 monthly customer, credit, sales, billing, balance, channel, and marketing
columns, built for DACON's 신용카드 고객 세그먼트 분류 AI 경진대회 (Credit-Card
Customer Segment Classification AI Competition).

**Problem.** Segments decide which card products and campaigns a customer is
offered, but classes are extremely uneven: in 2,400,000 training rows, segment
E is 80.1% and segments A and B together are under 0.05%. Many columns are
mostly empty, for reasons that differ by table.

**Result.** DACON private leaderboard score 0.6251 (58th, top 25%). On a 20k
internal sample, CatBoost + LR + MLP stacking reached weighted F1 0.8936 vs
0.7026 for always predicting E.

**Role.** I wrote the code, analysis, and documentation in this repository.

**Scope.** Internal F1 and the leaderboard score use different data and
scoring, and samples are drawn by row, not customer, so one customer can be on
both sides. Data not included; `notebooks/train_eval_20k.ipynb` keeps the 20k
comparison with its outputs.

[한국어](README.ko.md) · [Case study](PORTFOLIO.md) · [Experiment summary](reports/실험요약.md)

## Data

2,400,000 training and 600,000 test rows of monthly customer records; 857
columns after merging the monthly domain tables; multiclass `Segment` target.
Training segment counts: E 1,922,052, D 349,242, C 127,590, A 972, B 144. Raw
data is not included; `data/data` links to the competition data page.

## Approach

```mermaid
flowchart LR
    A[Monthly source tables<br/>not included] --> B[Merge on month and ID]
    B --> C[Missingness analysis<br/>and rule-based preprocessing]
    C --> D[Encoding and scaling]
    D --> E[Sampled model comparison]
    E --> F[XGBoost, CatBoost, NN,<br/>TabNet, and stacking]
    F --> G[Weighted F1 validation]
    G --> H[Competition submission]
```

- **Data assembly:** `src/baseline_xgb.py` concatenates monthly tables by
  domain, then left-merges domains on month and `ID`.
- **Missingness:** `src/missing_mechanism_analysis.py` computes missing rates
  and tests rule-based filling and multi-output random-forest imputation.
- **Feature preparation:** drops columns with at least 50% missingness, applies
  model-based treatment for 20–50%, and an `Unknown` category below 20%; log
  transformation and standardization.
- **Model comparison:** on 20k samples, compares CatBoost, logistic
  regression (LR), XGBoost, random forest, DNN/MLP/CNN (deep, multilayer, and
  convolutional neural networks), TabNet, and stacking configurations.

Design choices and rejected alternatives: [PORTFOLIO.md](PORTFOLIO.md#key-decisions).

## Results

**DACON leaderboard**

| Submission | Public | Private |
|---|---:|---:|
| Baseline XGBoost | 0.607 | – |
| Missingness-aware XGBoost | 0.625 | – |
| Final submission | 0.6464 (75th) | **0.6251** (58th, top 25%) |

**Internal 20k sample (weighted F1)**

| Model | Weighted F1 |
|---|---:|
| Always predict segment E (reference) | 0.7026 |
| TabNet | 0.8285 |
| CatBoost | 0.8893 |
| CatBoost + LR + MLP stacking (LR meta-learner) | **0.8936** |

All internal values come from the same 20,000-row evaluation sample drawn from
the training data. The two tables use different data and scoring, so their
numbers are not comparable.

![Weighted F1 comparison across model experiments](images/model_f1_comparison.png)

*Figure. Weighted F1 on the 20k evaluation sample. The axis starts at 0.82, so
differences look larger than they are; the majority-segment reference (0.7026)
is off the left edge.*

## How to run

```powershell
pip install -r requirements.txt
python src\preprocess_missing_features.py
```

With the competition files placed as described in
[research/RUN_MANIFEST.md](research/RUN_MANIFEST.md), this builds the
missingness-aware baseline CSV (columns at or above `--drop-threshold`, default
0.50, are dropped). The 20k model comparison is in
`notebooks/train_eval_20k.ipynb` with its outputs (run the notebook; its raw
`.py` export under `legacy/` does not parse). That notebook ran on Google
Colab (Python 3.11); `requirements.txt` pins the package versions its pip output
records and marks the few it does not (TabNet experiments ran separately on a
local Python 3.12.4 environment).

## Limitations

- The internal weighted F1 (0.8936) and the leaderboard score (0.6251) come from different data and scoring; only the leaderboard score is an external benchmark.
- Samples are drawn by monthly row, not by customer, so a customer can appear in both the training and evaluation samples.
- Weighted F1 is dominated by segment E; the evaluation sample has 11 segment-A rows and no segment-B rows, and per-class F1 was not computed.
- The stacking gain over CatBoost (+0.0043) comes from one sample with no repeated draws.

## Repository layout

```text
.
├── src/                         # Exported Colab preprocessing/training scripts (all compile)
├── notebooks/                   # Notebooks with saved outputs (source of the results)
├── legacy/notebook_exports/     # Raw .py exports that do not parse; reference only
├── reports/실험요약.md           # Experiment summary and scores
├── images/model_f1_comparison.png
└── data/                        # Placeholder with the data-source link
```

## Documentation

- [Portfolio case study](PORTFOLIO.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Run manifest](research/RUN_MANIFEST.md)

## License

No open-source license: the code and documents are public for review; reuse requires permission.
