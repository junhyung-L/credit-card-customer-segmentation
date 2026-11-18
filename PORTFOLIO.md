# Credit-Card Customer Segmentation: Designing Around Missingness in Wide Tabular Data

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## Overview

This competition-style project classifies credit-card customers into segments
from monthly customer, credit, sales, billing, balance, channel, and marketing
tables. The saved notebook outputs record 2,400,000 training rows of monthly
customer records and 857 columns after merging. In this setting, missingness
was not a cleanup detail: it was a modelling decision because different
columns were absent for different reasons.

On a 20,000-row evaluation sample, a CatBoost + logistic-regression + MLP stack
reached weighted F1 0.8936 against 0.7026 for always predicting the majority
segment. The experiment summary records a private leaderboard score of 0.6251
(58th; top 25%).

## The challenge

- **Width and gaps.** Eight domain tables merge into 857 columns, and many
  columns are mostly empty for reasons that differ by table.
- **Extreme imbalance.** Segment E is 80.1% of the 2,400,000 training rows;
  A (972 rows) and B (144 rows) together are under 0.05%.
- **Scale.** 2,400,000 rows by 857 columns is heavy for iterating over many
  model families, so the comparisons were run on 20k samples.

## Approach

The workflow first joins monthly tables on month and customer ID. It then
treats missingness by range rather than applying one global fill rule: columns
with at least 50% missingness become removal candidates; the 20–50% range is
considered for predictive imputation; lower-missing categorical fields can
retain an `Unknown` category. The repository retains scripts for missingness
analysis and a portable preprocessing baseline.

Model selection was deliberately comparative. XGBoost and CatBoost served as
tree-based baselines alongside logistic regression, random forest, DNN/MLP/CNN,
TabNet, and stacking. CatBoost was the strongest recorded single model, and a
CatBoost + logistic-regression + MLP stack tested whether differently shaped
errors could improve the final classification.

## Decisions

**1. How to treat missing values.**
Options: one global fill (mean/mode) for every column; drop every incomplete
column; or treat columns by missing-rate band.
Choice: bands—drop at 50% or more, predictive imputation considered for
20–50%, an explicit `Unknown` category below 20%.
Why: the missing values come from different tables and causes; a global fill
hides that, and dropping all incomplete columns would discard most of the
857. In the recorded XGBoost runs, adding missingness-pattern handling moved
the public score from 0.607 (baseline) to 0.625.
Cost: the 50% and 20% cut-offs are rules of thumb and were not tuned; the
multi-output random-forest imputation remained an experiment.

**2. How to compare models at this scale.**
Options: train every candidate on the full 2.4M rows; or draw a 20,000-row
training sample and a separate 20,000-row evaluation sample and compare all
models there.
Choice: the 20k samples (`random_state=42`).
Why: it made a ten-plus-model comparison, including neural and TabNet runs,
feasible in Colab.
Cost: rows were sampled, not customers, so a customer can appear in both
samples in different months; and the evaluation sample holds 11 segment-A rows
and no segment-B rows, so the comparison is decided almost entirely by E, D,
and C.

**3. Single model or stack.**
Options: ship the best single model (CatBoost 0.8893); or stack models that
make different kinds of errors.
Choice: a stack of CatBoost, logistic regression, and MLP with a
logistic-regression meta-learner (0.8936); a CatBoost meta-learner reached
0.8911.
Why: the three base models differ in inductive bias (trees, linear, neural),
which is where stacking can help.
Cost: a gain of 0.0043 weighted F1 from one sample, with no repeated draws,
for a model that is harder to train and explain.

**4. Which score to lead with.**
Options: quote the sampled F1 alone; or quote it next to a majority-class
reference and the leaderboard result.
Choice: all three, labelled.
Why: 0.89 sounds strong until it is set against 0.7026 for predicting E every
time, and the leaderboard scores (about 0.6) come from a different setting.
Cost: the headline is less flattering, but it survives a reviewer's first
question.

## Results and interpretation

| Model | Weighted F1, 20k evaluation sample |
|---|---:|
| Always predict E (reference) | 0.7026 |
| TabNet | 0.8285 |
| CatBoost | **0.8893** |
| CatBoost + LogReg + MLP stack | **0.8936** |

The small stacking gain is useful mainly as a decision signal: careful
handling of wide, incomplete tabular features mattered more than assuming a
more complex architecture would dominate.

The experiment summary also records competition public/private scores of
0.64636 (75th) and 0.6251 (58th; top 25%). Those are competition outcomes,
whereas the F1 values are sampled validation results; they should not be read
as the same metric or evaluation setting. The leaderboard metric definition is
not recorded in this repository.

![Weighted F1 comparison across retained experiments](images/model_f1_comparison.png)

*Figure 1. Weighted F1 by model on the retained 20k-sample experiments. The
axis starts at 0.82, and the stacking value of 0.8936 is a record from that
evaluation setting, not a rerun on the full source data.*

## Limitations

Raw data, submission files, and the official leaderboard export are not
retained. The 20k split is row-level rather than customer-level, and per-class
F1 and confusion matrices were not saved. The next iteration
should split by customer, record dataset version, split prevalence, per-class
F1, confusion matrices, and a submission hash in one manifest so
minority-segment errors are visible alongside weighted F1.

## Evidence

- [`reports/실험요약.md`](reports/실험요약.md)
- [`notebooks/train_eval_20k.ipynb`](notebooks/train_eval_20k.ipynb) (sampling, class counts, CatBoost and stacking-variant outputs)
- [`notebooks/preprocess_overview.ipynb`](notebooks/preprocess_overview.ipynb) (row counts and segment distribution)
- [`src/missing_mechanism_analysis.py`](src/missing_mechanism_analysis.py)
- [`docs/PROJECT_REVIEW.md`](docs/PROJECT_REVIEW.md)
