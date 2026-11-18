# Credit-Card Customer Segmentation: Designing Around Missingness in Wide Tabular Data

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## Overview

Classifies credit-card customers into segments from monthly customer, credit,
sales, billing, balance, channel, and marketing tables: 2,400,000 training
rows of monthly customer records and 857 columns after merging. Missingness is
a modelling decision here, not a cleanup detail, because different columns are
absent for different reasons.

On a 20,000-row evaluation sample, a CatBoost + logistic-regression + MLP
(multilayer perceptron) stack reaches weighted F1 0.8936 against 0.7026 for
always predicting the majority segment. Private leaderboard score: 0.6251
(58th; top 25%).

## The challenge

- **Width and gaps.** Eight domain tables merge into 857 columns, and many
  columns are mostly empty for reasons that differ by table.
- **Extreme imbalance.** Segment E is 80.1% of the 2,400,000 training rows;
  A (972 rows) and B (144 rows) together are under 0.05%.
- **Scale.** 2,400,000 rows by 857 columns is heavy for iterating over many
  model families, so comparisons run on 20k samples.

## Approach

Joins monthly tables on month and customer ID, then treats missingness by
range instead of one global fill rule: columns with at least 50% missingness
become removal candidates; the 20–50% range is considered for predictive
imputation; lower-missing categorical fields keep an `Unknown` category.
`src/missing_mechanism_analysis.py` computes missing rates and
`src/preprocess_missing_features.py` is the portable preprocessing baseline.

Compares XGBoost and CatBoost as tree baselines alongside logistic regression,
random forest, DNN/MLP/CNN, TabNet, and stacking. CatBoost is the strongest
single model; a CatBoost + logistic-regression + MLP stack tests whether
differently shaped errors improve the final classification.

## Decisions

**1. Treating missing values.**
Options: one global fill (mean/mode); drop every incomplete column; or treat
columns by missing-rate band.
Choice: bands: drop at 50% or more, predictive imputation considered for
20–50%, an explicit `Unknown` category below 20%.
Why: missing values come from different tables and causes; a global fill hides
that, and dropping all incomplete columns would discard most of the 857. In the
XGBoost runs, missingness-pattern handling moved the public score from 0.607
(baseline) to 0.625.
Cost: the 50% and 20% cut-offs are untuned rules of thumb; the multi-output
random-forest imputation stayed experimental.

**2. Comparing models at this scale.**
Options: train every candidate on the full 2.4M rows; or draw a 20,000-row
training sample and a separate 20,000-row evaluation sample.
Choice: the 20k samples (`random_state=42`).
Why: makes a ten-plus-model comparison, including neural and TabNet runs,
feasible in Colab.
Cost: row-level sampling and a near-absence of segments A and B in evaluation
(see Limitations).

**3. Single model or stack.**
Options: use the best single model (CatBoost 0.8893); or stack models that make
different kinds of errors.
Choice: CatBoost, logistic regression, and MLP with a logistic-regression
meta-learner (0.8936); a CatBoost meta-learner reached 0.8911.
Why: the three base models differ in inductive bias (trees, linear, neural),
which is where stacking can help.
Cost: a 0.0043 weighted-F1 gain from one sample, with no repeated draws, for a
model that is harder to train and explain.

**4. Which score to lead with.**
Options: the sampled F1 alone; or the sampled F1 next to a majority-class
reference and the leaderboard result.
Choice: all three, labelled.
Why: 0.89 sounds strong until set against 0.7026 for predicting E every time,
and the leaderboard scores (about 0.6) come from a different setting.
Cost: a less flattering headline.

## Results

| Model | Weighted F1, 20k evaluation sample |
|---|---:|
| Always predict E (reference) | 0.7026 |
| TabNet | 0.8285 |
| CatBoost | **0.8893** |
| CatBoost + LogReg + MLP stack | **0.8936** |

The small stacking gain is mainly a decision signal: careful handling of wide,
incomplete tabular features mattered more than a more complex architecture,
and TabNet trails the tree-based models on this data.

Competition scores from the experiment summary: public 0.64636 (75th), private
0.6251 (58th; top 25%).

![Weighted F1 comparison across retained experiments](images/model_f1_comparison.png)

*Figure 1. Weighted F1 by model on the 20k-sample experiments. The axis starts
at 0.82.*

## Limitations

- Sampled F1 and leaderboard scores are different settings; the leaderboard metric definition is not recorded.
- Raw data, submission files, and the leaderboard export are not included.
- The 20k samples are row-level, not customer-level, so a customer can appear in both samples in different months.
- The evaluation sample holds 11 segment-A rows and no segment-B rows, so E, D, and C decide the comparison; per-class F1 and confusion matrices were not saved.
- The stacking value of 0.8936 comes from the 20k evaluation setting, not a rerun on the full source data.

## Next iteration

Split by customer, and record dataset version, split prevalence, per-class F1,
confusion matrices, and a submission hash in one manifest so minority-segment
errors are visible alongside weighted F1.

## Evidence

- [`reports/실험요약.md`](reports/실험요약.md)
- [`notebooks/train_eval_20k.ipynb`](notebooks/train_eval_20k.ipynb) (sampling, class counts, CatBoost and stacking-variant outputs)
- [`notebooks/preprocess_overview.ipynb`](notebooks/preprocess_overview.ipynb) (row counts and segment distribution)
- [`src/missing_mechanism_analysis.py`](src/missing_mechanism_analysis.py)
- [`src/preprocess_missing_features.py`](src/preprocess_missing_features.py)
- [`docs/PROJECT_REVIEW.md`](docs/PROJECT_REVIEW.md)
