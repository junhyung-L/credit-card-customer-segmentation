# Credit-Card Customer Segment Classification

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## At a glance

Multiclass model that assigns credit-card customers to five segments (A–E) from 857 monthly columns, built for DACON's 신용카드 고객 세그먼트 분류 AI 경진대회 (Credit-Card Customer Segment Classification AI Competition).

**Problem.** Eight monthly domain tables merge into 857 columns, many mostly empty for reasons that differ by table. Segment E is 80.1% of the 2,400,000 training rows; A (972 rows) and B (144 rows) together are under 0.05%.

**Result.** DACON private leaderboard score 0.6251 (58th, top 25%). On a 20k internal sample, CatBoost + LR + MLP stacking reached weighted F1 0.8936 vs 0.7026 for always predicting E.

[README](README.md) · [Experiment summary](reports/실험요약.md) · [20k comparison notebook](notebooks/train_eval_20k.ipynb)

**Approach.** Monthly tables are joined on month and customer ID. Missingness is treated by range instead of one global fill rule. XGBoost and CatBoost serve as tree baselines alongside logistic regression (LR), random forest, DNN/MLP/CNN, TabNet, and stacking. At 2,400,000 rows by 857 columns, model comparisons run on 20k samples.

## Key decisions

**1. Treating missing values.**
Options: one global fill (mean/mode); drop every incomplete column; or treat columns by missing-rate band.
Choice: bands: drop at 50% or more, predictive imputation for 20–50%, an explicit `Unknown` category below 20%.
Why: missing values come from different tables and causes; a global fill hides that, and dropping all incomplete columns would discard most of the 857. In the XGBoost runs, missingness-pattern handling moved the public score from 0.607 (baseline) to 0.625.
Cost: the 50% and 20% cut-offs are untuned rules of thumb; the multi-output random-forest imputation stayed experimental.

**2. Comparing models at this scale.**
Options: train every candidate on the full 2.4M rows; or draw a 20,000-row training sample and a separate 20,000-row evaluation sample.
Choice: the 20k samples (`random_state=42`).
Why: makes a ten-plus-model comparison, including neural and TabNet runs, feasible in Colab.
Cost: row-level sampling and a near-absence of segments A and B in evaluation (see Limitations).

**3. Single model or stack.**
Options: use the best single model (CatBoost 0.8893); or stack models that make different kinds of errors.
Choice: CatBoost, LR, and MLP with an LR meta-learner (0.8936); a CatBoost meta-learner reached 0.8911.
Why: the three base models differ in inductive bias (trees, linear, neural), which is where stacking can help.
Cost: a harder model to train and explain for a small gain.

**4. Which score to lead with.**
Options: the internal F1 alone; or the leaderboard score first, then the internal F1 next to a majority-class reference.
Choice: leaderboard first, internal sample second, each labeled.
Why: 0.89 sounds strong until set against 0.7026 for predicting E every time, and only the leaderboard is an external benchmark.
Cost: a less flattering headline.

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
| Always predict E (reference) | 0.7026 |
| TabNet | 0.8285 |
| CatBoost | 0.8893 |
| CatBoost + LR + MLP stacking (LR meta-learner) | **0.8936** |

The stacking gain (+0.0043) is small. Missing-value handling mattered more to performance than model architecture, and TabNet trails the tree-based models on this data.

![Weighted F1 comparison across model experiments](images/model_f1_comparison.png)

*Figure 1. Weighted F1 by model on the 20k evaluation sample. The axis starts at 0.82.*

## Limitations

- The internal weighted F1 (0.8936) and the leaderboard score (0.6251) come from different data and scoring; only the leaderboard score is an external benchmark.
- The 20k samples are row-level, not customer-level, so a customer can appear in both samples in different months.
- The evaluation sample holds 11 segment-A rows and no segment-B rows, so E, D, and C decide the comparison; per-class F1 was not computed.
- The stacking gain over CatBoost (+0.0043) comes from one sample with no repeated draws.

## Next steps

- Split by customer and repeat the sample draws to put an interval on the stacking gain.
- Report per-class F1 and confusion matrices so minority-segment errors are visible alongside weighted F1.
- Stratified K-fold cross-validation and Optuna hyperparameter search, as listed in the experiment summary.

## Evidence

- [`reports/실험요약.md`](reports/실험요약.md) (leaderboard scores and all experiment rows)
- [`notebooks/train_eval_20k.ipynb`](notebooks/train_eval_20k.ipynb) (sampling, class counts, CatBoost and stacking-variant outputs)
- [`notebooks/preprocess_overview.ipynb`](notebooks/preprocess_overview.ipynb) (row counts and segment distribution)
- [`src/missing_mechanism_analysis.py`](src/missing_mechanism_analysis.py)
- [`src/preprocess_missing_features.py`](src/preprocess_missing_features.py)
