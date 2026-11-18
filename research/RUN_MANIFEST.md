# Run Manifest

## Refactored paths

`src/project_config.py` replaces hard-coded Colab and local-machine paths in
the core baseline and 20k evaluation scripts. Set the data root before running:

```powershell
$env:CCS_DATA_DIR = 'D:\path\to\credit-card-data'
$env:CCS_ARTIFACT_DIR = 'D:\path\to\experiment-artifacts'
python src\baseline_xgb.py
```

`src\train_eval_20k.py` has resolved paths but still contains notebook shell
commands (for example `pip install rtdl -q`), so it does not parse as a Python
script. Use `notebooks\train_eval_20k.ipynb`, which also holds the saved
outputs, for the 20k comparison.

`CCS_DATA_DIR` must reproduce the legacy directory layout used by the original
scripts. Optional overrides are available for `CCS_TRAIN_CSV`, `CCS_TEST_CSV`,
`CCS_TRAIN_SAMPLE_CSV`, and `CCS_EVALUATION_CSV`.

## Result-preservation boundary

The refactor changes only file and output path resolution in
`baseline_xgb.py` and `train_eval_20k.py`; model types, seeds, data sampling,
preprocessing expressions, and evaluation logic are unchanged. Raw data is
unavailable, so no training or numerical equivalence check was run.

## Environment

The 20k comparison (`notebooks/train_eval_20k.ipynb`) ran on Google Colab with
Python 3.11. `requirements.txt` pins the versions its pip output records:
numpy 1.24.3 (installed in the notebook over Colab's 2.0.2), pandas 2.2.2,
scipy 1.14.1, matplotlib 3.10.0, xgboost 2.1.4, catboost 1.2.8, lightgbm 4.5.0,
and torch 2.0.1 (explicit install). scikit-learn, seaborn, and pyarrow were not
recorded and are pinned to the 2026-11 review environment; pytorch-tabnet,
missingno, and dask are unpinned. Other notebooks used different runtimes: the
missing-feature notebooks installed xgboost 3.0.0 on Colab, and the TabNet
notebooks ran on a local Python 3.12.4 Anaconda environment (numpy 1.26.4,
rtdl 0.0.13). The pinned set was not install-tested as a whole.

## Legacy exports

Several `src/` files are direct notebook exports with invalid notebook-shell
commands or damaged text encoding. They remain as evidence and are not claimed
to be portable runnable modules. Normalize them from an authoritative UTF-8
source before consolidating additional experiments.
