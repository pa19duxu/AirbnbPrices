# Predicting Airbnb Prices in a City — London

Seminar paper project for the course *Big Data and Smart Cities* (Universität Leipzig,
Chair of Sustainable Real Estate and Urban Development, Prof. Melanie Krause, Ph.D.).

**Authors:** Steven Wagner, Felix König, Eliano Grahl

## What this project does

The project investigates how accurately nightly Airbnb prices in London can be predicted
from observable listing characteristics, and which characteristics drive them. On 45,608
listings with 51 features it combines

- **descriptive and inferential statistics** (Welch t-test on the Superhost premium,
  hedonic OLS regression with k−1 coding, Breusch-Pagan test, HC3-robust inference,
  centered VIFs),
- a **machine-learning comparison** (XGBoost, Random Forest, Extra Trees, MLP, Ridge,
  plus a naive mean baseline) under a leakage-free protocol: 5-fold cross-validation,
  Optuna hyperparameter tuning on a separate split, per-fold scaling and Duan smearing
  back-transformation, and
- **explainable AI**: exact SHAP decomposition of the OLS model, Tree-SHAP for XGBoost,
  gain importance, PDP/ICE plots, and SHAP waterfalls for individual listings.

## Key results

| Model | RMSE ($) | R² (out of sample) |
|---|---|---|
| XGBoost | 68.53 ± 1.86 | 0.726 |
| Extra Trees | 71.39 ± 1.88 | 0.703 |
| Random Forest | 72.05 ± 1.84 | 0.697 |
| Neural network (MLP) | 73.92 ± 1.79 | 0.681 |
| Ridge (linear benchmark) | 81.76 ± 2.12 | 0.610 |
| Mean baseline | 130.96 ± 1.72 | 0.000 |

Accommodation type, size, and location dominate the price; the Superhost badge is
associated with a conditional premium of about 12%. The hedonic OLS explains 74.4% of
the variance in log prices (in sample).

## Repository structure

```
README.md                      this file
Code/
  Combined_Analysis.ipynb      full analysis pipeline (data preparation → statistics →
                               ML comparison → XAI → evaluation)
  requirements.txt             pinned package versions (Python 3.14)
  data.csv                     raw data — not in git, see below
```

## Data

The raw data are the public Inside Airbnb snapshot of London from **September 16, 2025**
(96,871 listings, 79 columns). The file is too large for git:

1. Download the detailed `listings.csv.gz` for London from
   <https://insideairbnb.com/get-the-data/>
2. Unpack it and save it as `Code/data.csv`.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Code/requirements.txt
jupyter notebook Code/Combined_Analysis.ipynb   # then: Run All
```

A full run takes roughly 20–40 minutes (Optuna tuning and 5-fold cross-validation of
five models). All random components are seeded (`RANDOM_STATE = 42`), so the reported
numbers reproduce exactly.
