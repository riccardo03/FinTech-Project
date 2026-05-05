# Credit Card Default Prediction
### FinTech Final Assignment

Binary classification of credit card default risk using Random Forest and XGBoost on the UCI Taiwan Credit Default dataset (30,000 clients, April–September 2005).

---

## Problem Statement

Predict whether a credit card client will default on their next payment, given 6 months of demographic data, payment history, bill statements, and repayment amounts.

The dataset is **imbalanced**: ~22% of clients defaulted, ~78% did not. This is explicitly addressed in the modeling pipeline.

---

## Dataset

**Source:** UCI Machine Learning Repository — *Default of Credit Card Clients* (Taiwan, 2005)

**Size:** 30,000 clients × 25 variables

| Group | Variables |
|---|---|
| Client profile | `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` |
| Payment status (Apr–Sep) | `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6` |
| Bill amounts (Apr–Sep) | `BILL_AMT1` – `BILL_AMT6` |
| Payment amounts (Apr–Sep) | `PAY_AMT1` – `PAY_AMT6` |
| Target | `DEFAULT` (1 = default, 0 = no default) |

> **Note:** `PAY_*` columns in the raw data are shifted by −1 relative to the documentation. Values are corrected (+1) during preprocessing.

---

## Project Structure

```
project.ipynb        # Main notebook (all steps self-contained)
README.md            # This file
plot_*.png           # All visualizations saved during execution
```

---

## Pipeline Overview

### 1. Exploratory Data Analysis
- Class imbalance analysis
- Default rate by demographic group (gender, education, marital status)
- Age and credit limit distributions by class
- Temporal trends in bill and payment amounts (Apr–Sep)
- Credit utilization ratio analysis
- Correlation matrix (Pearson) across all features

### 2. Feature Engineering

Features engineered from raw PAY delay columns and utilization ratios:

| Feature | Description |
|---|---|
| `PAY0_ANY_DELAY` | Binary: any delay in most recent month |
| `PAY0_SEVERE` | Binary: delay ≥ 2 months in most recent month |
| `PAY0_DULY` | Binary: paid duly in most recent month |
| `N_DELAY_MONTHS` | Count of months with any payment delay |
| `N_SEVERE_MONTHS` | Count of months with severe delay (≥ 2) |
| `EVER_SEVERE` | Binary: ever had a severe delay across all 6 months |
| `WEIGHTED_DELAY` | Recency-weighted delay score (PAY_0 weighted most) |
| `BILL1_TO_LIMIT` | Most recent bill normalized by credit limit |
| `UTIL_X_DELAY` | Interaction: utilization ratio × PAY_0 |
| `REPAY_RATIO_1` | Fraction of September bill actually repaid |

### 3. Lasso Feature Selection

`LogisticRegressionCV` with L1 penalty (SAGA solver) trained on standardized features to identify and zero out uninformative variables before tree-based modeling.

- Optimal C selected via 3-fold CV optimizing ROC-AUC
- Features with zero coefficients excluded from the reduced feature set
- Result: 32 of 33 features retained (`BILL_AMT1` zeroed — redundant with `BILL1_TO_LIMIT`)

### 4. Modeling

Both models trained on full and Lasso-reduced feature sets. Class imbalance handled differently per model:

| Model | Imbalance Strategy |
|---|---|
| Random Forest | `class_weight="balanced"` |
| XGBoost | `scale_pos_weight` = ratio of negative to positive class |

**Hyperparameter tuning:** Two-stage approach
- Stage 1: `RandomizedSearchCV` (50 iterations, 3-fold stratified CV, scoring = ROC-AUC)
- Stage 2: `GridSearchCV` around Stage 1 best values *(available, commented out for speed)*

### 5. Evaluation & Comparison

Models evaluated on a stratified 80/20 train-test split using:
- ROC-AUC
- Accuracy
- F1, Precision, Recall on the minority (Default) class
- Confusion matrix
- Precision-Recall curve with optimal threshold analysis
- Feature importance comparison (normalized gain)

---

## Results

| Model | ROC-AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Random Forest | 0.7777 | 0.7782 | 0.5421 | 0.4987 | 0.5938 |
| **XGBoost** | **0.7815** | 0.7567 | 0.5324 | 0.4630 | **0.6262** |

**XGBoost** achieves the best ROC-AUC (0.7815) and recall (0.626), meaning it catches more actual defaulters. Random Forest is more conservative — higher precision, lower recall.

**Top predictors** (XGBoost feature importance by gain):

1. `EVER_SEVERE` — whether the client ever had a 2+ month delay
2. `WEIGHTED_DELAY` — recency-weighted delay score
3. `N_SEVERE_MONTHS` — count of severely late months
4. `PAY_0` — most recent payment status
5. `PAY0_SEVERE` — severe delay flag for most recent month

---

## Threshold Analysis (XGBoost)

| Threshold | F1 | Recall | Precision |
|---|---|---|---|
| 0.50 (default) | 0.5348 | 0.608 | 0.477 |
| 0.529 (max-F1) | 0.5426 | 0.576 | 0.513 |
| 0.404 (recall ≥ 0.75) | 0.4925 | 0.751 | 0.366 |

The optimal threshold depends on the business cost of false negatives vs false positives.

---

## Requirements

```
python >= 3.10
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
category_encoders
```

Install dependencies:
```bash
pip install xgboost imbalanced-learn category_encoders
```

> The notebook auto-detects GPU availability and configures XGBoost accordingly (`device='cuda'` if available, otherwise `cpu`).

---

## How to Run

1. Download the dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) and place it in the same directory as the notebook
2. Open `project.ipynb` in Jupyter or Google Colab
3. Run all cells in order

All plots are saved automatically as `plot_*.png` in the working directory.
