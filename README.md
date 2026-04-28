# Credit Card Default Prediction — FinTech Project Report

## Overview

This project builds and compares machine learning classifiers to predict credit card default payments using the **UCI Default of Credit Card Clients** dataset (Taiwan, April–September 2005). The dataset contains 30,000 observations with demographic information, credit history, bill statements, and payment records.

**Target variable:** `DEFAULT` — whether a client will default on their next payment (binary: 0/1).

---

## Dataset

| Property | Value |
|---|---|
| Observations | 30,000 |
| Features | 23 (after dropping ID) |
| Class ratio | ~3.5:1 (No Default : Default) |
| Missing values | None |

### Key variables

- **LIMIT_BAL** — credit limit (NT$)
- **PAY_0 … PAY_6** — repayment status for the last 6 months (-2=paid early, -1=on time, 1–9=months late)
- **BILL_AMT1 … BILL_AMT6** — bill statement amounts
- **PAY_AMT1 … PAY_AMT6** — previous payments made
- **SEX, EDUCATION, MARRIAGE, AGE** — demographic variables

---

## Exploratory Data Analysis

### Class imbalance

The dataset is significantly imbalanced: **77.9% No Default vs 22.1% Default** (ratio 3.52:1). This required explicit handling to avoid biased classifiers.

### Key findings

- **PAY_0** (most recent payment status) is the single strongest predictor of default (correlation +0.35). Clients with even one month of delay show sharply higher default rates.
- **LIMIT_BAL** is negatively correlated with default: clients with higher credit limits are less likely to default.
- **Age** and **marital status** show weak but consistent effects: younger clients and those with "Others" marital status default slightly more often.
- Bill amounts (BILL_AMT1–6) are highly correlated with each other (~0.9), indicating multicollinearity.
- Payment amounts (PAY_AMT1–6) have low but consistent negative correlation with default.

### Correlation with DEFAULT (top variables)

| Feature | Correlation |
|---|---|
| PAY_0 | +0.347 |
| PAY_2 | +0.305 |
| PAY_3 | +0.285 |
| PAY_4 | +0.267 |
| PAY_5 | +0.236 |
| LIMIT_BAL | −0.154 |
| PAY_AMT1 | −0.073 |

---

## Methodology

### Train/Test Split

80/20 stratified split, preserving the original class ratio in both sets.

### Handling Class Imbalance

SMOTE (Synthetic Minority Over-sampling Technique) was initially applied but **discarded** after empirical evaluation — it consistently degraded ROC-AUC and F1 for both models. The final approach uses:

- `class_weight="balanced"` for Random Forest
- `scale_pos_weight=3.5` for XGBoost

This approach is statistically sounder: it corrects the learning bias without introducing synthetic data that does not reflect the real distribution.

### Hyperparameter Tuning

A two-stage search strategy was used for both models:

1. **Stage 1 — RandomizedSearchCV** (15 iterations, 3-fold Stratified CV, scoring: ROC-AUC): broad exploration of the parameter space
2. **Stage 2 — GridSearchCV** (narrow grid around Stage 1 best values, 3-fold CV): fine-grained refinement

### Feature Selection via Lasso

Lasso logistic regression (`LogisticRegressionCV`, L1 penalty) was applied to identify redundant features:

- Fitted on the **original training set** (not SMOTE-balanced) with `class_weight="balanced"`
- Regularisation strength explored across `Cs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01]`
- Features with coefficient shrunk to zero were excluded from the reduced-feature models

### Feature Engineering

Ten engineered features were constructed and evaluated:

| Feature | Description | Corr. with DEFAULT |
|---|---|---|
| DELAY_RATIO | Fraction of months with payment delay | +0.398 |
| MAX_CONSEC_DELAY | Max consecutive months with delay | +0.389 |
| AVG_DELAY_MONTHS | Average months late (delay periods only) | +0.376 |
| REPAY_RATIO_MEAN | Mean monthly repayment ratio | −0.122 |
| TOTAL_REPAY_RATIO | Total paid / total billed | −0.112 |
| BILL1_TO_LIMIT | Most recent bill / credit limit | +0.088 |
| REPAY_RATIO_MIN | Minimum monthly repayment ratio | −0.078 |
| UTIL_STD | Standard deviation of monthly utilization | −0.017 |
| UTIL_TREND | Change in utilization Sep vs Apr | −0.024 |
| BILL_SLOPE | Linear trend of bill amounts over time | +0.024 |

The three most correlated features (DELAY_RATIO, MAX_CONSEC_DELAY, AVG_DELAY_MONTHS) were found to be **highly redundant** with the original PAY_0–PAY_6 variables (correlation >0.7). Since tree-based models already capture non-linear combinations of PAY_x internally, the engineered features did not meaningfully improve performance and were not included in the final models.

---

## Results

### Final model comparison

| Model | ROC-AUC | Accuracy | F1 (Default) | Precision (Default) | Recall (Default) |
|---|---|---|---|---|---|
| RF — Full features | 0.7769 | 0.7673 | 0.5322 | 0.4792 | 0.5983 |
| RF — Lasso reduced | 0.7773 | 0.7645 | 0.5313 | 0.4745 | 0.6036 |
| XGB — Full features | 0.7816 | 0.7625 | 0.5329 | 0.4716 | 0.6127 |
| **XGB — Lasso reduced** | **0.7812** | 0.7652 | **0.5373** | 0.4761 | **0.6164** |

### Key observations

**XGBoost outperforms Random Forest** on ROC-AUC (0.7816 vs 0.7769) and Recall (0.61 vs 0.60). In a credit risk context, Recall is particularly important: a missed default (false negative) is typically more costly than a false alarm (false positive).

**Lasso reduction does not degrade performance.** XGB Lasso-reduced achieves the highest F1 (0.5373) and Recall (0.6164) among all models, while using fewer features. This makes it the preferred model for deployment: simpler, equally accurate, and more interpretable.

**Random Forest is more conservative.** It achieves higher Precision (0.479 vs 0.476) at the cost of lower Recall — appropriate if the cost of false alarms is high.

### Most important drivers of default

Both models consistently rank the following features as most predictive:

1. **PAY_0** — most recent payment status (strongest single predictor)
2. **PAY_2, PAY_3** — payment status in previous months
3. **LIMIT_BAL** — credit limit (proxy for creditworthiness)
4. **PAY_AMT1, PAY_AMT2** — actual payment amounts

Demographic variables (age, gender, education, marriage) have marginal predictive power individually, though they contribute in combination.

---

## Discussion

### Why ~0.78 ROC-AUC is the realistic ceiling

This dataset is a well-known benchmark in the literature. Published results consistently fall in the **0.76–0.82 ROC-AUC range**, with the upper end achieved through heavy feature engineering (e.g. sequence modelling of payment delays, interaction terms) or ensemble stacking. The results obtained here sit solidly within this range without model stacking or external data.

The main limiting factor is information content: the PAY_x variables already encode most of the behavioural signal, and the remaining features (bill amounts, demographics) add relatively little once payment history is accounted for.

### SMOTE vs class weighting

Initial experiments with SMOTE (sampling_strategy=0.5 and 1.0) consistently underperformed class weighting:

- SMOTE inflated the training set with synthetic minority samples that do not reflect the true data distribution
- The test set retains the real 3.5:1 imbalance, creating a distribution mismatch
- Class weighting corrects the learning bias without altering the data, leading to better generalisation

### Lasso as a diagnostic tool

Even when Lasso does not eliminate features (with weak regularisation), it is informative: the fact that all features retained non-zero coefficients under moderate L1 penalty confirms that the original feature set is well-constructed and not redundant. Under stronger regularisation, a small number of features (primarily BILL_AMT2–6 and some PAY_AMT columns) are zeroed out, with negligible impact on predictive performance.

---

## Conclusions

| Question | Answer |
|---|---|
| Is the dataset unbalanced? | Yes, 3.52:1 (No Default : Default) |
| How was imbalance handled? | `class_weight="balanced"` / `scale_pos_weight` — outperforms SMOTE |
| Best model overall? | XGBoost — Full features (ROC-AUC 0.7816) |
| Best model for deployment? | XGBoost — Lasso reduced (highest F1 and Recall, fewer features) |
| Does Lasso help? | Marginally — maintains performance with reduced complexity |
| Does feature engineering help? | No — engineered features are redundant with PAY_x originals |
| Main drivers of default? | PAY_0, PAY_2, PAY_3, LIMIT_BAL, PAY_AMT1 |

---

## Repository Structure

```
├── project.ipynb          # Full analysis notebook
├── Dataset3.csv           # Raw dataset
└── REPORT.md              # This report
```

---
