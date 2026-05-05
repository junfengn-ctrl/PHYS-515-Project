# Loan Default Risk Modeling

This project studies binary loan default prediction using applicant income, loan amount, and employment status. The analysis compares a linear probabilistic baseline, a distance-based non-linear classifier, and polynomial feature expansion with regularized logistic regression.

## Project Overview

The project is organized around three modeling questions:

1. **Linear baseline and feature interpretation**  
   Perform EDA and preprocessing, build a Logistic Regression classifier, evaluate its predictive performance, and interpret learned feature coefficients through log-odds and odds ratios.

2. **Spatial proximity and non-linear classification**  
   Evaluate k-Nearest Neighbors (kNN), explain its distance metric, tune the number of neighbors using the validation set, and study the effect of feature scaling on distance-based classification.

3. **High-dimensional feature expansion and regularization**  
   Apply second-order polynomial expansion to continuous features, compare L1/L2-regularized Logistic Regression models, and analyze how regularization affects feature weights.

## Repository Structure

```text
.
├── .gitignore
├── FinalProject.ipynb
├── README.md
├── pipeline.py
├── loan_default_prediction.csv
├── X_train_processed.csv
├── X_val_processed.csv
├── X_test_processed.csv
├── y_train_processed.csv
├── y_val_processed.csv
├── y_test_processed.csv
├── results/
│   ├── coefficients/
│   ├── figures/
│   └── tables/
```

## Data

The raw dataset is stored in `loan_default_prediction.csv`.

Columns:

| Column | Description |
|---|---|
| `loan_id` | Loan identifier; removed before modeling |
| `income` | Applicant income |
| `loan_amount` | Requested loan amount |
| `employment_status` | Employment category: `Employed` or `Unemployed` |
| `default` | Binary target variable; `1` indicates default and `0` indicates non-default |

## Preprocessing

Preprocessing is implemented in `pipeline.py`.

The pipeline performs:

1. Removal of the non-predictive identifier column `loan_id`
2. One-hot encoding of `employment_status`
3. Stratified data splitting into:
   - 70% training set
   - 15% validation set
   - 15% test set
4. Standardization of continuous features:
   - `income`
   - `loan_amount`
5. Export of processed feature and target files:
   - `X_train_processed.csv`
   - `X_val_processed.csv`
   - `X_test_processed.csv`
   - `y_train_processed.csv`
   - `y_val_processed.csv`
   - `y_test_processed.csv`

Feature scaling is fit only on the training set and then applied to the validation and test sets to avoid data leakage.

The processed CSV files are included in the repository for convenience and reproducibility. They can also be regenerated from the raw dataset by running `pipeline.py`.

## How to Run

### 1. Generate processed data

Run:

```bash
python3 pipeline.py
```

This regenerates the processed train, validation, and test CSV files.

### 2. Run the experiment notebook

Open and run:

```text
FinalProject.ipynb
```

The notebook checks whether the processed CSV files already exist. If they are missing, it runs `pipeline.py` automatically.

## Models

The notebook trains and evaluates the following models:

| Model | Purpose |
|---|---|
| Logistic Regression | Linear baseline and coefficient interpretation |
| k-Nearest Neighbors | Distance-based non-linear classification |
| Polynomial + L2 Logistic Regression | Non-linear feature expansion with ridge regularization |
| Polynomial + L1 Logistic Regression | Non-linear feature expansion with lasso regularization |

## Notebook Organization

`FinalProject.ipynb` is organized to match the report questions:

| Section | Purpose |
|---|---|
| `1(a)` | EDA, preprocessing evidence, and processed train/validation/test splits |
| `1(b)` | Logistic Regression baseline |
| `1(c)` | Feature weights, log-odds, and odds-ratio interpretation |
| `2(a)` | kNN distance-based classification and scaling sensitivity |
| `2(b)` | Hyperparameter tuning for k |
| `2(c)` | Best kNN model and comparison with Logistic Regression |
| `3(a)` | Polynomial feature expansion |
| `3(b)` | L1/L2 regularized Logistic Regression and regularization tuning |
| `3(c)` | Coefficient effects, L1 sparsity, and best regularized polynomial model |

## Main Results

The final model comparison is saved in:

```text
results/tables/model_comparison.csv
```

Current test-set results:

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|
| Poly Logistic (L2, selected C=0.03) | 0.793 | 0.756 | 0.849 | 0.800 |
| kNN (k=39) | 0.780 | 0.750 | 0.822 | 0.784 |
| Poly Logistic (L2, fixed C=1) | 0.773 | 0.747 | 0.808 | 0.776 |
| Poly Logistic (L1, fixed C=1) | 0.773 | 0.747 | 0.808 | 0.776 |
| Logistic Regression | 0.760 | 0.740 | 0.781 | 0.760 |

The strongest test-set performance is achieved by the polynomial Logistic Regression model with L2 regularization. However, the performance differences are moderate, so conclusions should be interpreted with the dataset size and validation/test split variability in mind.

## Key Output Files

### Tables

```text
results/tables/model_comparison.csv
results/tables/logistic_regression_performance.csv
results/tables/knn_tuning_results.csv
results/tables/knn_scaling_comparison.csv
results/tables/regularization_tuning_results.csv
results/tables/polynomial_feature_summary.csv
results/tables/accuracy_wilson_ci.csv
results/tables/raw_dataset_summary.csv
```

### Coefficients

```text
results/coefficients/logistic_regression_coefficients.csv
results/coefficients/l1_l2_coefficients_C1.csv
results/coefficients/regularization_coefficients_by_C.csv
results/coefficients/best_regularized_polynomial_coefficients.csv
```

### Figures

```text
results/figures/eda_feature_distributions.png
results/figures/eda_income_loan_scatter.png
results/figures/eda_default_relationships.png
results/figures/eda_correlation_heatmap.png
results/figures/knn_train_validation_f1_by_k.png
results/figures/regularization_validation_f1_by_C.png
results/figures/l1_sparsity_by_C.png
results/figures/model_comparison_test_f1.png
```

## Interpretation Highlights

- `employment_status` is the strongest predictor in the baseline Logistic Regression model.
- The coefficient for `employment_status_Unemployed` is positive, indicating higher default risk for unemployed applicants relative to employed applicants.
- kNN performance improves substantially after standardization, confirming that distance-based methods are highly sensitive to feature scale.
- Very small kNN neighborhood sizes show high variance, while larger values smooth the decision boundary.
- Polynomial expansion improves model flexibility by allowing Logistic Regression to capture non-linear relationships in the original feature space.
- L1 regularization can produce sparse solutions under stronger regularization, while L2 regularization shrinks coefficients without setting them exactly to zero.

## Notes

- `pipeline.py` is the source of truth for preprocessing.
- `FinalProject.ipynb` is the main experiment notebook.
- The processed CSV files are versioned for convenient reruns and can be regenerated at any time by running `pipeline.py`.
- The `results/` directory contains reproducible outputs used for analysis and reporting.
- Report drafts and compiled PDFs are intentionally not versioned in this repository.
