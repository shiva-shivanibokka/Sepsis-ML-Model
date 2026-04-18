# Sepsis Early Prediction — ICU Machine Learning Project

This project builds machine learning models to predict whether a patient in the intensive care unit (ICU) will develop sepsis, using hourly vital sign and lab measurements from the PhysioNet/CinC Challenge 2019 dataset. Two models are built and compared: Random Forest and XGBoost.

**Note:** This is an introductory project intended to help students understand the core concepts and workflow of applied machine learning — data loading, exploratory analysis, preprocessing, feature selection, model training, hyperparameter tuning, and evaluation. The results are modest by design; the focus is on understanding the process, not achieving state-of-the-art performance.

The project is structured as four sequential Jupyter notebooks, designed for students learning applied machine learning. Every step is explained in plain language alongside the code.

---

## Table of Contents

- [Background](#background)
- [What is Sepsis?](#what-is-sepsis)
- [The Dataset](#the-dataset)
- [Class Imbalance](#class-imbalance)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Notebook 1 — Data Loading and EDA](#notebook-1--data-loading-and-eda)
- [Notebook 2 — Preprocessing and Feature Selection](#notebook-2--preprocessing-and-feature-selection)
- [Notebook 3 — Random Forest](#notebook-3--random-forest)
- [Notebook 4 — XGBoost and Model Comparison](#notebook-4--xgboost-and-model-comparison)
- [Key Design Decisions](#key-design-decisions)
- [Libraries Used](#libraries-used)
- [Results Summary](#results-summary)

---

## Background

Sepsis is one of the leading causes of death in hospital settings worldwide. It occurs when the body's response to an infection spirals out of control and begins damaging its own organs. Early treatment dramatically improves survival — every hour of delay in treating sepsis increases mortality risk.

ICU patients have their vital signs and lab values measured and recorded every hour. This creates a rich time-series of clinical data. The goal of this project is to build a machine learning model that can identify, from these measurements, which patients are likely to develop sepsis — before it becomes life-threatening.

---

## What is Sepsis?

Sepsis is not a disease itself — it is the body's dysregulated response to infection. When the immune system fights an infection, it normally produces a controlled inflammatory response. In sepsis, this response becomes overwhelming and systemic, damaging organs including the kidneys, liver, lungs, and heart.

Clinically, sepsis is identified by a combination of abnormal vital signs (elevated heart rate, high respiratory rate, abnormal temperature) and laboratory values (elevated white blood cell count, elevated lactate, abnormal organ function markers). These are exactly the features that appear in this dataset.

**Why early prediction matters:** Sepsis progresses rapidly. A patient who is septic but not yet recognised as such may receive inadequate treatment for hours. Machine learning models that can flag high-risk patients earlier — even before the clinical definition of sepsis is met — can directly save lives.

---

## The Dataset

**Source:** PhysioNet/CinC Challenge 2019 — Early Prediction of Sepsis from Clinical Data  
**Format:** One row per hour per patient (time-series)

The raw dataset contains hourly ICU measurements for thousands of patients. Each row represents one hour of monitoring for one patient.

### Features

The dataset contains two types of columns:

**Vital signs (8 features):** Measurements taken continuously at the bedside.

| Column | Description |
|---|---|
| `HR` | Heart rate (beats per minute) |
| `O2Sat` | Oxygen saturation (%) |
| `Temp` | Body temperature (°C) |
| `SBP` | Systolic blood pressure (mmHg) |
| `MAP` | Mean arterial pressure (mmHg) |
| `DBP` | Diastolic blood pressure (mmHg) |
| `Resp` | Respiratory rate (breaths per minute) |
| `EtCO2` | End-tidal carbon dioxide |

**Lab values (26 features):** Blood tests ordered periodically, not every hour.

| Column | Description |
|---|---|
| `BaseExcess`, `HCO3`, `pH`, `PaCO2` | Blood gas — acid-base balance |
| `FiO2`, `SaO2` | Oxygenation |
| `AST`, `Bilirubin_direct`, `Bilirubin_total` | Liver function |
| `BUN`, `Creatinine` | Kidney function |
| `Alkalinephos`, `Calcium`, `Chloride` | Electrolytes and minerals |
| `Glucose`, `Lactate` | Metabolism and tissue perfusion |
| `Magnesium`, `Phosphate`, `Potassium` | Electrolytes |
| `TroponinI` | Cardiac damage marker |
| `Hct`, `Hgb`, `PTT`, `WBC`, `Fibrinogen`, `Platelets` | Blood count and clotting |

**Demographics (2 features):**

| Column | Description |
|---|---|
| `Age` | Patient age in years |
| `Gender` | 0 = Female, 1 = Male |

**Target:**

| Column | Description |
|---|---|
| `SepsisLabel` | 0 = no sepsis at this hour, 1 = sepsis onset at this hour |

**Columns dropped before modelling:**

| Column | Reason |
|---|---|
| `Unit1`, `Unit2` | Administrative ICU unit identifiers — not clinical signals |
| `HospAdmTime` | Time between hospital admission and ICU transfer — metadata |
| `ICULOS` | ICU length of stay — changes every hour, is metadata |

---

## Class Imbalance

Sepsis is a rare event. Only a small fraction of ICU patients develop it. This **class imbalance** is one of the most important characteristics of this dataset and drives several modelling decisions.

**Why accuracy is a misleading metric here:** If 95% of patients are non-sepsis, a model that always predicts Non-Sepsis achieves 95% accuracy — while catching zero actual sepsis cases. This is clinically useless.

**What we do instead:**
- Use **F1 score** as the primary evaluation metric — it balances precision (avoiding false alarms) and recall (catching real cases)
- Use `class_weight='balanced'` in Random Forest — automatically penalises missing a sepsis patient more than a false alarm
- Use `scale_pos_weight` in XGBoost — the equivalent parameter for that library
- Use **stratified sampling** to ensure the sepsis rate in our 5,000-patient sample matches the full dataset

---

## Project Structure

```
Sepsis-ML-Model/
│
├── 01_eda_loading.ipynb        ← data loading, feature engineering, EDA
├── 02_preprocessing.ipynb      ← feature selection, train/test split
├── 03_random_forest.ipynb      ← Random Forest classifier
└── 04_xgboost.ipynb            ← XGBoost classifier + model comparison
```

The notebooks must be run **in order**. Each one saves its outputs to Google Drive and the next one reads them.

```
01_eda_loading.ipynb
    └── saves: patient_data.csv

02_preprocessing.ipynb
    └── reads: patient_data.csv
    └── saves: X_train.csv, X_test.csv, y_train.csv, y_test.csv

03_random_forest.ipynb
    └── reads: X_train.csv, X_test.csv, y_train.csv, y_test.csv

04_xgboost.ipynb
    └── reads: X_train.csv, X_test.csv, y_train.csv, y_test.csv
```

---

## How to Run

### Requirements

- A Google account with Google Drive
- Google Colab (free — runs in your browser)
- The raw dataset `sepsis_dataset.csv` uploaded to a folder on your Google Drive

### Setup Steps

1. Create a folder on your Google Drive, for example: `My Drive/sepsis_data/`
2. Upload `sepsis_dataset.csv` into that folder
3. Open each notebook in Google Colab
4. In **each notebook**, find the configuration cell and update `data_dir` to match your folder path:

```python
data_dir = Path('/content/drive/MyDrive/sepsis_data')
```

5. Run the notebooks in order: `01` → `02` → `03` → `04`

---

## Notebook 1 — Data Loading and EDA

**File:** `01_eda_loading.ipynb`  
**Input:** `sepsis_dataset.csv`  
**Output:** `patient_data.csv`

### Feature Engineering

The raw dataset is a time-series with one row per hour per patient. Machine learning models expect one row per patient. This notebook converts the time-series into a flat table using `groupby().agg()`.

For each patient, we compute:
- `HR_min`, `HR_max`, `HR_mean` — minimum, maximum, and mean heart rate across all ICU hours
- The same three aggregations for all 33 vital sign and lab columns
- `Age` and `Gender` — taken as the first value (same every hour)
- `Sepsis` — 1 if the `SepsisLabel` was ever 1 across any hour, else 0

This produces one row per patient with `33 × 3 + 2 = 101` feature columns plus the Sepsis target.

The multi-level column names from `.agg()` (e.g. `('HR', 'min')`) are flattened into single strings (`'HR_min'`).

### EDA Plots

| Plot | What it shows |
|---|---|
| Patient Count by Class | Bar chart comparing total non-sepsis vs sepsis patients |
| Class Proportion | Pie chart of the sepsis/non-sepsis split |
| Features by Missing Value % | Bar chart showing null rate per feature, with 90% and 50% threshold lines |
| Feature Distributions | Grid of histograms for every mean-aggregated feature, sepsis vs non-sepsis overlaid |
| Box Plots by Sepsis Status | Grid of box plots comparing IQR per feature between the two classes |
| Pearson Correlation Heatmap | Correlation matrix of all mean features (lower triangle only) |
| Age Distribution by Sepsis Status | Overlaid histograms of patient age |
| Gender vs Sepsis | Grouped bar chart of Male/Female split by sepsis status |

---

## Notebook 2 — Preprocessing and Feature Selection

**File:** `02_preprocessing.ipynb`  
**Input:** `patient_data.csv`  
**Output:** `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

### Data Leakage

All preprocessing steps are **fit on the training set only**. The scaler, imputer, and variance threshold all see only training data when being configured. They are then applied to the test set using those training-derived parameters. This ensures the test set is never seen — directly or indirectly — during model preparation.

### Stratified Sampling

We work with a subset of 5,000 patients to keep the RFE step fast enough to run in Colab. The sample is **stratified** — the sepsis rate in the sample is the same as in the full dataset. Two sampling methods are shown:

- **Method A:** A custom `sample_group()` function passed to `groupby().apply()` — makes the proportional logic explicit
- **Method B:** `train_test_split(test_size=5000, stratify=y)` — more concise, same result

### Feature Reduction Pipeline

| Step | Method | Threshold |
|---|---|---|
| 1 | Zero-variance removal | variance == 0 |
| 2 | High-null removal | > 90% missing |
| 3 | Imputation | median strategy |
| 4 | VarianceThreshold | 0.01 |
| 5 | StandardScaler | mean=0, std=1 |
| 6 | RFE | best k by F1 sweep |

**Zero-variance removal:** Any feature with the same value for every training patient is dropped — it provides no discriminative information.

**High-null removal:** Features where more than 90% of training patients have no value are dropped. Imputing 90% of a column is not meaningful.

**Imputation:** Remaining missing values are filled with the training median using `SimpleImputer(strategy='median')`. The median is more robust than the mean when clinical outliers are present. Imputation must happen before VarianceThreshold, which cannot handle NaN values.

**VarianceThreshold:** Removes features with near-zero variance after scaling. `get_support()` returns a True/False array indicating which features to keep.

**StandardScaler:** Rescales every feature to mean=0, std=1. This prevents features with large numeric ranges (e.g. glucose) from dominating over features with small ranges (e.g. pH).

**RFE sweep:** Starting from N//4 features and halving each step, we test several values of `k`. For each `k`, a lightweight Random Forest ranks the features, selects the top `k`, and a slightly larger Random Forest evaluates the result. The F1 score is plotted against `k` and the best value is used for the final RFE. The plot includes an annotated red dashed line at the best `k`.

---

## Notebook 3 — Random Forest

**File:** `03_random_forest.ipynb`  
**Input:** `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

### How Random Forest Works

A Random Forest builds many decision trees independently. Each tree is trained on a random sample of the training data (bootstrap sample) and considers a random subset of features at each split. The final prediction is the majority vote across all trees.

The randomness makes each tree slightly different, which means the ensemble's errors are uncorrelated — they cancel out rather than compound. This is what makes Random Forest more robust than a single decision tree.

### Hyperparameter Tuning — GridSearchCV

GridSearchCV tests every combination of the parameter grid using 5-fold stratified cross-validation, optimising for F1 score.

| Parameter | Values | Description |
|---|---|---|
| `n_estimators` | 100, 200 | Number of trees |
| `max_depth` | None, 10, 20 | Maximum tree depth |
| `min_samples_split` | 2, 5 | Minimum samples to split a node |
| `max_features` | 'sqrt', 'log2' | Features considered at each split |

Total combinations: 2 × 3 × 2 × 2 = 24. Total fits: 24 × 5 = 120.

### Evaluation Outputs

- **Classification report:** Precision, recall, F1 and support per class
- **Confusion matrix:** TN, FP, FN, TP with human-readable labels (e.g. "False Negatives (missed sepsis)")
- **ROC curve:** True positive rate vs false positive rate, with AUC
- **Feature importances:** Horizontal bar chart of top 20 features by mean decrease in impurity

---

## Notebook 4 — XGBoost and Model Comparison

**File:** `04_xgboost.ipynb`  
**Input:** `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

### How XGBoost Works

XGBoost uses **boosting** — it builds trees sequentially rather than independently. Each new tree focuses on the patients the previous ensemble got wrong. The model learns by minimising a loss function, and each tree is a small correction to the current predictions.

XGBoost is generally faster and more accurate than Random Forest on tabular data, though it has more hyperparameters to tune.

### Class Imbalance — `scale_pos_weight`

XGBoost does not have a `class_weight='balanced'` parameter. Instead, `scale_pos_weight = count(non-sepsis) / count(sepsis)` is used. This tells the model how much more to penalise missing a sepsis patient compared to a false alarm. The value is computed directly from the training set counts.

### Hyperparameter Tuning — GridSearchCV

Same setup as Notebook 3 (5-fold stratified CV, F1 scoring) to keep the comparison fair.

| Parameter | Values | Description |
|---|---|---|
| `n_estimators` | 100, 200 | Number of boosting rounds |
| `max_depth` | 3, 5, 7 | Tree depth |
| `learning_rate` | 0.05, 0.1 | Contribution of each new tree |
| `subsample` | 0.8, 1.0 | Fraction of patients used per tree |

### Model Comparison

The final section of Notebook 4 re-runs the best Random Forest (using the best parameters from Notebook 3) and compares both models on four metrics:

| Metric | What it measures |
|---|---|
| F1 Score | Harmonic mean of precision and recall — primary metric |
| ROC-AUC | Area under the ROC curve — overall discriminative ability |
| Precision | Of all patients flagged as sepsis, how many actually had it |
| Recall | Of all patients who had sepsis, how many were correctly flagged |

The comparison is shown as:
1. A printed table with all four metrics for both models
2. A grouped bar chart (blue = Random Forest, orange = XGBoost) with values annotated
3. Overlaid ROC curves for both models on a single plot

---

## Key Design Decisions

### Why F1 score instead of accuracy?

Accuracy is misleading on imbalanced data. If 95% of patients are non-sepsis, always predicting Non-Sepsis gives 95% accuracy. F1 score balances precision (how many of our sepsis predictions are correct) and recall (how many real sepsis cases we catch). It forces the model to actually identify sepsis patients, not just optimise for the majority class.

### Why a stratified sample of 5,000?

RFE trains many models internally — one for each candidate `k` value, multiplied by the cross-validation folds. Running this on tens of thousands of patients would take many hours in Colab. The stratified 5,000-patient sample preserves the real sepsis rate, so the model sees the same class balance as the full dataset.

### Why `class_weight='balanced'` and `scale_pos_weight`?

Without these, both models would implicitly optimise for the majority class. `class_weight='balanced'` (Random Forest) and `scale_pos_weight` (XGBoost) both do the same thing in their respective APIs: increase the penalty for missing a minority class (sepsis) prediction.

### Why the same train/test split for both models?

Both Notebook 3 and Notebook 4 read the same `X_train.csv` and `X_test.csv` files. This means both models are trained on identical data and evaluated on identical data. Without this, any difference in performance could be due to different random splits rather than the model itself.

### Why is imputation done after the null-column removal step?

We first drop columns with more than 90% missing. Then we impute the remaining missing values. If we imputed first, we would be filling 90% of a column with guessed values — these would then have non-trivial variance and would pass through VarianceThreshold, introducing noise into the model. Dropping first avoids this.

---

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, manipulation, aggregation |
| `numpy` | Numerical operations |
| `matplotlib` | All plots and visualisations |
| `seaborn` | Plot styling and heatmap |
| `scikit-learn` | Preprocessing, feature selection, RF, GridSearchCV, metrics |
| `xgboost` | XGBoost classifier |

All libraries are pre-installed in Google Colab.

---

## Results Summary

### Feature Reduction (Notebook 2)

| Stage | Features Remaining |
|---|---|
| After aggregation (Notebook 1) | 104 |
| After zero-variance removal | 104 |
| After high-null removal (>90%) | 98 |
| After VarianceThreshold (0.01) | 95 |
| After RFE (best k=23) | 23 |

**23 features selected by RFE:** `Temp_max`, `Temp_mean`, `PaCO2_min`, `BUN_max`, `BUN_mean`, `Creatinine_min`, `Creatinine_max`, `Lactate_min`, `Lactate_max`, `Lactate_mean`, `Phosphate_min`, `Phosphate_mean`, `Potassium_mean`, `Hct_max`, `Hct_mean`, `Hgb_max`, `Hgb_mean`, `PTT_mean`, `WBC_min`, `WBC_max`, `Fibrinogen_max`, `Platelets_min`, `Platelets_max`

### Model Performance (Notebooks 3 and 4)

| Model | F1 Score | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| Random Forest (baseline) | 0.0000 | 0.7167 | — | — |
| Random Forest (tuned, GridSearch) | 0.0879 | 0.6965 | 0.2222 | 0.0548 |
| XGBoost (baseline) | 0.1538 | 0.6977 | — | — |
| XGBoost (tuned, GridSearch) | **0.2581** | **0.6999** | 0.2439 | 0.2740 |

XGBoost outperforms Random Forest on every metric and catches nearly 5× more sepsis cases (recall 27.4% vs 5.5%).

### Confusion Matrices

**Random Forest (tuned):** TN=913, FP=14, FN=69, TP=4  
**XGBoost (tuned):** TN=865, FP=62, FN=53, TP=20

---

## Interpretation

These results are modest, and that is expected. This is a basic introductory project — the goal is to understand the end-to-end machine learning workflow, not to build a production-ready clinical tool.

### What the numbers mean

The baseline Random Forest predicted every patient as non-sepsis (F1 = 0.00). This is a classic failure mode on imbalanced data — the model found it easier to ignore the minority class entirely. Adding `class_weight='balanced'` and tuning with GridSearchCV helped, but only marginally (F1 = 0.09). The tuned Random Forest still misses 69 out of 73 sepsis patients in the test set.

XGBoost does considerably better after tuning (F1 = 0.26), catching 20 of 73 sepsis cases. This is still far from clinically useful — more than half of all sepsis patients are still missed — but the improvement over Random Forest is meaningful and illustrates why boosting methods often outperform bagging methods on tabular, imbalanced datasets.

The ROC-AUC scores (~0.70 for both models) suggest the models have some genuine discriminative ability — they are not just guessing — but they struggle to convert that ability into good classification at a fixed decision threshold, primarily because of how imbalanced the classes are.

The selected features are clinically interpretable: temperature, kidney markers (BUN, creatinine), lactate, and blood counts are all well-established indicators of physiological stress and organ dysfunction, which aligns with the clinical understanding of sepsis progression.

### Why the performance is limited

Several factors contribute to the modest results:

- **Small training set:** We used only 5,000 of the 40,336 available patients. The sepsis subset within that sample is tiny — around 365 patients — which gives the models very little signal to learn from.
- **Heavy aggregation:** Collapsing a time-series into min/max/mean destroys the temporal dynamics of how vital signs and labs change over time. A patient whose lactate is rising rapidly is very different from one whose lactate is stable at the same mean value — but our features cannot represent this.
- **Aggressive feature reduction:** Reducing to 23 features may have discarded information that would have been useful.
- **Grid search limitations:** GridSearchCV only tests a small, manually defined grid of hyperparameter combinations. Many potentially better configurations are never evaluated.

### Ways to improve the models

This project intentionally keeps things simple. For students who want to go further, the following are natural next steps:

**Better handling of class imbalance**

- **SMOTE (Synthetic Minority Oversampling Technique):** Instead of re-weighting existing samples, SMOTE generates synthetic sepsis patients by interpolating between real ones. This gives the model more minority-class examples to learn from without simply repeating the same data points. It must be applied to the training set only, after the train/test split.
- **Increase the sepsis sample size:** Rather than taking a stratified 5,000-patient sample, deliberately oversample from the sepsis patients in the full dataset. For example, include all 2,932 sepsis patients and sample a proportional number of non-sepsis patients. This immediately gives the model more real sepsis cases to learn from.

**Better hyperparameter search**

- **Bayesian optimisation:** GridSearchCV evaluates all combinations exhaustively and treats each one independently. Bayesian optimisation (e.g. using `optuna` or `scikit-optimize`) builds a probabilistic model of how hyperparameters affect performance, and uses that model to choose the next combination to try. It finds better configurations with far fewer evaluations — particularly useful when the search space is large.

**Better features**

- **Temporal features:** Rather than just min/max/mean, compute features like rate of change (e.g. lactate increase over the last 6 hours), time since last measurement, or the number of hours with an abnormal value. These capture the dynamics that static aggregations miss.
- **Retain more patients:** Running on the full 40,336-patient dataset would give far more training signal, at the cost of longer compute time.

These improvements are not implemented here because the purpose of this project is to teach the fundamentals clearly. Each notebook is kept short enough to run and understand in one sitting. The improvements listed above are well-documented in the literature and are straightforward to implement once the baseline pipeline is understood.
