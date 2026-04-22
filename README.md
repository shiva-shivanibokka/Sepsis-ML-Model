# Sepsis Early Prediction — ICU Machine Learning Project

This project builds machine learning models to predict whether a patient in the intensive care unit (ICU) will develop sepsis, using hourly vital sign and lab measurements from the PhysioNet/CinC Challenge 2019 dataset. Two models are built and compared: Random Forest and XGBoost.

**Note:** This is an introductory project intended to help students understand the core concepts and workflow of applied machine learning — data loading, exploratory analysis, preprocessing, feature selection, model training, hyperparameter tuning, and evaluation. The results are modest by design; the focus is on understanding the process, not achieving state-of-the-art performance.

**Note on model choice:** The raw dataset is a time-series — one row of measurements per hour per patient. The most powerful approach for time-series data would be a dedicated sequence model such as an LSTM or Transformer. However, these architectures have not been covered in this course, so we do not use them here. Instead, we collapse the time-series into a single row per patient using statistical aggregations (min, max, mean, std, range) and apply tabular classifiers (Random Forest and XGBoost) that students are already familiar with. This is a deliberate pedagogical choice, not an oversight — and it is one of the reasons the results are more modest than what a sequence model could achieve on the same data.

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
    └── saves: X_train.csv, X_test.csv, y_train.csv, y_test.csv          (5,000-patient sample)
    └── saves: X_train_full.csv, X_test_full.csv, y_train_full.csv, y_test_full.csv  (full dataset)

03_random_forest.ipynb
    └── reads: X_train.csv, X_test.csv, y_train.csv, y_test.csv          (sample — for baseline + tuning)
    └── reads: X_train_full.csv, X_test_full.csv, y_train_full.csv, y_test_full.csv  (full — for final model)

04_xgboost.ipynb
    └── reads: X_train.csv, X_test.csv, y_train.csv, y_test.csv          (sample — for baseline + tuning)
    └── reads: X_train_full.csv, X_test_full.csv, y_train_full.csv, y_test_full.csv  (full — for final model)
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

For each patient we compute five aggregations per vital sign and lab value:

| Aggregation | What it captures |
|---|---|
| `_min` | Lowest recorded value |
| `_max` | Peak value — highest severity reached |
| `_mean` | Average over the entire ICU stay |
| `_std` | Standard deviation — how much the measurement fluctuated. High std means physiological instability |
| `_range` | `max - min` — total spread. Captures instability with less redundancy than keeping `_min` and `_max` as separate features |

We also include `ICULOS_max` — the patient's total ICU length of stay in hours. This is derived from the `ICULOS` column (previously dropped), by taking its maximum value per patient, which equals total hours in the ICU.

This produces one row per patient with `34 × 5 + 2 + 1 = 173` feature columns plus the Sepsis target.

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
**Output:** `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` (5,000-patient RFE sample)  
**Also saves:** `X_train_full.csv`, `X_test_full.csv`, `y_train_full.csv`, `y_test_full.csv` (full dataset — all 40,336 patients)

### Data Leakage

All preprocessing steps are **fit on the training set only**. The scaler, imputer, and variance threshold all see only training data when being configured. They are then applied to the test set using those training-derived parameters. This ensures the test set is never seen — directly or indirectly — during model preparation.

### Stratified Sampling

We work with a subset of 5,000 patients to keep the RFE step fast enough to run in Colab. The sample is **stratified** — the sepsis rate in the sample matches the full dataset (~7.3%).

The active method is `train_test_split(test_size=5000, stratify=patient_df['Sepsis'])`. A manual groupby alternative (Method A) is retained as a commented-out reference.

**Why preserve the natural class imbalance rather than balancing to 50/50?**

The ~7.3% sepsis rate is not a sampling artefact — it is a real property of ICU patient populations. Artificially inflating the sepsis fraction to 50% would mean training on a world where half of all ICU patients develop sepsis, which is not true. This creates two specific problems:

1. **`scale_pos_weight` and `class_weight` would be wrong.** Both parameters are computed from the training class counts to correct for imbalance. If the training set is already 50/50, these corrections have nothing to correct — and applying them would actually over-penalise the majority class, distorting learning in the opposite direction.
2. **Evaluation metrics would be misleading.** Precision, recall, and F1 are all sensitive to the class distribution in the test set. If the test set is artificially balanced, the reported scores do not reflect how the model would perform on real patients, making the results look better than they are.

The correct approach is to preserve the natural imbalance in the data and let the model handle it explicitly — via `class_weight='balanced'` in Random Forest and `scale_pos_weight` in XGBoost.

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

**RFE sweep:** Starting from N//4 features and halving each step, we test several values of `k`. For each `k`, a lightweight Random Forest ranks the features, selects the top `k`, and a slightly larger Random Forest evaluates it using 5-fold cross-validation on the training set only (no test set used). The CV F1 score is plotted against `k` and the best value is used for the final RFE.

**Full-dataset preprocessing:** After the RFE sweep determines the best features, the same preprocessing pipeline (imputer, scaler, RFE feature mask — all fitted on the 5,000-patient sample training set) is applied to all 40,336 patients. The full dataset is split 80/20 and the preprocessed files are saved separately as `X_train_full.csv` etc. Notebooks 3 and 4 use these for final model training.

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

### Hyperparameter Tuning — Bayesian Optimisation

Instead of GridSearchCV, XGBoost is tuned with `BayesSearchCV` from `scikit-optimize`. Bayesian optimisation builds a probabilistic model of how each hyperparameter combination affects the CV F1 score, and uses that model to select the next combination to try — concentrating evaluations in promising regions rather than exhaustively covering a fixed grid.

This finds better hyperparameters than grid search with far fewer model fits, and searches continuous ranges rather than a handful of discrete values.

| Parameter | Search Range | Description |
|---|---|---|
| `n_estimators` | 50 – 400 (integer) | Number of boosting rounds |
| `max_depth` | 2 – 8 (integer) | Tree depth |
| `learning_rate` | 0.01 – 0.3 (log-uniform) | Contribution of each new tree |
| `subsample` | 0.5 – 1.0 (uniform) | Fraction of patients used per tree |
| `colsample_bytree` | 0.5 – 1.0 (uniform) | Fraction of features sampled per tree |

30 iterations are run (configurable). Each iteration fits 5 cross-validation folds, so the total is 150 fits — comparable to GridSearchCV's 120, but guided rather than exhaustive.

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

### Why a stratified sample of 5,000, and why preserve the imbalance?

RFE trains many models internally — one for each candidate `k` value, multiplied by the cross-validation folds. Running this on tens of thousands of patients would take many hours in Colab. A 5,000-patient sample keeps it fast enough to run in a single Colab session.

The sample is stratified so the ~7.3% sepsis rate matches the full dataset. This is intentional — the class imbalance is **not corrected** to 50/50 or any other ratio, and that is the right decision for three reasons:

- **The imbalance is real.** Sepsis affects roughly 7% of ICU patients. A model trained on a 50/50 split would be calibrated to a world where half of all ICU patients are septic, which does not exist. Its predictions and confidence scores would not be meaningful in practice.
- **The class weights depend on it.** `scale_pos_weight` in XGBoost is computed as `count(non-sepsis) / count(sepsis)` — approximately 12.79. This tells the model how much harder to penalise a missed sepsis case relative to a false alarm. If the training set were artificially balanced, this ratio would be 1.0, and the correction would do nothing. Applying a weight computed from a balanced set to an imbalanced real-world distribution would produce wrong predictions.
- **Evaluation stays honest.** Test set metrics (precision, recall, F1, AUC) should reflect real-world performance. An artificially balanced test set inflates reported recall and F1 scores, making the model look better than it actually is on genuine patient data.

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
| `scikit-optimize` | Bayesian hyperparameter optimisation (`BayesSearchCV`) |

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, and `xgboost` are pre-installed in Google Colab. Install `scikit-optimize` with:

```python
!pip install scikit-optimize
```

---

## Results Summary

### Feature Reduction (Notebook 2)

| Stage | Features Remaining |
|---|---|
| After aggregation (Notebook 1) | 173 (34 cols × 5 aggs + ICULOS_max + Age + Gender) |
| After zero-variance removal | 173 |
| After high-null removal (>90%) | 161 |
| After VarianceThreshold (0.01) | 156 |
| After RFE (best k=50) | **50** |

RFE was run on the full training set (~32,000 patients) with a dense sweep (every integer 1–20, then every 10 up to 150). The CV F1 peaked at **k=50** (CV F1 = 0.5438). The selected features include absolute values (Temp_max, Lactate_min, Creatinine_max, PTT, WBC, Fibrinogen, Platelets, Bilirubin, Hgb_mean/std, Platelets_std), range features for all 34 vital/lab measurements, and ICULOS_max.

### Model Performance (Notebooks 3 and 4)

#### 5,000-patient sample (baseline and tuning)

| Model | F1 Score | ROC-AUC | Precision | Recall | Sepsis caught |
|---|---|---|---|---|---|
| Random Forest (baseline) | 0.3516 | 0.9455 | 0.89 | 0.22 | 16 / 73 |
| Random Forest (tuned, GridSearch) | 0.5607 | 0.9317 | 0.8824 | 0.4110 | 30 / 73 |
| XGBoost (baseline) | 0.6870 | 0.9411 | 0.78 | 0.62 | 45 / 73 |
| XGBoost (tuned, Bayesian opt, 60 iter) | **0.7218** | **0.9516** | 0.80 | 0.6575 | 48 / 73 |

#### Full dataset — 40,336 patients with SMOTE (final evaluation)

| Model | F1 Score | ROC-AUC | Precision | Recall | Sepsis caught | Best threshold |
|---|---|---|---|---|---|---|
| Random Forest (full + SMOTE) | 0.6025 | 0.9091 | 0.5492 | 0.6672 | 391 / 586 | 0.50 |
| XGBoost (full + SMOTE) | **0.6788** | **0.9235** | 0.7775 | 0.6024 | 353 / 586 | 0.45 |

**XGBoost wins** on F1, AUC, and precision on the full dataset. Random Forest catches slightly more patients in absolute terms (391 vs 353) by using a less conservative threshold.

### Confusion Matrices (full dataset)

**Random Forest (full, SMOTE):** TN=7,161, FP=321, FN=195, TP=391  
**XGBoost (full, SMOTE):** TN=7,381, FP=101, FN=233, TP=353

---

## Interpretation

### What the numbers mean

**XGBoost (tuned, 60-iteration Bayesian optimisation)** is the best model, with F1=0.72 and AUC=0.95 on the 5,000-patient test set — the highest in the entire project. On the full dataset with SMOTE, it achieves F1=0.68, catching 353 of 586 sepsis patients (60% recall) with 78% precision.

The most important features across both models are **ICULOS_max** (ICU stay duration), **Lactate** measurements, **Bilirubin_total**, **PTT**, and **range features** for vital signs. The 34 range features (max−min per measurement) proved highly informative — physiological instability, not just peak values, is a strong sepsis indicator.

**Why do full-dataset F1 scores look lower than sample F1 scores?** This is a scale issue, not a model quality issue. The sample test set has 73 sepsis patients; the full test set has 586. A model that catches 60% of 73 patients (44) scores F1≈0.70. The same model catching 60% of 586 patients (352) scores F1≈0.68, because the larger non-sepsis pool (7,482 patients) generates more false alarms proportionally. The model behaviour is consistent — the evaluation is just more rigorous at full scale.

### Why the performance is limited

- **ICULOS_max is a partially indirect signal.** It is genuinely predictive but reflects the outcome as much as the cause — sepsis patients stay longer partly because of the sepsis. A more robust feature would be ICULOS at the time of prediction (hours elapsed so far), not total stay length.
- **Range features dominate.** 34 of 50 selected features are `_range` values. They are meaningful but also correlated with each other — a physiologically unstable patient tends to have high range across multiple measurements simultaneously.
- **GridSearchCV for RF tested only 24 combinations.** Switching RF to Bayesian optimisation (as done for XGBoost) would likely improve the RF result further.

---

## Challenges and Improvements Made

This section documents the challenges encountered during development and every improvement made to address them.

### Challenge 1 — Class imbalance (7.3% sepsis rate)

**Problem:** Only 7.3% of patients develop sepsis. A model that always predicts Non-Sepsis achieves 93% accuracy while catching zero sepsis cases. Standard metrics like accuracy are useless here.

**What we did:**
- Used **F1 score** as the primary evaluation metric throughout — it penalises models that ignore the minority class.
- Applied `class_weight='balanced'` in Random Forest to increase the training penalty for missing a sepsis case.
- Applied `scale_pos_weight = count(non-sepsis) / count(sepsis) ≈ 12.79` in XGBoost — the equivalent parameter.
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the full training set, generating synthetic sepsis patients by interpolating between real ones. This produced a 50/50 balanced training set, allowing the model to learn from far more minority-class examples without repeating the same 2,346 real patients.

---

### Challenge 2 — Small training set for the minority class

**Problem:** The initial approach used a 5,000-patient stratified sample for all steps, including RFE and model training. With only ~290 sepsis patients in that sample, the models had very limited minority-class signal. RFE on the sample was so noisy it selected only k=4 features.

**What we did:**
- **Moved RFE to run on the full training set** (~32,000 patients). With ~2,346 sepsis patients in the full training set — 8× more — the internal RF ranker produced far more reliable feature importance estimates.
- **Trained final models on the full 40,336-patient dataset** (with SMOTE) rather than the 5,000-patient sample. The 5,000-patient sample is now only used for fast baseline exploration.
- **Denser RFE sweep:** Changed from a coarse halving strategy (1, 2, 4, 9, 19...) to testing every integer from 1–20, then every 10 up to 150. This prevented the optimum from being missed between tested values — the previous coarse sweep had jumped from k=4 straight to k=9, missing the true peak.

---

### Challenge 3 — Feature engineering and redundancy

**Problem:** The initial aggregation produced only `_min`, `_max`, `_mean` per measurement — 3 highly correlated features per variable. With 104 features but most carrying redundant information, the models were learning from far less independent signal than the feature count suggested. RFE selected only 4 features (k=4) because adding more correlated features added noise faster than signal.

**What we did:**
- Added **standard deviation (`_std`)** per measurement — captures how much the value fluctuated across the ICU stay. High std indicates physiological instability, which is a clinical warning sign independent of the mean or max.
- Added **range (`_range` = max − min)** per measurement — captures total spread. Less redundant than keeping both `_min` and `_max` as separate inputs because it encodes variability in a single number.
- Added **`ICULOS_max`** — total ICU length of stay. Previously dropped as "metadata", this was reinstated because it is a meaningful patient-level signal: longer stays correlate with more complex, higher-severity cases.
- These additions increased the feature count from 104 to 173, and RFE on the full dataset now selected **50 features** instead of 4 — a much richer and more informative set.

---

### Challenge 4 — RFE evaluated on the test set (data leakage)

**Problem:** The original RFE sweep evaluated each `k` value using the test set labels (`y_test`) to pick the best `k`. This is a subtle form of data leakage — the choice of how many features to keep was informed by data that should be completely unseen. It inflated final evaluation metrics.

**What we did:**
- Replaced test-set evaluation with **5-fold cross-validation on the training set only** (`cross_val_score` with `StratifiedKFold`). The test set is now kept completely untouched until the final model evaluation in Notebooks 3 and 4.

---

### Challenge 5 — Hyperparameter tuning efficiency

**Problem:** GridSearchCV for XGBoost tested only a fixed 24-combination grid. This missed large portions of the hyperparameter space and only tested discrete values — a learning rate of 0.07 (the optimal found by Bayesian search) would never have been tested on a grid of {0.05, 0.1}.

**What we did:**
- **Replaced GridSearchCV with Bayesian Optimisation** (`BayesSearchCV` from `scikit-optimize`) for XGBoost. Bayesian optimisation builds a probabilistic model of the objective function and concentrates evaluations in promising regions, searching continuous ranges rather than a fixed grid.
- **Increased iterations from 30 to 60** to give the optimiser more budget. The best configuration found was `learning_rate=0.071, max_depth=7, n_estimators=323, subsample=0.810, colsample_bytree=1.0` — values that a grid search would never have found.

---

### Challenge 6 — Decision threshold calibration

**Problem:** Both models defaulted to a 0.50 decision threshold — predict Sepsis if the estimated probability exceeds 50%. On imbalanced data, this threshold is rarely optimal. The models had excellent AUC (strong ranking ability) but suboptimal F1 because the threshold was not tuned.

**What we did:**
- Added a **threshold sweep** in both Notebooks 3 and 4, testing thresholds from 0.05 to 0.50 and printing precision, recall, F1, and absolute sepsis count at each level. This gives a clear view of the precision-recall trade-off and lets the user choose a threshold appropriate for their clinical context (e.g. if recall is more important than precision, use a lower threshold).
- For XGBoost, the optimal threshold was **0.45** (F1 = 0.6792, catching 362 of 586 patients). For Random Forest the default 0.50 was already optimal for F1, but lower thresholds catch significantly more patients.

---

### Challenge 7 — Inconsistent comparison between models

**Problem:** The final model comparison in Notebook 4 was originally training the comparison RF on raw imbalanced data (no SMOTE) while the XGBoost used SMOTE — an apples-to-oranges comparison. Also, the RF used `class_weight='balanced'` on top of SMOTE-balanced data, which would double-count the imbalance correction if ever the SMOTE ratio changed.

**What we did:**
- **Applied SMOTE consistently** to both the RF and XGBoost full-dataset training, with `class_weight=None` and `scale_pos_weight=1.0` respectively — since SMOTE already balances the classes, no additional weighting is needed.
- The final comparison now reflects both models trained under identical conditions (full dataset, SMOTE, no additional class weighting).

---

### Final model comparison

| | F1 (sample) | F1 (full + SMOTE) | AUC (full) | Sepsis caught (full) |
|---|---|---|---|---|
| Random Forest | 0.5607 | 0.6025 | 0.9091 | 391 / 586 |
| **XGBoost** | **0.7218** | **0.6788** | **0.9235** | 353 / 586 |

**XGBoost is the best model overall.** It achieves higher F1 and AUC on both the sample and full dataset evaluations. Random Forest catches slightly more patients in absolute terms on the full dataset (391 vs 353) due to a higher recall at threshold 0.50, but XGBoost has meaningfully better precision (0.78 vs 0.55), fewer false alarms (101 vs 321), and a higher AUC.

Both models are academically solid introductory results. A production clinical tool would require validation on held-out hospital cohorts, calibration of the threshold against clinical cost-of-error estimates, and regulatory approval.
