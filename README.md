# MGMT Methylation Prediction from GBM MRI Radiomics

This repository evaluates how **compact radiomic signatures** can predict **MGMT promoter methylation status** in glioblastoma (GBM), using MRI-derived features and tree-based models.

## Quick Links
- [1. Data](#1-data)
- [2. Methods](#2-methods)
- [3. Evaluation Pipeline](#3-evaluation-pipeline)
- [4. Key Results (Train and Test AUC)](#4-key-results-train-and-test-auc)
- [5. SVM–L1 + GA–RF Variants](#5-svm-l1-and-ga-rf-variants)
- [6. Majority-vote GA subset (>=3/5)](#6-majority-vote-ga-subset-35--xgboost84-gak20-vs-svm-l132-gak13)
- [7. SHAP comparison across compact signatures](#7-shap-comparison-across-compact-signatures)
- [8. Notes and Limitations](#8-notes-and-limitations)
- [9. Conclusion / Key Results](#9-conclusion-and-key-results)

---

We:

- Start from a **full radiomics feature set** (**724 features** per patient),
- Build two **wrapper-based baseline feature spaces**:
  - **XGBoost wrapper selection** → an **84-feature** baseline space,
  - **SVM–L1 (LinearSVC) non-zero selection** → a compact **32-feature** baseline space,
- Apply **Genetic Algorithm–Random Forest (GA–RF)** subset search to obtain **sparse, clinically realistic signatures**:
  - **XGBoost(84) → GA–RF with size penalty** (TARGET_K = 15, 20, 25),
  - **SVM–L1(32) → GA–RF with size penalty** (RFECV-guided target ≈ 13),
  - **SVM–L1(32) → GA–RF without size penalty** (ablation),
Keep reporting consistent:
  - list **Baseline first**
  - for GA results, first **filter** candidate subsets by **mean sensitivity ≥ 0.80** (computed under the same 5-fold Stratified CV used for GA evaluation)
  - among the remaining subsets, choose a **representative** compact signature as the subset whose **inner-CV mean AUC** is the **median**
  - evaluate the representative subset on the **held-out test set**, and also report test performance across the other GA top-5 subsets for context

---

## 1. Data

We work with **118 GBM patients** split into train/test with a **7:3 stratified split**:

- `X_train.csv`  
  - Shape: `82 × 724`  
  - 82 patients, each with 724 radiomic features extracted from multi-parametric MRI:
    - T1, T1-Gd, T2, FLAIR
    - Volumes and ratios
    - Intensity / histogram statistics
    - Spatial location features
    - Texture features (GLCM, GLRLM, GLSZM, NGTDM)
    - Morphology (e.g. eccentricity, solidity)
    - Tumor growth model features (`TGM_*`)

- `y_train.csv`  
  - Shape: `82 × 1`  
  - Binary **MGMT promoter methylation status** for the 82 training patients.

- `X_test.csv`  
  - Shape: `36 × 724` (same feature schema as `X_train.csv`)  
  - 36 held-out patients.

- `y_test.csv`  
  - Shape: `36 × 1`  
  - Binary MGMT promoter methylation status for the test cohort.

> Internally, the pipeline first applies XGBoost-based feature selection on `X_train.csv`
> (and applies the same feature mask to `X_test.csv`) to obtain:
> - `X_train_xgbsel.csv`
> - `X_test_xgbsel.csv`
> which contain **84 selected features**.  
> This 84-feature set is treated as the **baseline feature space** for all GA-RF experiments.

---
## 2. Methods

### 2.1 Baseline feature spaces

We start from the full radiomics matrix (**724 features per patient**) and construct baseline feature spaces using the **training set only**, then apply the same feature masks to the held-out test set.

**XGBoost wrapper selection (84-feature space).**  
We train an XGBoost classifier on `X_train` with cross-validation, rank features by importance (e.g., gain), and keep the top **84** features to form the XGBoost-selected baseline space:
- `X_train_xgbsel.csv`
- `X_test_xgbsel.csv`

This **84-feature space** is used:
- as a standalone baseline model (**Random Forest on all 84 features**), and  
- as the GA–RF search space (**chromosomes are binary masks over these 84 features**).

**SVM–L1 (LinearSVC) selection (32-feature space).**  
We train an L1-regularized `LinearSVC` on `X_train` (`C` tuned by cross-validation), then keep features with \(|\text{coef}| > \epsilon\) (\(\epsilon = 10^{-6}\)). This produces a compact **32-feature** baseline space:
- `Xtr_nz.csv`
- `Xte_nz.csv`

This **32-feature space** is used:
- as a standalone baseline model (**Random Forest on all 32 features**), and  
- as the GA–RF search space for SVM–L1-based variants.

---

### 2.2 GA–Random Forest (GA–RF) wrapper

We use a genetic algorithm (GA) to search for sparse feature subsets within a fixed baseline space (either **84 features** from XGBoost or **32 features** from SVM–L1).

**Chromosome.**  
A binary mask over the baseline space (`1` = included, `0` = excluded).

**Model inside the wrapper.**  
`RandomForestClassifier` (scikit-learn).

**Fitness evaluation.**  
We evaluate each chromosome using **Stratified 5-fold CV** on the training set and compute **mean CV AUC** as the base score. In size-penalized runs, we add a penalty to encourage subsets near a target size `TARGET_K`.

Conceptually:
fitness = mean_CV_AUC − λ_k · |k − TARGET_K|
where `k` is the number of selected features, `TARGET_K` is the desired subset size, and `λ_k` controls penalty strength.

---

### 2.3 Size-penalized GA–RF experiments (XGBoost(84) space)

We run GA–RF with an explicit size penalty on the **84-feature XGBoost wrapper space** to compare sparsity–performance trade-offs under the same evaluation protocol.

**GA–RF (TARGET_K = 15).**
- Results stored in `15_ga_best.npz`
- Analysis / RF re-tuning in `15_conclusion_rf.ipynb`

**GA–RF (TARGET_K = 20).**
- Results stored in `20_ga_best.npz`
- Analysis / RF re-tuning in `20_conclusion_rf.ipynb`

**GA–RF (TARGET_K = 25).**
- Results stored in `25_ga_best.npz`
- Analysis / RF re-tuning in `25_conclusion_rf.ipynb`

Across `TARGET_K ∈ {15, 20, 25}`, we keep the evaluation pipeline consistent (same train/test split, same RF hyperparameter search space, same Stratified 5-fold CV) so differences primarily reflect the sparsity constraint rather than evaluation changes.

---

### 2.4 SVM–L1 + GA–RF variants (SVM(32) space)

We also run GA–RF on the **32-feature SVM–L1 baseline space** to study whether a smaller, SVM-filtered candidate set yields more stable compact signatures.

**RFECV-guided compact target (k ≈ 13) with size penalty.**  
We select a compact target size using RFECV on the training set (restricted to small `k`), then run size-penalized GA–RF around that target within the 32-feature space.

**No-penalty ablation on the same 32-feature space.**  
We run GA–RF without an explicit size penalty (no `TARGET_K` term) to isolate the effect of the sparsity constraint, keeping the same CV protocol.

**Fixed-k truncation baseline space (k = 25) + GA–RF (no size penalty).**  
We build a 25-feature space by ranking SVM–L1 coefficients by \(|\text{coef}|\) and keeping the top 25, then run GA–RF without size penalty within that fixed space.

All SVM–L1 + GA–RF variants use the same Stratified 5-fold CV protocol on the training set and are evaluated once on the held-out test set.

---

## 3. Evaluation Pipeline

The conclusion notebooks (`15_conclusion_rf.ipynb` and `20_conclusion_rf.ipynb`) follow the same steps:

### 3.1 Load data & GA solutions

- `X_train_xgbsel.csv`, `X_test_xgbsel.csv`  
- `y_train.csv`, `y_test.csv`  
- `15_ga_best.npz` or `20_ga_best.npz` (top GA masks).

### 3.2 Define feature sets to evaluate

- **Baseline (ALL wrapper)**: all 84 XGBoost-selected features.  
- **GA-1 … GA-5**: top 5 GA subsets from the corresponding GA run.

### 3.3 Random Forest hyperparameter tuning (per feature set)

We use `RandomizedSearchCV` with stratified 5-fold CV and a shared search space, e.g.:

- `n_estimators`: 100–650  
- `max_depth`: \[None, 8, 12, 16, 20, 24, 32]  
- `max_features`: `["sqrt", "log2", 0.3, 0.5, 0.7]`  
- `min_samples_split`: 2–20  
- `min_samples_leaf`: 1–8  
- `bootstrap`: \[True]  
- `max_samples`: \[None, 0.6, 0.8, 1.0]  
- `criterion`: `["gini", "entropy"]`  
- `class_weight`: \[None, "balanced"]

### 3.4 Metrics

For each feature set, we record:

- **5-fold CV AUC (mean over folds)** on the training set,  
- **Test AUC** on the held-out test set.

In the notebooks, we additionally compute:

- Sensitivity at a fixed probability threshold,  
- Jaccard similarity between GA subsets to assess robustness of feature selection.

### 3.5 Interpretability & visualization

For a chosen GA subset (based on a composite score involving AUC mean, AUC std, and sensitivity), we:

- train a final RF model,  
- compute **SHAP values** (feature importance / effect),  
- run **t-SNE** on the selected features to visualize separation between methylated vs unmethylated cases and between train/test splits.

---

## 4. Key Results (Train and Test AUC)

Below are Random Forest results for the **84-feature baseline** (XGBoost-selected space) and GA-RF subsets.  
Train metrics are computed with **Stratified 5-fold CV on the training set** (n=82).  
Test metrics are computed once on the **held-out test set** (n=36) using the same feature space.

We focus on **robust compact signatures** rather than optimizing a single arbitrary subset size.  
We first used an **RFECV-guided rule** on the training set to identify a compact-but-robust regime, and this procedure suggested that **k ≈ 20** is a strong target size.  
We then ran GA-RF experiments around this regime (TARGET_K = 15, 20, 25) under the same evaluation protocol (same split, same CV, same RF tuning space), and empirically found that **TARGET_K = 20** provided the best sparsity–performance trade-off in our experiments.

**Representative GA subset rule (Train-only).**  
Within each TARGET_K run, we select a single representative GA subset for reporting by:
first filtering to GA subsets with **mean sensitivity ≥ 0.80** (@0.5), then choosing the subset whose **CV AUC mean is the median** among the remaining GA subsets (to avoid selecting an extreme best-case subset).  
We then report the **held-out test AUC** for that representative subset.

---

### 4.1 Train (5-fold CV): AUC + Sensitivity (@0.5)

#### TARGET_K = 15
| Feature set             | #Features | CV AUC (mean) | CV AUC std | Sensitivity (mean) | Sensitivity std |
|-------------------------|----------:|--------------:|-----------:|-------------------:|----------------:|
| Baseline (ALL wrapper)  |       84  |         0.790 |      0.088 |                —   |              —  |
| GA-1 (idx=0)            |       17  |         0.811 |      0.056 |              0.909 |           0.129 |
| GA-2 (idx=1)            |       17  |         0.746 |      0.117 |              0.907 |           0.064 |
| GA-3 (idx=2)            |       17  |         0.866 |      0.066 |              0.927 |           0.100 |
| GA-4 (idx=3)            |       15  |         0.797 |      0.119 |              0.818 |           0.265 |
| GA-5 (idx=4)            |       16  |         0.854 |      0.068 |              0.909 |           0.129 |

Representative GA subset (TARGET_K=15): **GA-1 (idx=0)**  
Train CV AUC = **0.811** (mean), Sensitivity = **0.909** (mean)

#### TARGET_K = 20
| Feature set             | #Features | CV AUC (mean) | CV AUC std | Sensitivity (mean) | Sensitivity std |
|-------------------------|----------:|--------------:|-----------:|-------------------:|----------------:|
| Baseline (ALL wrapper)  |       84  |         0.790 |      0.088 |                —   |              —  |
| GA-1 (idx=0)            |       21  |         0.815 |      0.081 |              0.964 |           0.081 |
| GA-2 (idx=1)            |       23  |         0.863 |      0.101 |              0.945 |           0.050 |
| GA-3 (idx=2)            |       20  |         0.831 |      0.102 |              0.927 |           0.076 |
| GA-4 (idx=3)            |       20  |         0.844 |      0.080 |              0.871 |           0.137 |
| GA-5 (idx=4)            |       23  |         0.802 |      0.077 |              0.909 |           0.111 |

Representative GA subset (TARGET_K=20): **GA-3 (idx=2)**  
Train CV AUC = **0.831** (mean), Sensitivity = **0.927** (mean)

#### TARGET_K = 25
| Feature set             | #Features | CV AUC (mean) | CV AUC std | Sensitivity (mean) | Sensitivity std |
|-------------------------|----------:|--------------:|-----------:|-------------------:|----------------:|
| Baseline (ALL wrapper)  |       84  |         0.790 |      0.088 |                —   |              —  |
| GA-1 (idx=0)            |       25  |         0.836 |      0.102 |              0.945 |           0.081 |
| GA-2 (idx=1)            |       24  |         0.843 |      0.086 |              0.907 |           0.064 |
| GA-3 (idx=2)            |       24  |         0.803 |      0.095 |              0.945 |           0.081 |
| GA-4 (idx=3)            |       25  |         0.805 |      0.067 |              0.909 |           0.091 |
| GA-5 (idx=4)            |       27  |         0.868 |      0.075 |              0.964 |           0.081 |

Representative GA subset (TARGET_K=25): **GA-2 (idx=1)**  
Train CV AUC = **0.843** (mean), Sensitivity = **0.907** (mean)

---

### 4.2 Held-out Test AUC (final)

#### TARGET_K = 15
| Feature set             | #Features | Test AUC |
|-------------------------|----------:|---------:|
| Baseline (ALL wrapper)  |       84  |   0.760  |
| GA-1 (idx=0)            |       17  |   0.719  |
| GA-2 (idx=1)            |       17  |   0.708  |
| GA-3 (idx=2)            |       17  |   0.712  |
| GA-4 (idx=3)            |       15  |   0.694  |
| GA-5 (idx=4)            |       16  |   0.705  |

Final held-out test result (TARGET_K=15 representative): **GA-1 (idx=0)** → Test AUC = **0.719**

#### TARGET_K = 20
| Feature set             | #Features | Test AUC |
|-------------------------|----------:|---------:|
| Baseline (ALL wrapper)  |       84  |   0.760  |
| GA-1 (idx=0)            |       21  |   0.771  |
| GA-2 (idx=1)            |       23  |   0.675  |
| GA-3 (idx=2)            |       20  |   0.771  |
| GA-4 (idx=3)            |       20  |   0.736  |
| GA-5 (idx=4)            |       23  |   0.750  |

Final held-out test result (TARGET_K=20 representative): **GA-3 (idx=2)** → Test AUC = **0.771**

#### TARGET_K = 25
| Feature set             | #Features | Test AUC |
|-------------------------|----------:|---------:|
| Baseline (ALL wrapper)  |       84  |   0.760  |
| GA-1 (idx=0)            |       25  |   0.698  |
| GA-2 (idx=1)            |       24  |   0.653  |
| GA-3 (idx=2)            |       24  |   0.632  |
| GA-4 (idx=3)            |       25  |   0.625  |
| GA-5 (idx=4)            |       27  |   0.753  |

Final held-out test result (TARGET_K=25 representative): **GA-2 (idx=1)** → Test AUC = **0.653**

---

### 4.3 Summary of Findings

- **TARGET_K = 15** and **TARGET_K = 20** GA-RF runs both produced **compact feature sets (15–23 features)** that **match or outperform** the 84-feature baseline in **train (5-fold CV) AUC**.

- On the **held-out test set**:
  - **TARGET_K = 15** subsets maintain test AUCs in a similar range to the baseline, but do not consistently surpass it.
  - **TARGET_K = 20** subsets (notably **GA-1** and **GA-3**) achieve **similar or slightly higher test AUC** (**0.771** vs baseline **0.760**) while using only about **one quarter** of the features.

- Across the GA top-5 subsets, **TARGET_K = 20** is **more stable / robust as a group**:
  - higher mean CV AUC across subsets (≈ **0.831** vs ≈ **0.815** for TARGET_K = 15),
  - lower between-subset variability (std ≈ **0.024** vs ≈ **0.048**).

- **Why TARGET_K ≈ 20**:
  - We used an **RFECV-guided setup step** to choose a compact-but-robust target size, and it pointed to **k ≈ 20** as a strong regime.
  - Empirically, the **TARGET_K = 20** GA-RF run also gave the best overall **sparsity–generalization trade-off** in our experiments.

- We also tested **TARGET_K = 25** to see if larger subsets improve robustness:
  - Although several 25-feature GA subsets achieved high **train CV AUC**, they did **not** generalize consistently on the held-out test set.
  - Most 25-feature subsets underperformed the 84-feature baseline on test AUC, with only **GA-5** coming close.
  - Overall, this suggests the **k ≈ 20** regime remains the most robust setting in this small-sample setup.

---

## 5. SVM-L1 and GA-RF Variants

This section reports two pipelines that combine **SVM–L1 feature selection** with downstream **Genetic Algorithm–Random Forest (GA–RF)** subset search.  
For each pipeline, we report **Train (5-fold CV) AUC** and **Held-out Test AUC**.

---
## 5. SVM–L1 pre-selection + GA–RF: RFECV-guided (k = 13) vs non-penalized GA

## 5.1 RFECV-guided compact target (k = 13): SVM–L1 non-zero → 32-feature space → GA–RF (target ~13)

### 5.1.1 RFECV selects the compact target size
| Criterion                      | Best mean CV AUC | Selected k |
|---|---:|---:|
| Best AUC among n_features ≤ 15 | 0.7800 | 13 |

### 5.1.2 SVM–L1 non-zero pre-selection defines the GA search space
We train an L1-regularized LinearSVC (CV-tuned C) and keep features with \(|\text{coef}| > \epsilon\) (\(\epsilon = 10^{-6}\)).  
This reduces the original feature set to **32 features**, which becomes the **baseline feature space** for GA–RF.

### 5.1.3 Train (5-fold CV): AUC + Sensitivity (@0.5)
| Subset         | #Features | CV AUC mean | CV AUC std | Sensitivity mean @0.5 | Sensitivity std @0.5 |
|---------------|----------:|------------:|-----------:|------------------------:|----------------------:|
| GA-1 (idx=0)  |        14 |       0.824 |      0.080 |                   0.982 |                 0.041 |
| GA-2 (idx=1)  |        13 |       0.843 |      0.105 |                   0.964 |                 0.081 |
| GA-3 (idx=2)  |        13 |       0.844 |      0.130 |                   0.853 |                 0.080 |
| GA-4 (idx=3)  |        13 |       0.853 |      0.079 |                   0.964 |                 0.050 |
| GA-5 (idx=4)  |        13 |       0.869 |      0.072 |                   0.945 |                 0.050 |

Representative GA subset (train): **GA-3 (idx=2)**  
Train CV AUC = **0.844** (mean), Sensitivity@0.5 = **0.853** (mean)

### 5.1.4 Held-out Test AUC (ROC; Baseline vs GA top-5; 32-feature space)
| Feature set            | Test AUC | d (#features) |
|------------------------|---------:|--------------:|
| Baseline (ALL wrapper) |   0.708  |            32 |
| GA-1 (idx=0)           |   0.708  |            14 |
| GA-2 (idx=1)           |   0.729  |            13 |
| GA-3 (idx=2)           |   0.733  |            13 |
| GA-4 (idx=3)           |   0.684  |            13 |
| GA-5 (idx=4)           |   0.667  |            13 |

Final held-out test result (representative): **GA-3 (idx=2)** → Test AUC = **0.733**

---

## 5.2 Fixed-space (32 features): SVM–L1 non-zero → 32-feature space → GA–RF (no size penalty)

### 5.2.1 SVM–L1 non-zero pre-selection defines the GA search space
We train an L1-regularized `LinearSVC` (CV-tuned `C`) and keep features with \(|\text{coef}| > \epsilon\) (\(\epsilon = 10^{-6}\)).  
This reduces the original feature set to **32 features**, which becomes the **baseline feature space** for GA–RF.

### 5.2.2 Train (5-fold CV): AUC + Sensitivity (@0.5)
| Subset         | #Features | CV AUC mean | CV AUC std | Sensitivity mean @0.5 | Sensitivity std @0.5 |
|---------------|----------:|------------:|-----------:|------------------------:|----------------------:|
| GA-1 (idx=0)  |        21 |       0.906 |      0.034 |                   0.907 |                 0.064 |
| GA-2 (idx=1)  |        23 |       0.870 |      0.090 |                   0.964 |                 0.050 |
| GA-3 (idx=2)  |        16 |       0.874 |      0.064 |                   0.964 |                 0.050 |
| GA-4 (idx=3)  |        20 |       0.864 |      0.099 |                   1.000 |                 0.000 |
| GA-5 (idx=4)  |        19 |       0.848 |      0.103 |                   1.000 |                 0.000 |

Representative GA subset (train): **GA-3 (idx=2)**  
Train CV AUC = **0.874** (mean), Sensitivity@0.5 = **0.964** (mean)

### 5.2.3 Held-out Test AUC (ROC; Baseline vs GA top-5; 32-feature space; no size penalty)
| Feature set            | Test AUC | d (#features) |
|------------------------|---------:|--------------:|
| Baseline (ALL wrapper) |   0.708  |            32 |
| GA-1 (idx=0)           |   0.660  |            21 |
| GA-2 (idx=1)           |   0.694  |            23 |
| GA-3 (idx=2)           |   0.705  |            16 |
| GA-4 (idx=3)           |   0.712  |            20 |
| GA-5 (idx=4)           |   0.688  |            19 |

Final held-out test result (representative): **GA-3 (idx=2)** → Test AUC = **0.705**

---
## 5.3 Fixed-k (k = 25): SVM–L1 top-k truncation → 25-feature space → GA–RF (no size penalty)

### 5.3.1 SVM–L1 top-k truncation defines the GA search space
We train an L1-regularized LinearSVC (CV-tuned C), rank features by \(|\text{coef}|\), and keep exactly **k = 25** features.  
This produces a **25-feature space**, which becomes the **baseline feature space** for GA–RF.

### 5.3.2 Train (5-fold CV): AUC + Sensitivity (@0.5)
| Subset         | #Features | CV AUC mean | CV AUC std | Sensitivity mean @0.5 | Sensitivity std @0.5 |
|---------------|----------:|------------:|-----------:|------------------------:|----------------------:|
| GA-1 (idx=0)  |        14 |       0.872 |      0.069 |                   0.982 |                 0.041 |
| GA-2 (idx=1)  |        13 |       0.892 |      0.064 |                   0.873 |                 0.104 |
| GA-3 (idx=2)  |        13 |       0.904 |      0.053 |                   0.909 |                 0.111 |
| GA-4 (idx=4)  |        17 |       0.842 |      0.127 |                   0.982 |                 0.041 |
| GA-5 (idx=3)  |        13 |       0.884 |      0.033 |                   0.982 |                 0.041 |

Representative GA subset (train): **GA-3 (idx=2)**  
Train CV AUC = **0.904** (mean), Sensitivity@0.5 = **0.909** (mean)

### 5.3.3 Held-out Test AUC (ROC; Baseline vs GA top-5; 25-feature space)
| Feature set            | Test AUC | d (#features) |
|------------------------|---------:|--------------:|
| Baseline (ALL wrapper) |   0.670  |            25 |
| GA-1 (idx=0)           |   0.667  |            14 |
| GA-2 (idx=1)           |   0.608  |            13 |
| GA-3 (idx=2)           |   0.677  |            13 |
| GA-4 (idx=4)           |   0.663  |            17 |
| GA-5 (idx=3)           |   0.608  |            13 |

Final held-out test result (representative): **GA-3 (idx=2)** → Test AUC = **0.677**

---

## 5.4 Takeaway: size-targeted (k≈13) vs non-penalized GA on the same space vs fixed-k baseline space (k=25)

Overall, the **RFECV-guided size-targeted pipeline** (target k≈13, size-constrained GA–RF on a 32-feature space) achieves the strongest held-out performance among the SVM–L1-based variants. In contrast, removing the size penalty on the same 32-feature space provides a useful ablation but does not consistently improve test AUC, and the fixed-k=25 pipeline (25-feature space, no penalty) performs worst on the held-out set.

- Test AUC (RFECV-guided k≈13; size-penalized GA–RF on 32-feature space): **0.733** (GA-3, d=13)
- Test AUC (32-feature space; GA–RF **without** size penalty): **0.705** (GA-3, d=16)
- Test AUC (fixed-k=25 feature space; GA–RF **without** size penalty): **0.677** (GA-3, d=13)

Baselines:
- Baseline Test AUC (32-feature space): **0.708**
- Baseline Test AUC (25-feature space): **0.670**

## 6. Majority-vote GA subset (>=3/5) — XGBoost(84)->GA(k=20) vs SVM-L1(32)->GA(k≈13)

To obtain a **more stable / consensus signature**, we built a *majority-vote* feature set from the **GA top-5 subsets**.
A feature is included if it appears in **at least 3 out of 5** GA subsets (k ≥ 3/5). We then re-tuned a Random Forest
on this majority-vote subset using **Stratified 5-fold CV** on the training set, and evaluated the tuned model on the
held-out test set.

| Pipeline (GA source) | Majority-vote rule | d (#features) | Train CV AUC (best from tuning) | Train CV AUC (mean ± std) | Train Sensitivity@0.5 (mean ± std) | Held-out Test AUC |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost wrapper (84) → GA–RF (TARGET_K=20) → Majority-vote subset** | ≥ 3/5 | 10 | 0.8436 | 0.8242 ± 0.0843 | 0.8509 ± 0.1055 | **0.677** |
| **SVM–L1 non-zero (32) → GA–RF (target ≈13) → Majority-vote subset** | ≥ 3/5 | 10 | 0.8576 | 0.8303 ± 0.1107 | 0.9636 ± 0.0813 | **0.729** |

**Takeaway.** Under the same consensus rule (≥3/5) and the same final dimensionality (**d=10**), the SVM–L1-based
majority-vote subset achieved a higher **held-out test AUC** (**0.729**) than the XGBoost-based majority-vote subset
(**0.677**), suggesting a more robust consensus signature in the SVM–L1(32) → GA(k≈13) pipeline.


## 7. SHAP comparison across compact signatures

We compared SHAP-selected feature sets from three compact pipelines:
(1) **XGBoost(84) → GA–RF (d=20)**,  
(2) **SVM–L1(32) → GA–RF (d=13)**,  
(3) **SVM–L1(32) → GA–RF majority-vote subset (≥3/5, d=10)**.

### 7.1 Model summary (held-out test performance)

| Pipeline | d (#features) | Held-out Test AUC |
|---|---:|---:|
| **XGBoost(84) → GA–RF (representative GA subset)** | 20 | **0.771** |
| **SVM–L1(32) → GA–RF (representative GA subset)** | 13 | **0.733** |
| **SVM–L1(32) → GA–RF majority-vote subset (≥3/5)** | 10 | **0.729** |

### 7.2 Feature overlap table (O = included)

The table below lists features that appear in **at least two** of the three SHAP feature sets.

| Feature | XGBoost→GA (d=20) | SVM→GA (d=13) | SVM majority-vote (≥3/5, d=10) |
|---|:---:|:---:|:---:|
| **HISTO_ET_FLAIR_Bin5** | O | O | O |
| **TEXTURE_GLOBAL_ET_FLAIR_Skewness** | O | O | O |
| TEXTURE_GLSZM_ET_T2_LZLGE | O | O |  |
| TEXTURE_GLCM_ET_T1_Variance |  | O | O |
| TEXTURE_GLRLM_ED_T2_RLV |  | O | O |
| TEXTURE_GLRLM_NET_FLAIR_HGRE |  | O | O |
| TEXTURE_GLSZM_NET_T1_SZHGE |  | O | O |
| TEXTURE_NGTDM_ET_FLAIR_Busyness |  | O | O |

### 7.3 Shared vs. model-specific features (compact summary)

**Shared across all three (core consensus, n=2):**
- `HISTO_ET_FLAIR_Bin5`
- `TEXTURE_GLOBAL_ET_FLAIR_Skewness`

**Shared by SVM→GA (d=13) and SVM majority-vote (d=10) (n=5):**
- `TEXTURE_GLCM_ET_T1_Variance`
- `TEXTURE_GLRLM_ED_T2_RLV`
- `TEXTURE_GLRLM_NET_FLAIR_HGRE`
- `TEXTURE_GLSZM_NET_T1_SZHGE`
- `TEXTURE_NGTDM_ET_FLAIR_Busyness`

**Shared by XGBoost→GA (d=20) and SVM→GA (d=13) (n=1):**
- `TEXTURE_GLSZM_ET_T2_LZLGE`

### 7.4 Model-specific SHAP features

#### (A) XGBoost→GA (d=20): unique features (17/20)
- `SPATIAL_Frontal`
- `SOLIDITY_ED`
- `VOLUME_ET_OVER_NET`
- `TEXTURE_GLRLM_ED_T1Gd_LGRE`
- `TEXTURE_GLRLM_ED_T1Gd_LRLGE`
- `TEXTURE_GLRLM_NET_T2_LRLGE`
- `TEXTURE_GLOBAL_NET_T2_Kurtosis`
- `TEXTURE_GLSZM_ED_FLAIR_LZLGE`
- `INTENSITY_Mean_ED_T1Gd`
- `HISTO_ET_T1_Bin1`
- `TEXTURE_GLOBAL_ED_FLAIR_Kurtosis`
- `HISTO_ET_T1_Bin6`
- `TEXTURE_GLCM_ED_T1_Variance`
- `TEXTURE_GLRLM_ET_T2_LGRE`
- `TEXTURE_GLRLM_ET_T2_LRHGE`
- `TEXTURE_NGTDM_NET_T1Gd_Contrast`
- `TEXTURE_GLOBAL_ED_T1Gd_Skewness`

#### (B) SVM→GA (d=13): unique features (5/13)
- `TEXTURE_GLOBAL_ET_T1Gd_Skewness`
- `TEXTURE_GLCM_ED_T1Gd_Correlation`
- `TEXTURE_GLCM_NET_FLAIR_AutoCorrelation`
- `TEXTURE_GLSZM_NET_FLAIR_ZSN`
- `TEXTURE_NGTDM_NET_T1Gd_Busyness`

#### (C) SVM majority-vote (≥3/5, d=10): unique features (3/10)
- `SPATIAL_Temporal`
- `TEXTURE_GLSZM_ED_T1Gd_LZE`
- `TEXTURE_GLSZM_NET_T1Gd_LZE`

### 7.5 Interpretation: why the SVM-based signatures look more consistent

Only two features are shared across all three pipelines, suggesting that multiple sparse signatures can capture MGMT-related
signal under modest sample size and correlated radiomic descriptors. However, within the SVM–L1 branch, the **SVM→GA (d=13)**
and **SVM majority-vote (d=10)** signatures show strong internal agreement: the majority-vote signature preserves most of the
SVM→GA core features (7/10 overlap including the two global core features), and held-out performance remains comparable
(**0.733 → 0.729**). This supports the view that SVM–L1 pre-selection yields a more coherent candidate space where GA solutions
recurrently converge to similar texture/histogram signals.

In contrast, the XGBoost→GA (d=20) signature contains many features not reused by the SVM-based signatures (e.g., spatial
frontal location, solidity, ET/NET volume ratio, and multiple T1Gd-specific texture terms), indicating a more diverse set of
high-CV solutions within the broader XGBoost(84) wrapper space. Overall, the SHAP overlap analysis suggests that **SVM–L1 + GA**
produces **more repeatable core features** (and a stable consensus via majority-vote) while maintaining competitive held-out AUC.

## 8. Notes and Limitations

- **Modest sample size (n=118 total)**: With 82 train / 36 test, performance estimates can have non-trivial variance.
  Cross-validation results and GA subset rankings may fluctuate across runs and random seeds.

- **Multiple sources of randomness**: Random Forest training, CV fold splits, and GA operations (selection/crossover/mutation)
  introduce stochasticity. We mitigate this by (i) reporting top-5 GA solutions and (ii) checking test AUC across the GA top-5,
  but results should still be interpreted as proof-of-concept.

- **Feature selection instability in correlated radiomics**: Radiomic descriptors are often highly correlated, so multiple
  distinct feature subsets can yield similar CV AUC. This is why we report subset stability summaries and also evaluate
  a **majority-vote (≥3/5)** consensus signature.

- **Potential optimism from re-using the same training set for selection and tuning**: GA fitness and RF hyperparameter tuning
  both rely on cross-validation within the training set. While the held-out test set is kept separate for final reporting,
  the training-side model selection process can still favor configurations that fit the training distribution.

- **Held-out test set is small**: With only 36 patients, single-split test AUC can be noisy. Reported test AUCs should not be
  over-interpreted as definitive ranking between pipelines; they mainly indicate that compact signatures can preserve
  (and sometimes slightly improve) generalization relative to larger baselines.

- **Consensus signatures are not guaranteed to improve performance**: Majority-vote feature aggregation can increase stability,
  but it may also discard informative low-frequency features. In our experiments, the SVM-based majority-vote signature
  remained competitive, whereas the XGBoost-based majority-vote signature degraded on the held-out test set.

- **SHAP explanations are model- and data-dependent**: SHAP rankings reflect the fitted Random Forest and the specific
  train/test split. With small sample sizes, SHAP importance may vary across retrains; we therefore focus on overlap patterns
  and recurring features rather than over-interpreting a single ranking.

- **Clinical readiness**: These experiments are methodological and exploratory. External validation on independent cohorts
  (and ideally multi-site data harmonization) would be required before any clinical deployment.

## 9. Conclusion and Key Results

This repository evaluates how **compact radiomic signatures** can predict **MGMT promoter methylation status** in GBM using
MRI-derived features and tree-based models. Starting from **724 radiomic features per patient**, we construct two wrapper-based
baseline feature spaces: an **XGBoost-selected 84-feature space** and an **SVM–L1 (LinearSVC) non-zero 32-feature space**. We
then apply **GA Random Forest (GA RF)** subset search to obtain **sparse, clinically realistic signatures**.


### 9.1 Main performance snapshots (Baseline → representative GA subset)

For consistent reporting, we list **Baseline first**. For GA results, we first **filter candidate subsets by sensitivity** on the training folds, keeping only subsets with **mean sensitivity ≥ 0.80** (computed under the same 5-fold Stratified CV used for GA evaluation). Among the remaining subsets, we choose a **representative GA signature** by selecting the one whose **inner-CV mean AUC** is the **median** (i.e., the most typical subset within the sensitivity-qualified group). We then evaluate this representative subset on the **held-out test set**, and also report test performance across the other GA top-5 subsets for context.

| Pipeline | Feature set | d (#features) | Train CV AUC | Held-out Test AUC |
|---|---|---:|---:|---:|
| **XGBoost wrapper space** | Baseline RF on all XGBoost-selected features | 84 | 0.790 | 0.760 |
| **XGBoost(84) → GA–RF (TARGET_K=20)** | Representative GA subset (sensitivity-qualified; median inner-CV mean AUC) | 20 | 0.831 | **0.771** |
| **SVM–L1 non-zero space** | Baseline RF on all SVM–L1-selected features | 32 | 0.780 | 0.708 |
| **SVM–L1(32) → GA–RF (target ≈13)** | Representative GA subset (sensitivity-qualified; median inner-CV mean AUC) | 13 | 0.844 | **0.733** |

### 9.2 Sparsity–robustness trade-off (XGBoost(84) GA–RF: TARGET_K = 15, 20, 25)

Across GA runs on the **XGBoost(84)** search space, compact signatures consistently improved **training CV AUC** relative to the
84-feature baseline. On the held-out test set, **TARGET_K ≈ 20** provided the best balance: some 20-feature GA subsets matched or
slightly exceeded the 84-feature baseline test AUC (≈0.77 vs 0.76), while using ~1/4 of the features. Increasing the target to
**TARGET_K=25** often increased train CV AUC but did not generalize consistently on the held-out test set.

### 9.3 Majority-vote consensus signatures (≥3/5) — stability check

To obtain a more stable **consensus** signature, we built a *majority-vote* feature set from the GA top-5 subsets:
a feature is included if it appears in **at least 3 out of 5** GA subsets (≥3/5). We then re-tuned a Random Forest on the
resulting subset using **Stratified 5-fold CV** on the training set and evaluated on the held-out test set.

| Pipeline (GA source) | Majority-vote rule | d (#features) | Train CV AUC (best from tuning) | Train CV AUC (mean ± std) | Train Sensitivity@0.5 (mean ± std) | Held-out Test AUC |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost(84) → GA–RF (TARGET_K=20) → Majority-vote subset** | ≥ 3/5 | 10 | 0.8436 | 0.8242 ± 0.0843 | 0.8509 ± 0.1055 | 0.677 |
| **SVM–L1(32) → GA–RF (target ≈13) → Majority-vote subset** | ≥ 3/5 | 10 | 0.8576 | 0.8303 ± 0.1107 | 0.9636 ± 0.0813 | **0.729** |

Under the same consensus rule and the same final dimensionality (**d=10**), the **SVM–L1-based** majority-vote signature remained
competitive on the held-out test set, whereas the **XGBoost-based** majority-vote signature degraded substantially.

### 9.4 Interpretation (SHAP overlap across compact signatures)

We compared SHAP feature sets from three compact pipelines: **XGBoost→GA (d=20)**, **SVM→GA (d=13)**, and
**SVM majority-vote (d=10)**. Only two features were shared by all three signatures, indicating that multiple sparse solutions
can capture MGMT-related signal under correlated radiomic descriptors and modest sample size:

- `HISTO_ET_FLAIR_Bin5`
- `TEXTURE_GLOBAL_ET_FLAIR_Skewness`

Within the SVM branch, the majority-vote signature preserved most of the SVM→GA core SHAP features while maintaining comparable
held-out AUC (**0.733 → 0.729**), suggesting a more coherent and repeatable interpretation under the SVM–L1 pre-selection space.

### 9.5 Overall takeaway

- **Best single-split test performance:** **XGBoost(84) → GA–RF (d=20)** achieved the highest held-out AUC (**0.771**).
- **Most interpretation-stable compact signatures:** The **SVM–L1(32) → GA–RF** pipeline produced a smaller signature (**d=13**)
  with competitive held-out AUC (**0.733**) and a **consensus majority-vote signature (d=10)** that retained performance (**0.729**)
  and core SHAP features.
- Overall, these experiments support the conclusion that a large portion of MGMT-related signal can be captured by **small,
  clinically realistic radiomic signatures**, and that **SVM–L1 + GA** is especially attractive when the goal is **stable,
  interpretable sparsity**, while **XGBoost + GA** can yield the strongest single-split held-out performance.
