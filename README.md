# Case Study: Exploratory Data Analysis and Visualization of an E-Commerce Purchasing Intention Dataset

[![Course](https://img.shields.io/badge/Course-U21ADP05--EDA--Visualization-navy.svg)](#course-details)
[![Department](https://img.shields.io/badge/Department-AI%20%26%20Data%20Science-blue.svg)](#student-details)
[![Institution](https://img.shields.io/badge/Institution-KPR--Institute--of--Engineering--and--Technology-red.svg)](#student-details)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic--Use--Only-lightgrey.svg)](#)

---

## 📌 Title Page & Student Details
* **Course Code & Title:** U21ADP05 - Exploratory Data Analysis and Visualization
* **Assignment Title:** Assignment II (Case Study): Exploratory Data Analysis and Visualization of a Real-World Dataset
* **Student Name:** Jayavanth Palanisamy
* **Roll Number:** `<<ROLL NO>>`
* **Department:** Artificial Intelligence and Data Science
* **Institution:** KPR Institute of Engineering and Technology, Coimbatore, Tamil Nadu
* **Year / Semester:** III-YR / V Semester (B.Tech. AI & DS)
* **Academic Year:** 2026-2027 (Odd Semester)
* **Submission Date:** 15 October 2026
* **Public GitHub Repository:** `https://github.com/jayavanthpalanisamy-cmd/EDA`

---

## 📖 Executive Summary & Problem Relevance
In modern online retail, understanding user navigation pathways and predicting visitor purchase intention in real time is fundamental for **Conversion Rate Optimization (CRO)**, cart abandonment intervention, and targeted merchandising. Typically, only **2% to 4%** of e-commerce sessions finalize in an actual sale, leading to extreme class skewness and pronounced zero-inflation in duration and engagement metrics.

This case study executes a comprehensive, publication-grade exploratory data analysis, parametric and non-parametric statistical distribution modeling, and deep learning classification on the **Online Shoppers Purchasing Intention Dataset** (12,330 visitor sessions, 18 raw features, UCI Machine Learning Repository ID: 468).

Following **Ben Fry's Seven Stages of Visualizing Data** (*Acquire, Parse, Filter, Mine, Represent, Refine, Interact*), the study examines browsing volume, page-level dwell durations, Google Analytics indicators, and temporal factors to uncover core behavioral patterns. It addresses the **5.46:1 class imbalance** via cost-sensitive learning and threshold calibration strictly on validation data, benchmarking **Cost-Sensitive Logistic Regression**, **Random Forest**, and a **PyTorch Deep Tabular Multi-Layer Perceptron (MLP)**.

---

## 📊 Dataset Overview
* **Dataset Name:** Online Shoppers Purchasing Intention Dataset
* **Source:** UCI Machine Learning Repository ([Dataset ID: 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset))
* **Original Study:** Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y. (2019). *Neural Computing and Applications*, 31(10), 6893–6908.
* **Total Instances:** 12,330 individual visitor sessions (collected over a full 1-year period)
* **Total Raw Features:** 18 (10 numerical continuous/discrete, 7 categorical nominal/binary, 1 binary target `Revenue`)
* **Engineered Features:** 6 domain-driven metrics (`Total_PageViews`, `Total_Duration`, `Avg_Duration_Per_Page`, `Bounce_To_Exit_Ratio`, `PageValue_Per_PageView`, `High_Intent_Indicator`)
* **Missing Values:** 0 missing values across all features
* **Exact Duplicate Sessions:** 125 (0.97%) — verified as authentic zero-engagement bounce events and retained to avoid sampling distortion.
* **Target Variable:** `Revenue` (Boolean)
  * `False` (No Purchase): **10,422 sessions (84.53%)**
  * `True` (Purchase Finalized): **1,908 sessions (15.47%)**
  * **Class Imbalance Ratio:** **5.46 : 1**

---

## 📈 Probability Distribution Modeling & Statistical Tests (Rubric CO4)

### 1. Discrete Distribution Modeling (ProductRelated Pageviews)
* **Empirical Moments:** Sample Mean $\mu = 31.73$, Sample Variance $\sigma^2 = 1,977.91$
* **Dispersion Ratio:** $\sigma^2 / \mu = 62.33 \gg 1$ (**Massive Overdispersion**)
* **Model Comparison:**
  * **Poisson Model:** $\text{AIC} = 529,386.1$ (Fails completely; assumes equi-dispersion $\text{Var} = \text{Mean}$)
  * **Negative Binomial Model:** $\text{AIC} = 112,011.9$ (Superior fit; captures compound Poisson-Gamma variance)
  * **$\Delta \text{AIC}$:** **$417,374.2$** in favor of Negative Binomial.

### 2. Continuous Distribution Modeling (Non-Zero ProductRelated_Duration)
Evaluated via Maximum Likelihood Estimation (MLE) and Kolmogorov-Smirnov (KS) goodness-of-fit diagnostics:
* **Log-Normal Distribution:** $\text{AIC} = 187,817.2$, $\text{KS } D = 0.0487$, $p = 2.87 \times 10^{-24}$ (**Best Continuous Family**)
* **Gamma Distribution:** $\text{AIC} = 187,913.5$, $\text{KS } D = 0.0455$, $p = 2.67 \times 10^{-21}$ (Close competitor)
* **Exponential Distribution:** $\text{AIC} = 188,648.5$, $\text{KS } D = 0.0942$, $p = 7.27 \times 10^{-90}$
* **Normal (Gaussian) Distribution:** $\text{AIC} = 208,224.2$, $\text{KS } D = 0.2571$, $p = 0.0$ (**Catastrophic failure; extreme right-tail departure**)

### 3. Non-Parametric Hypothesis Testing
* **Mann-Whitney U Test (Continuous Drivers vs. Revenue):**
  * `PageValue_Per_PageView`: Rank-biserial correlation $r_{rb} = -0.7296$ ($p < 0.0001$, Large effect)
  * `PageValues`: $r_{rb} = -0.6974$ ($p < 0.0001$, Large effect)
  * `ExitRates`: $r_{rb} = +0.3842$ ($p < 0.0001$, Medium effect)
* **Chi-Square Test of Independence (Categorical Drivers vs. Revenue):**
  * `High_Intent_Indicator`: $\chi^2 = 4,323.5$, Cramér's $V = 0.5922$ (Strong association)
  * `Month`: $\chi^2 = 384.2$, Cramér's $V = 0.1767$ (Moderate seasonal association; November conversion = 25.4% vs. February = 1.6%)
  * `VisitorType`: $\chi^2 = 135.0$, Cramér's $V = 0.1047$ (New visitors = 24.9% vs. Returning = 13.9%)

---

## 🏆 Key Machine Learning Benchmarking Results (Test Set: 1,850 Sessions)
All models were trained strictly on the **Training partition (8,630 sessions)** and had decision thresholds calibrated exclusively on the **Validation partition (1,850 sessions)**. Metrics below represent unseen performance on the **Held-Out Test partition (1,850 sessions)**:

| Predictive Model | Threshold Setting | Decision Threshold ($\tau$) | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (Average Precision) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Default Threshold | $0.50$ | 0.8649 | 0.5437 | 0.7832 | 0.6418 | 0.9191 | 0.7009 |
| **Logistic Regression** | **Validation-Tuned** | **$0.83$** | **0.8957** | **0.6883** | **0.5944** | **0.6379** | **0.9191** | **0.7009** |
| **Random Forest Classifier** | Default Threshold | $0.50$ | 0.9038 | 0.7500 | 0.5664 | 0.6454 | 0.9311 | 0.7506 |
| **Random Forest Classifier** | **Validation-Tuned** | **$0.29$** | **0.8849** | **0.5984** | **0.7762** | **0.6758** | **0.9311** | **0.7506** |
| **PyTorch Tabular Deep MLP** | Default Threshold | $0.50$ | 0.8551 | 0.5196 | 0.8357 | 0.6408 | 0.9286 | 0.7320 |
| **PyTorch Tabular Deep MLP** | **Validation-Tuned** | **$0.63$** | **0.8773** | **0.5774** | **0.7692** | **0.6597** | **0.9286** | **0.7320** |

### 🔍 Honest Comparative Takeaway:
* **Best Overall Discriminator:** **Random Forest** achieved the highest PR-AUC (**0.7506**) and ROC-AUC (**0.9311**), reaching an optimal F1-score of **0.6758** at $\tau = 0.29$.
* **Deep Learning Performance:** The **PyTorch Tabular Deep MLP** achieved competitive performance (**PR-AUC = 0.7320**, **ROC-AUC = 0.9286**, **F1 = 0.6597** at $\tau = 0.63$), outperforming Logistic Regression by capturing non-linear interactions between page values and exit rates.
* **Why Random Forest edged out Deep MLP:** Tabular decision trees create orthogonal, axis-aligned partition boundaries (e.g. `PageValues == 0`), which perfectly align with zero-inflated e-commerce features compared to continuous sigmoidal activations.

---

## 📁 Repository Structure
```text
d:\EDA\
├── .gitignore                                 <- Git ignore rules (caches, IDE, checkpoints)
├── requirements.txt                           <- Python environment dependencies
├── README.md                                  <- Executive summary, results, and execution guide
├── data\
│   └── online_shoppers_intention.csv          <- Authentic UCI 468 dataset (12,330 sessions)
├── notebooks\
│   └── online_shoppers_eda_dl.ipynb           <- Fully executed Jupyter notebook with all outputs
├── figures\                                   <- 24 publication PNG figures (150 DPI) + 1 interactive HTML
│   ├── fig01_class_balance.png                <- Class distribution & imbalance ratio (5.46:1)
│   ├── fig02_fry_stages_pipeline.png          <- Ben Fry's 7 stages mapped to project
│   ├── fig03_univariate_pageviews.png         <- Pageview count distributions & boxplots
│   ├── fig04_univariate_durations.png         <- Dwell time distributions & zero-inflation
│   ├── fig05_univariate_rates.png             <- Google Analytics rates & page values
│   ├── fig06_discrete_distribution_fits.png   <- Poisson vs. Negative Binomial (overdispersion)
│   ├── fig07_continuous_distribution_fits.png <- Normal vs. Log-Normal vs. Gamma vs. Exponential
│   ├── fig08_qq_plots.png                     <- Quantile-Quantile diagnostics of candidate fits
│   ├── fig09_bivariate_conversion_by_month.png<- Monthly browsing volume vs conversion rate
│   ├── fig10_bivariate_conversion_by_visitor_weekend.png <- Visitor type & weekend conversion rates
│   ├── fig11_correlation_matrix_heatmap.png   <- Spearman rank correlation matrix
│   ├── fig12_pairplot_key_features.png        <- Bivariate pairplot of top discriminators
│   ├── fig13_mann_whitney_effect_sizes.png    <- Non-parametric Mann-Whitney U effect sizes
│   ├── fig14_cramers_v_heatmap.png            <- Cramér's V categorical associations
│   ├── fig15_multivariate_interaction_pagevalue_exit.png <- 4x4 PageValue & ExitRate conversion heatmap
│   ├── fig16_interactive_scatter_preview.png  <- Static preview of interactive Plotly explorer
│   ├── fig17_interactive_session_explorer.html<- Standalone interactive Plotly HTML chart (Fry Stage 7)
│   ├── fig18_training_validation_loss.png     <- Deep MLP train vs. validation loss curve
│   ├── fig19_training_validation_accuracy.png <- Deep MLP classification accuracy curve
│   ├── fig20_hyperparameter_tuning_grid.png   <- 10-architecture validation F1 tuning chart
│   ├── fig21_confusion_matrices_comparison.png<- Normalized test confusion matrices comparison
│   ├── fig22_roc_curves_comparison.png        <- Test set ROC curves (AUC comparison)
│   ├── fig23_pr_curves_comparison.png         <- Test set PR curves (PR-AUC comparison)
│   ├── fig24_prediction_probability_distribution.png <- Calibrated threshold score density
│   └── fig25_permutation_feature_importance.png <- Test set permutation importance (PR-AUC drop)
└── report\
    ├── Case_Study_EDA_and_Visualization.docx  <- Formatted Word report (A4, page borders, styled tables)
    ├── Case_Study_EDA_and_Visualization.pdf   <- Exported 42-page formal PDF report (Word COM)
    ├── summary_numbers.json                   <- Comprehensive JSON record of all verified metrics
    └── test_metrics.csv                       <- Exported test benchmarking table
```

---

## 🚀 How to Run & Reproduce

### 1. Environment Setup
Clone your repository and install the verified dependencies:
```bash
git clone https://github.com/jayavanthpalanisamy-cmd/EDA.git
cd EDA
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
Open and run the pre-executed notebook:
```bash
jupyter notebook notebooks/online_shoppers_eda_dl.ipynb
```
* **Deterministic Guarantee:** Fixed global seeds (`SEED = 42` across Python, NumPy, and PyTorch CPU) guarantee **exact bit-for-bit numerical reproducibility**. Re-running will yield identical metrics to `test_metrics.csv` and `summary_numbers.json`.

### 3. Open the Interactive Visualization
Open the interactive Plotly multi-dimensional session explorer directly in any web browser:
```bash
# Double-click or open directly in your favorite browser:
figures/fig17_interactive_session_explorer.html
```

---

## 📚 Literature Review & References

1. **Breiman, L. (2001).** Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
2. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).** SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357. https://doi.org/10.1613/jair.953
3. **Fry, B. (2008).** *Visualizing Data: Exploring and Explaining Data with the Processing Environment*. O'Reilly Media.
4. **He, H., & Garcia, E. A. (2009).** Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239
5. **Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019).** PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32, 8026–8037.
6. **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011).** Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
7. **Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y. (2019).** Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks. *Neural Computing and Applications*, 31(10), 6893–6908. https://doi.org/10.1007/s00521-018-3523-0

---
*Prepared for Course U21ADP05 – Exploratory Data Analysis and Visualization, Department of AI & DS, KPR Institute of Engineering and Technology.*
