# Astronomical Object Classification (Broad Cosmic Object Categorization)

A basic machine-learning project for classifying astronomical objects into **stars, galaxies, and quasars (QSO)** using photometric and observational features from the **SDSS17 stellar classification dataset**.

## Project Overview

Astronomical object classification is a fundamental task in astronomy. Large astronomical surveys contain enormous numbers of observations, making machine-learning methods useful for identifying and categorizing different types of objects.

In this project, supervised and unsupervised machine-learning techniques are applied to the SDSS17 dataset. The supervised models classify objects into **STAR, GALAXY, and QSO**, while K-Means clustering is used as an exploratory unsupervised-learning exercise on the STAR population.

The project is designed as a **basic ML/data-science project**, with emphasis on understanding the complete machine-learning workflow rather than developing a research-grade astronomical classifier.

## Objectives

- Explore and understand the SDSS17 astronomical-object dataset.
- Clean invalid and extreme observations.
- Perform exploratory data analysis (EDA).
- Select physically/observationally meaningful numerical features.
- Compare different supervised classification algorithms.
- Evaluate classification performance using multiple metrics.
- Examine Random Forest feature importance.
- Use learning curves to investigate model generalization.
- Apply K-Means clustering to the STAR population as an exploratory unsupervised-learning analysis.

## Dataset

The project uses the **Stellar Classification Dataset - SDSS17**, containing observations of:

- **Stars**
- **Galaxies**
- **Quasars (QSO)**

The dataset contains approximately **100,000 observations** and includes photometric measurements, sky coordinates, redshift, and observational metadata.

### Dataset Source

Kaggle: [Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)

## Features Used

For the final machine-learning analysis, the following features are used:

| Feature | Description |
|---|---|
| `u` | u-band photometric magnitude |
| `g` | g-band photometric magnitude |
| `r` | r-band photometric magnitude |
| `i` | i-band photometric magnitude |
| `z` | z-band photometric magnitude |
| `redshift` | Redshift measurement |
| `alpha` | Right ascension / sky coordinate |
| `delta` | Declination / sky coordinate |

Identifier and observational metadata columns are excluded from the final feature set because they are not appropriate as general physical/observational predictors for the classification task.

## Project Workflow

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ├── Handle -9999 sentinel values
   └── Identify extreme/anomalous observations using LOF
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Supervised ML
   ├── Logistic Regression
   ├── Decision Tree
   └── Random Forest
   ↓
Model Evaluation
   ├── Accuracy
   ├── Precision
   ├── Recall
   ├── F1-score
   └── Confusion Matrix
   ↓
Feature Importance & Learning Curves
   ↓
K-Means Clustering of STAR objects
```

## Data Cleaning and Outlier Detection

The dataset contains `-9999` values that are treated as invalid/missing sentinel values in the relevant photometric features. These values are replaced with `NaN` and the corresponding observations are removed.

After this cleaning step, **Local Outlier Factor (LOF)** is used to identify highly anomalous observations. This step was particularly useful because a small number of extreme observations strongly affected the scale and readability of exploratory scatter plots.

LOF is used here as an exploratory data-cleaning/outlier-detection method. An observation identified as an LOF outlier is considered anomalous relative to its local neighbourhood; this does not by itself establish that the observation is physically incorrect.

## Supervised Machine Learning

Three classification models are compared.

### 1. Logistic Regression

Logistic Regression is used as a simple baseline classification model.

Because Logistic Regression is sensitive to feature scale, `StandardScaler` is applied to the training data. The scaler is fitted only on the training set and then used to transform the test set.

### 2. Decision Tree

A Decision Tree is used to provide a nonlinear classification model. Unlike Logistic Regression, feature scaling is not required for the tree-based model.

### 3. Random Forest

Random Forest combines multiple decision trees and is used as the main ensemble model.

The Random Forest provides both strong classification performance and an estimate of impurity-based feature importance.

## Model Evaluation

The models are evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion matrices**

The project also examines **learning curves** using stratified 5-fold cross-validation to investigate training and validation performance as the training-set size increases.

### Classification Results

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| Logistic Regression | **95.80%** | **0.95** |
| Decision Tree | **96.53%** | **0.96** |
| Random Forest | **97.95%** | **0.98** |

Among the three models, **Random Forest performs best** on the reported test-set metrics.

## Random Forest Feature Importance

The Random Forest feature importances indicate that **redshift** is the most important feature for the fitted model.

The approximate impurity-based feature importances are:

| Feature | Importance |
|---|---:|
| `redshift` | 0.6016 |
| `z` | 0.0967 |
| `i` | 0.0774 |
| `u` | 0.0772 |
| `g` | 0.0712 |
| `r` | 0.0444 |
| `alpha` | 0.0158 |
| `delta` | 0.0156 |

These values describe how heavily the fitted Random Forest relied on the features when making its decisions. They should not be interpreted as evidence of physical causation.

## Unsupervised Learning: K-Means

K-Means clustering is applied **only to objects already labelled as STAR**.

The purpose is exploratory: to investigate whether the STAR population contains naturally separated groups in the selected feature space.

The number of clusters is selected automatically using the **silhouette score**. Values of `k` from 2 through 9 are tested, and the value producing the highest silhouette score is selected.

For the current dataset:

- **Best k = 2**
- **Silhouette score = 0.3439**

The two clusters are therefore data-driven groups within the STAR sample. They should **not** be interpreted as confirmed physical stellar subclasses because the dataset does not provide ground-truth labels for stellar subtypes.

## Key Findings

- All three supervised models achieve high classification accuracy.
- Random Forest gives the best overall performance among the tested models.
- Random Forest achieves approximately **97.95% test accuracy** and a **0.98 macro F1-score**.
- Redshift is the dominant feature according to Random Forest impurity-based feature importance.
- The class distribution is moderately imbalanced rather than severely imbalanced, and the per-class metrics do not indicate a major minority-class performance problem; therefore, SMOTE was not considered necessary.
- K-Means identifies **2 clusters** as the best configuration among the tested values of `k = 2–9` according to the silhouette score.
- The K-Means analysis is exploratory and does not establish physical stellar subclasses.

## Limitations

This project is intended as a basic machine-learning study and has several limitations:

- The classification task depends on the features available in the SDSS17 dataset.
- LOF identifies statistical anomalies but does not determine whether an astronomical observation is physically incorrect.
- The K-Means clusters do not have labelled stellar-subtype ground truth and therefore cannot be directly assigned physical meanings.
- Random Forest feature importance indicates model reliance on features, not causal relationships.
- Hyperparameter optimization is intentionally limited because the project focuses on understanding the basic ML workflow.
- The results should not be interpreted as a replacement for detailed astronomical analysis or a research-grade astronomical classification pipeline.

## Libraries Used

The project uses Python and common scientific/ML libraries, including:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd astronomical-object-classification
```

### 2. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Open the notebook

```bash
jupyter notebook "Astronomical Object Classification.ipynb"
```

Alternatively, the notebook can be opened directly in **Google Colab**.

### 4. Dataset

Place `star_classification.csv` in the same directory as the notebook if the CSV is included in the repository.

If the dataset is not included because of repository-size considerations, download it from the Kaggle source linked above and place the CSV in the expected location.

## Project Structure

A recommended repository structure is:

```text
astronomical-object-classification/
│
├── Astronomical Object Classification.ipynb
├── star_classification.csv
├── README.md
└── .gitignore
```

## Conclusion

This project demonstrates a complete introductory machine-learning workflow for astronomical object classification, from exploratory data analysis and cleaning through supervised classification, evaluation, feature-importance analysis, learning curves, and exploratory clustering.

Among the tested supervised models, **Random Forest provides the strongest overall classification performance**, while the K-Means analysis suggests that **two clusters provide the best silhouette score among the tested cluster counts** for the STAR population.

---

### Author

**Om Jitendrabhai Patel**

This project was developed as a basic machine-learning/data-science semester project.
