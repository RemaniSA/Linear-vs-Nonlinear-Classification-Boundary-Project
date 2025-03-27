# Linear vs Nonlinear Classification Boundary Project (2024/25)

This project was developed as part of the MSc Mathematical Trading and Finance programme at Bayes Business School (formerly Cass). It investigates whether a linear or nonlinear decision boundary best classifies Coronary Heart Disease (CHD) in a high-risk male population from the Western Cape, South Africa.

## Overview

Using a dataset of 462 patients and nine clinical, lifestyle, and demographic features, the analysis compares the performance of linear and nonlinear classification models. The objective is twofold:

1. Identify the model with the best classification performance.
2. Determine whether the underlying decision boundary is linear.

## Methodology

- **Exploratory Data Analysis**: Distribution plots, boxplots, PCA, and pairplots were used to assess class separability and outlier impact.
- **Baseline Model**: Logistic Regression with Ridge Penalty was selected as a regularised linear classifier and benchmark.
- **Model Comparison**: Eleven classifiers were tested using 10-fold cross-validation and tuned via grid search.

Models tested:

- Logistic Regression (L1, L2, None)
- Linear Discriminant Analysis (LDA)
- Support Vector Machine (Linear Kernel)
- Quadratic Discriminant Analysis (QDA)
- Naive Bayes
- Decision Tree
- k-Nearest Neighbours (CV-Optimised)
- AdaBoost
- Gradient Boosting

## Results Summary

| Model            | F1-Score | ROC-AUC | Precision | Recall | Decision Boundary |
|------------------|----------|---------|-----------|--------|--------------------|
| **LDA**          | **66%**  | 0.81    | 63%       | 69%    | Linear (Best)      |
| Logistic (L2)    | 62%      | 0.82    | 61%       | 63%    | Linear             |
| SVM              | 62%      | 0.80    | 61%       | 63%    | Linear             |
| AdaBoost         | 58%      | 0.79    | 70%       | 50%    | Nonlinear          |
| Gradient Boost   | 56%      | 0.76    | 56%       | 56%    | Nonlinear          |
| Decision Tree    | 46%      | 0.59    | 40%       | 53%    | Nonlinear          |
| kNN (CV)         | 46%      | 0.71    | 52%       | 41%    | Nonlinear          |

> LDA achieved the highest F1-Score and recall, suggesting a linear boundary is most effective for this dataset.

## Sensitivity Analysis

To test robustness, the models were re-evaluated in two modified settings:

- **Sans Alcohol**: Alcohol was removed from the feature set. Naive Bayes outperformed others slightly (66% F1), but linear models remained stable.
- **Sans Outliers**: All outliers were removed using IQR filtering. Performance dropped across models. QDA led this scenario with a 59% F1-Score, suggesting the original linear boundary may be influenced by edge cases.

## Repository Structure

```
Linear-vs-Nonlinear-Classification-Boundary-Project/
├── chd_classification.py         # Python script
├── heart-disease.csv             # Dataset
├── Report.pdf                    # Final report
├── Task.pdf                      # Coursework brief
├── Images/                       # Plots and coursework brief in Images
├── requirements.txt              # Project requirements
└── README.md
```

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/RemaniSA/Linear-vs-Nonlinear-Classification-Boundary-Project.git
cd Linear-vs-Nonlinear-Classification-Boundary-Project
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run `chd_classification.py`

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn

See `requirements.txt` for exact versions.

## Author

Shaan Ali Remani

# Coursework Brief
 
![Individual Cousework for Machine Learning for Quantitative Professionals p1](https://github.com/RemaniSA/Linear-vs-Nonlinear-Classification-Boundary-Project/blob/main/images/Task_Page1.jpg)

![Individual Cousework for Machine Learning for Quantitative Professionals p2](https://github.com/RemaniSA/Linear-vs-Nonlinear-Classification-Boundary-Project/blob/main/images/Task_Page2.jpg)
