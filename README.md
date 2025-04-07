# Coronary Heart Disease Classifier Project

This project was developed as part of the MSc Mathematical Trading and Finance programme at Bayes Business School (formerly Cass). It investigates whether a linear or nonlinear classifier best models Coronary Heart Disease (CHD) risk in a high-risk male population from the Western Cape, South Africa.

## Problem

The aim of the project was to classify CHD risk using clinical and lifestyle data from 462 patients, drawing on nine input variables. Eleven classification algorithms were tested, including logistic regression, LDA, SVMs, decision trees, and boosting methods. The central question was whether a nonlinear boundary significantly improves classification performance or if simpler models suffice in practice.

## My Reflections

This project reinforced the idea that the use case determines the metric; are we screening a population broadly (precision), or ruling people out with certainty (recall)? While it's tempting to reach for ensemble or deep learning models, I found that simpler methods (LDA, Logistic Regression) can outperform more complex models, especially when the decision boundary is inherently linear. One of the biggest takeaways was learning that model robustness often hinges more on boundary geometry than model class.

## Methods

- Exploratory Data Analysis: PCA, boxplots, class distribution analysis, pairplots
- Baseline Model: Logistic Regression with Ridge penalty
- Model Comparison: 11 classifiers evaluated using 10-fold cross-validation
- Sensitivity Analysis:
  - Removed alcohol consumption feature
  - Filtered outliers via IQR method

## Classifiers Tested

- Logistic Regression (L1, L2, None)
- Linear Discriminant Analysis (LDA)
- Support Vector Machine (Linear Kernel)
- Quadratic Discriminant Analysis (QDA)
- Naive Bayes
- Decision Tree
- k-Nearest Neighbours (CV-Optimised)
- AdaBoost
- Gradient Boosting

## Repository Structure

```
Linear-vs-Nonlinear-Classification-Boundary-Project/
├── datasets/
├── images/
├── .gitignore
├── README.md
├── Report.pdf
├── Task.pdf
├── chd_classification.py
├── requirements.txt
```

## Summary of Results

| Model            | F1-Score | ROC-AUC | Precision | Recall | Decision Boundary |
|------------------|----------|---------|-----------|--------|--------------------|
| **LDA**          | **66%**  | 0.81    | 63%       | 69%    | Linear (Best)      |
| Logistic (L2)    | 62%      | 0.82    | 61%       | 63%    | Linear             |
| SVM              | 62%      | 0.80    | 61%       | 63%    | Linear             |
| AdaBoost         | 58%      | 0.79    | 70%       | 50%    | Nonlinear          |
| Gradient Boost   | 56%      | 0.76    | 56%       | 56%    | Nonlinear          |
| Decision Tree    | 46%      | 0.59    | 40%       | 53%    | Nonlinear          |
| kNN (CV)         | 46%      | 0.71    | 52%       | 41%    | Nonlinear          |

## Requirements

```bash
pip install -r requirements.txt
```

See `requirements.txt` for package versions.

## How to Run

```bash
git clone https://github.com/RemaniSA/Linear-vs-Nonlinear-Classification-Boundary-Project.git
cd Linear-vs-Nonlinear-Classification-Boundary-Project
python chd_classification.py
```

## Further Reading

- James, Witten, Hastie & Tibshirani: *An Introduction to Statistical Learning*
- Hastie, Tibshirani & Friedman: *The Elements of Statistical Learning*
- Kuhn & Johnson: *Applied Predictive Modeling*

## Author

- Shaan Ali Remani

---

### Connect

- [LinkedIn](https://www.linkedin.com/in/shaan-ali-remani)  
- [GitHub](https://github.com/RemaniSA)
