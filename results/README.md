# Results

Run `python run_project.py` to generate the CSV metrics and plots in this folder.

## Verified Regression Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Multiple Linear Regression | 73.950 | 92.558 | 0.955 |
| KNN Regressor | 169.932 | 217.179 | 0.750 |
| Linear Regression (Income only) | 202.975 | 256.959 | 0.650 |

**Best regression model: Multiple Linear Regression**

## Verified Classification Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.920 | 0.922 | 0.914 | 0.918 | 0.985 |
| KNN Classifier | 0.848 | 0.852 | 0.836 | 0.844 | 0.918 |
| Bernoulli Naive Bayes | 0.810 | 0.785 | 0.845 | 0.814 | 0.901 |

**Best classification model: Logistic Regression**

The dataset and result files are generated reproducibly by `run_project.py` for the fully offline workshop project.
