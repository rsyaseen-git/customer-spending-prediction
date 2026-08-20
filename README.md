# Customer Spending Prediction Using Machine Learning

## ML Workshop Project

### Objective
Predict a customer's total spending using demographic and purchasing-behavior features.

### Problem Statement
**Customer Spending Prediction**

This project demonstrates a complete regression workflow using the algorithms covered in the ML workshop.

### Workflow
1. Generate/load customer data
2. Inspect and clean the data
3. Perform exploratory data analysis
4. Create the `Total_Spending` target
5. Split data into training and testing sets
6. Train multiple regression models
7. Evaluate using MAE, RMSE and R²
8. Compare models and select the best model

### Models
- Linear Regression (single-feature baseline)
- Multiple Linear Regression
- KNN Regressor

### Results from the offline experiment
| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression (1 feature) | 176.58 | 223.00 | 0.571 |
| Multiple Linear Regression | 68.91 | 87.60 | **0.934** |
| KNN Regressor | 138.56 | 178.61 | 0.725 |

Multiple Linear Regression performed best in this experiment.

### Important ML practice
The target is not reconstructed from the same spending variables supplied to the model. This avoids target leakage and makes the prediction task meaningful.

### Dataset note
The offline build uses a synthetic customer dataset so that the complete project can run without an internet connection. The feature structure is designed for the workshop problem and includes demographic and purchasing-behavior variables.

### Run locally
```bash
pip install -r requirements.txt
python run_project.py
```

The script creates the dataset, trains the models, prints the metrics, and saves result plots in `results/`.

### Project structure
```text
customer-spending-prediction/
├── data/
│   └── .gitkeep
├── notebooks/
│   └── customer_spending_prediction.ipynb
├── results/
│   └── README.md
├── run_project.py
├── requirements.txt
├── .gitignore
└── README.md
```
