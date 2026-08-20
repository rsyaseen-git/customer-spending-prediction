import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. Reproducible offline customer dataset
n = 2240
age = np.random.randint(25, 76, n)
income = np.clip(np.random.normal(60000, 25000, n), 15000, 120000).round(0)
recency = np.random.randint(1, 100, n)
deals = np.random.poisson(3, n)
web = np.random.poisson(5, n)
catalog = np.random.poisson(4, n)
store = np.random.poisson(6, n)
visits = np.random.poisson(5, n)
kids = np.random.randint(0, 3, n)
teens = np.random.randint(0, 3, n)
complaint = np.random.binomial(1, 0.15, n)
response = np.random.binomial(1, 0.25, n)
education = np.random.choice(["Basic", "Graduation", "Master", "PhD"], n, p=[0.12, 0.55, 0.25, 0.08])
marital = np.random.choice(["Single", "Married", "Together", "Divorced"], n)

spending = (350 + 0.014*income + 7*age - 2.2*recency + 24*deals + 42*web
            + 34*catalog + 50*store + 8*visits - 70*kids - 45*teens
            + 180*response - 90*complaint + np.random.normal(0, 90, n))
spending = np.maximum(spending, 100).round(2)

df = pd.DataFrame({
    "Age": age, "Income": income, "Education": education,
    "Marital_Status": marital, "Kidhome": kids, "Teenhome": teens,
    "Recency": recency, "NumDealsPurchases": deals,
    "NumWebPurchases": web, "NumCatalogPurchases": catalog,
    "NumStorePurchases": store, "NumWebVisitsMonth": visits,
    "Complaint": complaint, "Response": response,
    "Total_Spending": spending
})
df.to_csv("data/customer_spending.csv", index=False)

features = ["Income", "Age", "Recency", "NumDealsPurchases", "NumWebPurchases",
            "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth",
            "Kidhome", "Teenhome", "Complaint", "Response",
            "Education", "Marital_Status"]
X = df[features]
y = df["Total_Spending"]
categorical = ["Education", "Marital_Status"]
numeric = [c for c in features if c not in categorical]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

# 2. EDA
plt.figure(figsize=(8, 5))
plt.hist(y, bins=30)
plt.xlabel("Total Spending")
plt.ylabel("Number of Customers")
plt.title("Customer Spending Distribution")
plt.tight_layout()
plt.savefig("results/spending_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df["Income"], df["Total_Spending"], alpha=0.5)
plt.xlabel("Income")
plt.ylabel("Total Spending")
plt.title("Income vs Total Spending")
plt.tight_layout()
plt.savefig("results/income_vs_spending.png", dpi=150)
plt.close()

# 3. Regression models
multi_pre = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])
income_pre = ColumnTransformer([("num", StandardScaler(), ["Income"])])

regression_models = {
    "Linear Regression (Income only)": Pipeline([
        ("preprocessor", income_pre), ("model", LinearRegression())
    ]),
    "Multiple Linear Regression": Pipeline([
        ("preprocessor", multi_pre), ("model", LinearRegression())
    ]),
    "KNN Regressor": Pipeline([
        ("preprocessor", multi_pre), ("model", KNeighborsRegressor(n_neighbors=7))
    ])
}

regression_rows = []
regression_predictions = {}
for name, model in regression_models.items():
    train_x = X_train[["Income"]] if "Income only" in name else X_train
    test_x = X_test[["Income"]] if "Income only" in name else X_test
    model.fit(train_x, y_train)
    pred = model.predict(test_x)
    regression_predictions[name] = pred
    regression_rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred)
    })

regression_metrics = pd.DataFrame(regression_rows).sort_values("R2", ascending=False)
regression_metrics.to_csv("results/regression_metrics.csv", index=False)
print("\nREGRESSION RESULTS")
print(regression_metrics.round(3).to_string(index=False))

plt.figure(figsize=(9, 5))
plt.bar(regression_metrics["Model"], regression_metrics["RMSE"])
plt.ylabel("RMSE")
plt.title("Regression Model Comparison")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("results/regression_comparison.png", dpi=150)
plt.close()

best_reg_name = regression_metrics.iloc[0]["Model"]
plt.figure(figsize=(7, 5))
plt.scatter(y_test, regression_predictions[best_reg_name], alpha=0.6)
plt.xlabel("Actual Spending")
plt.ylabel("Predicted Spending")
plt.title(f"Actual vs Predicted Spending — {best_reg_name}")
plt.tight_layout()
plt.savefig("results/actual_vs_predicted.png", dpi=150)
plt.close()

# 4. Classification extension required by the workshop
# Threshold is calculated from training data only.
threshold = y_train.median()
y_train_cls = (y_train >= threshold).astype(int)
y_test_cls = (y_test >= threshold).astype(int)

classification_pre = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

# Bernoulli NB requires binary features. Numeric features are converted
# to two quantile bins and categorical features are one-hot encoded.
bernoulli_pre = ColumnTransformer([
    ("num", KBinsDiscretizer(n_bins=2, encode="onehot-dense", strategy="quantile"), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

classification_models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", classification_pre),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ]),
    "KNN Classifier": Pipeline([
        ("preprocessor", classification_pre),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Bernoulli Naive Bayes": Pipeline([
        ("preprocessor", bernoulli_pre),
        ("model", BernoulliNB())
    ])
}

classification_rows = []
classification_predictions = {}
for name, model in classification_models.items():
    model.fit(X_train, y_train_cls)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    classification_predictions[name] = pred
    classification_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test_cls, pred),
        "Precision": precision_score(y_test_cls, pred, zero_division=0),
        "Recall": recall_score(y_test_cls, pred, zero_division=0),
        "F1": f1_score(y_test_cls, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test_cls, prob)
    })

classification_metrics = pd.DataFrame(classification_rows).sort_values("F1", ascending=False)
classification_metrics.to_csv("results/classification_metrics.csv", index=False)
print("\nCLASSIFICATION RESULTS")
print(classification_metrics.round(3).to_string(index=False))
print(f"\nHigh-spender threshold: {threshold:.2f}")

best_cls_name = classification_metrics.iloc[0]["Model"]
cm = confusion_matrix(y_test_cls, classification_predictions[best_cls_name])
disp = ConfusionMatrixDisplay(cm, display_labels=["Low Spender", "High Spender"])
disp.plot()
plt.title(f"Confusion Matrix — {best_cls_name}")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()

with open("results/project_summary.txt", "w", encoding="utf-8") as f:
    f.write("Customer Spending Prediction — Offline ML Workshop Project\n")
    f.write("=" * 62 + "\n\n")
    f.write(f"Dataset rows: {len(df)}\n")
    f.write(f"Regression best model: {best_reg_name}\n")
    f.write(f"Classification best model: {best_cls_name}\n")
    f.write(f"High-spender threshold: {threshold:.2f}\n\n")
    f.write("Regression results:\n")
    f.write(regression_metrics.round(4).to_string(index=False))
    f.write("\n\nClassification results:\n")
    f.write(classification_metrics.round(4).to_string(index=False))

print(f"\nBEST REGRESSION MODEL: {best_reg_name}")
print(f"BEST CLASSIFICATION MODEL: {best_cls_name}")
print("Project completed successfully.")
