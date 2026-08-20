import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Synthetic customer dataset for a fully offline workshop project
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
complaint = np.random.binomial(1, .15, n)
response = np.random.binomial(1, .25, n)
education = np.random.choice(['Basic','Graduation','Master','PhD'], n, p=[.12,.55,.25,.08])
marital = np.random.choice(['Single','Married','Together','Divorced'], n)

spending = (350 + 0.014*income + 7*age - 2.2*recency + 24*deals + 42*web +
            34*catalog + 50*store + 8*visits - 70*kids - 45*teens +
            180*response - 90*complaint + np.random.normal(0, 90, n))
spending = np.maximum(spending, 100).round(2)

df = pd.DataFrame({'Age':age,'Income':income,'Education':education,'Marital_Status':marital,
                   'Kidhome':kids,'Teenhome':teens,'Recency':recency,
                   'NumDealsPurchases':deals,'NumWebPurchases':web,
                   'NumCatalogPurchases':catalog,'NumStorePurchases':store,
                   'NumWebVisitsMonth':visits,'Complaint':complaint,
                   'Response':response,'Total_Spending':spending})
df.to_csv('data/customer_spending.csv', index=False)

features = ['Income','Age','Recency','NumDealsPurchases','NumWebPurchases',
            'NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth',
            'Kidhome','Teenhome','Complaint','Response','Education','Marital_Status']
X = df[features]
y = df['Total_Spending']
cat = ['Education','Marital_Status']
num = [c for c in features if c not in cat]
pre = ColumnTransformer([('num', StandardScaler(), num), ('cat', OneHotEncoder(handle_unknown='ignore'), cat)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=RANDOM_STATE)

models = {
    'Linear Regression (1 feature)': Pipeline([('prep', ColumnTransformer([('num', StandardScaler(), ['Income'])])), ('model', LinearRegression())]),
    'Multiple Linear Regression': Pipeline([('prep', pre), ('model', LinearRegression())]),
    'KNN Regressor': Pipeline([('prep', pre), ('model', KNeighborsRegressor(n_neighbors=7))])
}
results = []
for name, model in models.items():
    model.fit(X_train if '1 feature' not in name else X_train[['Income']], y_train)
    pred = model.predict(X_test if '1 feature' not in name else X_test[['Income']])
    results.append([name, mean_absolute_error(y_test,pred), mean_squared_error(y_test,pred)**.5, r2_score(y_test,pred)])

metrics = pd.DataFrame(results, columns=['Model','MAE','RMSE','R2'])
print('\nRegression model comparison:')
print(metrics.to_string(index=False, formatters={'MAE':'{:.2f}'.format,'RMSE':'{:.2f}'.format,'R2':'{:.3f}'.format}))
metrics.to_csv('results/regression_metrics.csv', index=False)

best = metrics.loc[metrics.R2.idxmax()]
plt.figure(figsize=(8,5)); plt.bar(metrics.Model, metrics.RMSE); plt.xticks(rotation=20, ha='right'); plt.ylabel('RMSE'); plt.title('Regression Model Error Comparison'); plt.tight_layout(); plt.savefig('results/regression_error_comparison.png', dpi=150); plt.close()

best_model = models[best.Model]
best_model.fit(X_train if '1 feature' not in best.Model else X_train[['Income']], y_train)
pred = best_model.predict(X_test if '1 feature' not in best.Model else X_test[['Income']])
plt.figure(figsize=(7,5)); plt.scatter(y_test,pred,alpha=.6); plt.xlabel('Actual Spending'); plt.ylabel('Predicted Spending'); plt.title('Actual vs Predicted Spending'); plt.tight_layout(); plt.savefig('results/actual_vs_predicted.png', dpi=150); plt.close()

plt.figure(figsize=(7,5)); plt.hist(y,bins=30); plt.xlabel('Total Spending'); plt.ylabel('Customers'); plt.title('Customer Spending Distribution'); plt.tight_layout(); plt.savefig('results/spending_distribution.png', dpi=150); plt.close()
print(f'\nBest model: {best.Model} | R² = {best.R2:.3f}')
