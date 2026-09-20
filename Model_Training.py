"""
Model Training — Algerian Forest Fires (FWI Regression)

Consolidated from Model_Training.ipynb.
Trains and compares Linear Regression, Lasso, LassoCV, Ridge, RidgeCV,
ElasticNet, and ElasticNetCV on the Algerian Forest Fires dataset to
predict the Fire Weather Index (FWI).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import (
    LinearRegression,
    Lasso, LassoCV,
    Ridge, RidgeCV,
    ElasticNet, ElasticNetCV,
)

# 1. Load data

df = pd.read_csv('Algerian_forest_fires_dataset_Cleaned.csv')
print(df.head())
print(df.columns)

# 2. Clean up / drop unneeded columns
#done in the cleaning file
# drop day, month and year
#df.drop(['day', 'month', 'year'], axis=1, inplace=True)
#print(df.head())
#print(df['Classes'].value_counts())

# 3. Encoding
#done in the cleaning file
#df['Classes'] = np.where(df['Classes'].str.contains("not fire"), 0, 1)
#print(df.tail())
#print(df['Classes'].value_counts())

# 4. Independent and dependent features

X = df.drop('FWI', axis=1)
y = df['FWI']
print(X.head())
print(y)

# 5. Train/Test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train.shape, X_test.shape)

# 6. Feature selection based on correlation / multicollinearity

print(X_train.corr())

plt.figure(figsize=(12, 10))
corr = X_train.corr()
sns.heatmap(corr, annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()


def correlation(dataset, threshold):
    """Return the set of column names with pairwise correlation above threshold."""
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


# threshold -- domain expertise
corr_features = correlation(X_train, 0.85)
print("Highly correlated features to drop:", corr_features)

# drop features when correlation is more than 0.85
X_train.drop(corr_features, axis=1, inplace=True)
X_test.drop(corr_features, axis=1, inplace=True)
print(X_train.shape, X_test.shape)

# 7. Feature scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled)

plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=X_train)
plt.title('X_train Before Scaling')
plt.subplot(1, 2, 2)
sns.boxplot(data=X_train_scaled)
plt.title('X_train After Scaling')
plt.show()

# 8. Linear Regression

linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)
y_pred = linreg.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("\n--- Linear Regression ---")
print("Mean absolute error", mae)
print("R2 Score", score)
plt.scatter(y_test, y_pred)
plt.title("Linear Regression: Actual vs Predicted")
plt.show()

# 9. Lasso Regression

lasso = Lasso()
lasso.fit(X_train_scaled, y_train)
y_pred = lasso.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("\n--- Lasso ---")
print("Mean absolute error", mae)
print("R2 Score", score)
plt.scatter(y_test, y_pred)
plt.title("Lasso: Actual vs Predicted")
plt.show()

# 10. LassoCV

lassocv = LassoCV(cv=5)
lassocv.fit(X_train_scaled, y_train)
print("\n--- LassoCV ---")
print("Best alpha:", lassocv.alpha_)
print("Alphas tried:", lassocv.alphas_)
print("MSE path:", lassocv.mse_path_)

y_pred = lassocv.predict(X_test_scaled)
plt.scatter(y_test, y_pred)
plt.title("LassoCV: Actual vs Predicted")
plt.show()
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("Mean absolute error", mae)
print("R2 Score", score)

# 11. Ridge Regression

ridge = Ridge()
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("\n--- Ridge ---")
print("Mean absolute error", mae)
print("R2 Score", score)
plt.scatter(y_test, y_pred)
plt.title("Ridge: Actual vs Predicted")
plt.show()

# 12. RidgeCV

ridgecv = RidgeCV(cv=5)
ridgecv.fit(X_train_scaled, y_train)
y_pred = ridgecv.predict(X_test_scaled)
plt.scatter(y_test, y_pred)
plt.title("RidgeCV: Actual vs Predicted")
plt.show()
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("\n--- RidgeCV ---")
print("Mean absolute error", mae)
print("R2 Score", score)
print("Params:", ridgecv.get_params())

# 13. ElasticNet

elastic = ElasticNet()
elastic.fit(X_train_scaled, y_train)
y_pred = elastic.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("\n--- ElasticNet ---")
print("Mean absolute error", mae)
print("R2 Score", score)
plt.scatter(y_test, y_pred)
plt.title("ElasticNet: Actual vs Predicted")
plt.show()

# 14. ElasticNetCV

elasticcv = ElasticNetCV(cv=5)
elasticcv.fit(X_train_scaled, y_train)
y_pred = elasticcv.predict(X_test_scaled)
plt.scatter(y_test, y_pred)
plt.title("ElasticNetCV: Actual vs Predicted")
plt.show()
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)
print("\n--- ElasticNetCV ---")
print("Mean absolute error", mae)
print("R2 Score", score)
print("Alphas tried:", elasticcv.alphas_)