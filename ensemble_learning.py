import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# Load Dataset
df = pd.read_csv("train.csv")

# Select Features and Target Variable
features = ["GrLivArea", "YearBuilt"]
target = "SalePrice"
X = df[features]
y = df[target]

# Split into Training (80%) and Validation (20%) Sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Problem 1: Blending (Diverse Models & Weighted Averaging)
print("\n--- Blending ---")

# Define diverse models with preprocessing
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

svr = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', SVR(kernel="rbf", C=10))
])

dt = Pipeline([
    ('pca', PCA(n_components=1)),
    ('regressor', DecisionTreeRegressor(max_depth=5, random_state=42))
])

models = [lr, svr, dt]

# Train models and make predictions
preds = []
for model in models:
    model.fit(X_train, y_train)
    preds.append(model.predict(X_val))

preds = np.array(preds)

# Compute individual model MSE
mse_values = [mean_squared_error(y_val, pred) for pred in preds]
for i, mse in enumerate(mse_values):
    print(f"Model {i+1} MSE: {mse}")

# Weighted Blending Optimization
def mse_loss(weights):
    blended_preds = np.sum(weights[:, None] * preds, axis=0)
    return mean_squared_error(y_val, blended_preds)

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1)] * len(models)
initial_weights = np.ones(len(models)) / len(models)
result = minimize(mse_loss, initial_weights, bounds=bounds, constraints=constraints)
optimal_weights = result.x

# Blended Predictions with Optimal Weights
blended_preds = np.sum(optimal_weights[:, None] * preds, axis=0)
mse_blended = mean_squared_error(y_val, blended_preds)

# Print MSE comparisons
print(f"Blending MSE: {mse_blended}")
print(f"Optimal Weights: {optimal_weights}")

# Check if blending is better than at least three individual models
better_count = sum(mse_blended < mse for mse in mse_values)
print(f"Blending performed better than {better_count} models.")

# Problem 2: Bagging (Bootstrap Aggregating)
print("\n--- Bagging ---")

# Parameters
n_estimators = 10  # Number of bootstrap samples
bagged_preds = np.zeros_like(y_val, dtype=float)

# Train multiple models on different bootstrap samples
for i in range(n_estimators):
    X_sample, y_sample = X_train.sample(frac=1, replace=True, random_state=i), y_train.sample(frac=1, replace=True, random_state=i)
    model = DecisionTreeRegressor(max_depth=5, random_state=i)
    model.fit(X_sample, y_sample)
    bagged_preds += model.predict(X_val) / n_estimators

# Compute Bagging MSE
mse_bagging = mean_squared_error(y_val, bagged_preds)
print(f"Bagging MSE: {mse_bagging}")

# Compare with a single Decision Tree
single_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
single_tree.fit(X_train, y_train)
single_tree_preds = single_tree.predict(X_val)
mse_single_tree = mean_squared_error(y_val, single_tree_preds)
print(f"Single Decision Tree MSE: {mse_single_tree}")

# Check if Bagging improves accuracy
if mse_bagging < mse_single_tree:
    print("Bagging achieved better accuracy than a single Decision Tree.")
else:
    print("Bagging did not improve accuracy over a single Decision Tree.")

# Problem 3: Stacking (Multi-Level Learning)
print("\n--- Stacking ---")

# Define base models
base_models = [
    ('lr', LinearRegression()),
    ('svr', SVR(kernel='rbf', C=10)),
    ('dt', DecisionTreeRegressor(max_depth=5, random_state=42))
]

# Train base models and get predictions
meta_features = np.zeros((X_val.shape[0], len(base_models)))
for i, (name, model) in enumerate(base_models):
    model.fit(X_train, y_train)
    meta_features[:, i] = model.predict(X_val)

# Define meta-model
meta_model = RandomForestRegressor(n_estimators=50, random_state=42)
meta_model.fit(meta_features, y_val)

# Get final stacked predictions
stacked_preds = meta_model.predict(meta_features)
mse_stacking = mean_squared_error(y_val, stacked_preds)
print(f"Stacking MSE: {mse_stacking}")

# Compare stacking with single model performances
mse_single_models = [mean_squared_error(y_val, meta_features[:, i]) for i in range(len(base_models))]
for i, mse in enumerate(mse_single_models):
    print(f"Base Model {i+1} MSE: {mse}")

# Check if Stacking improves accuracy
better_than_single = sum(mse_stacking < mse for mse in mse_single_models)
print(f"Stacking performed better than {better_than_single} base models.")
