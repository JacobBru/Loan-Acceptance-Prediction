import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load training data
file_path = "../datasets/cleaned_loan_data.csv"
data = pd.read_csv(file_path)

# Remove feature column
X = data.drop(columns=["loan_status_1"])
y = data["loan_status_1"]

# Convert True/False columns to 1/0
X = X.astype(int)

# Scale the dataset
# (This helped it converge when nothing else would)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the default logistic regression model
# (This is just to have something to train the grid search on,
# The actual model with the best parameters will be implemented
# below)
default_model = LogisticRegression(max_iter=5000)

# Hyperparameter options for GridSearchCV
param_grid = {
    'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Set up K-fold cross validation (starting with 5 folds)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Run grid search
grid_search = GridSearchCV(estimator=default_model, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_scaled, y)

# Display the best parameters and mean cross-validation score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Split the dataset and scale it appropriately
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Use the best model from the grid search
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# List results
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

