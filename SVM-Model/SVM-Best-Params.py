import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the data
df = pd.read_csv('../cleaned_loan_data.csv')

# Prepare the data
X = df.drop('loan_status_1', axis=1)
y = df['loan_status_1']

# Sample the data
X = X.sample(n=10000, random_state=42)
y = y.sample(n=10000, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid for RBF kernel
param_grid_rbf = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.01, 0.1, 1, 10, 100]
}

# Define the parameter grid for polynomial kernel
param_grid_poly = {
    'C': [0.1, 1, 10, 100, 1000],
    'degree': [2, 3, 4, 5, 6]
}

# Define the parameter grid for linear kernel
param_grid_linear = {
    'C': [0.1, 1, 10, 100, 1000]
}

# Create the SVM models
svm_rbf = SVC(kernel='rbf')
svm_poly = SVC(kernel='poly')
svm_linear = SVC(kernel='linear')

# Perform GridSearchCV for RBF kernel
grid_search_rbf = GridSearchCV(svm_rbf, param_grid_rbf, cv=5, scoring='accuracy')
grid_search_rbf.fit(X_train, y_train)
best_params_rbf = grid_search_rbf.best_params_
best_score_rbf = grid_search_rbf.best_score_

print(f"Best parameters for RBF kernel: {best_params_rbf}")
print(f"Best cross-validation accuracy for RBF kernel: {best_score_rbf}")

# Perform GridSearchCV for polynomial kernel
grid_search_poly = GridSearchCV(svm_poly, param_grid_poly, cv=5, scoring='accuracy')
grid_search_poly.fit(X_train, y_train)
best_params_poly = grid_search_poly.best_params_
best_score_poly = grid_search_poly.best_score_

print(f"Best parameters for polynomial kernel: {best_params_poly}")
print(f"Best cross-validation accuracy for polynomial kernel: {best_score_poly}")

# Perform GridSearchCV for linear kernel
grid_search_linear = GridSearchCV(svm_linear, param_grid_linear, cv=5, scoring='accuracy')
grid_search_linear.fit(X_train, y_train)
best_params_linear = grid_search_linear.best_params_
best_score_linear = grid_search_linear.best_score_

print(f"Best parameters for linear kernel: {best_params_linear}")
print(f"Best cross-validation accuracy for linear kernel: {best_score_linear}")

# Evaluate the best models on the test set
best_model_rbf = grid_search_rbf.best_estimator_
y_pred_rbf = best_model_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"Test set accuracy for RBF kernel: {accuracy_rbf}")

best_model_poly = grid_search_poly.best_estimator_
y_pred_poly = best_model_poly.predict(X_test)
accuracy_poly = accuracy_score(y_test, y_pred_poly)
print(f"Test set accuracy for polynomial kernel: {accuracy_poly}")

best_model_linear = grid_search_linear.best_estimator_
y_pred_linear = best_model_linear.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print(f"Test set accuracy for linear kernel: {accuracy_linear}")