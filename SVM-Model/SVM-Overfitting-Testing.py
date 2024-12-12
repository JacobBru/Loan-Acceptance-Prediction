import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the data
df = pd.read_csv('../cleaned_loan_data.csv')

# Prepare the data
X = df.drop('loan_status_1', axis=1)
y = df['loan_status_1']

# Sample the data
X = X.sample(n=45000, random_state=42)
y = y.sample(n=45000, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model with the best parameters
model = SVC(kernel='rbf', C=1, gamma=0.1)
model.fit(X_train, y_train)

# Evaluate the model on the training set
y_train_pred = model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
print(f"Training set accuracy: {accuracy_train}")

# Evaluate the model on the test set
y_test_pred = model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
print(f"Test set accuracy: {accuracy_test}")

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean()}")

# Confusion matrix for the training set
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
print(f"Training set confusion matrix:\n{conf_matrix_train}")

# Confusion matrix for the test set
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print(f"Test set confusion matrix:\n{conf_matrix_test}")
