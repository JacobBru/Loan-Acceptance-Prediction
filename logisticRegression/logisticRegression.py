import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load training data
file_path = "../datasets/cleaned_loan_data.csv"
data = pd.read_csv(file_path)

# Remove feature column
X = data.drop(columns=["loan_status_1"])
y = data["loan_status_1"]

# Convert True/False columns to 1/0
X = X.astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data to hopefully improve convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Initialize the logistic regression model
model = LogisticRegression(solver='liblinear', penalty='l1', max_iter=5000, C=0.1) # 1000 is a magic number that I have to tweak

# Training and prediction
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# List results
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

