import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

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


############################################################################################################
#Testing accuracy for different kernels
# Define the SVM models
# svm_models = {
#     'linear': SVC(kernel='linear'),  0.888 accuracy
#     'rbf': SVC(kernel='rbf', gamma=0.1),  0.896 accuracy
#     'poly': SVC(kernel='poly', degree=3)  0.895 accuracy
# }
# Train and evaluate each model
# for kernel, model in svm_models.items():
#     print(f"Training SVM model with {kernel} kernel...")
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"{kernel.capitalize()} kernel accuracy: {accuracy}")
############################################################################################################

############################################################################################################
# testing for Linear kernel with different C values
# Define the C values to test
c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Store the accuracy results
accuracies = []

# Train and evaluate the model for each C value
for c in c_values:
    print(f"Training SVM model with linear kernel and C={c}...")
    model = SVC(kernel='linear', C=c)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Linear kernel with C={c} accuracy: {accuracy}")


# Plot the results
plt.plot(c_values, accuracies, marker='o')
plt.xlabel('C value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs C value for Linear Kernel')
plt.xscale('log')
plt.show()
############################################################################################################

############################################################################################################
# testing for rbf kernel with different C values and gamma values
c_values = [0.1, 1, 10, 100, 1000]

# Store the accuracy results for C values
accuracies_c = []

# Train and evaluate the model for each C value
for c in c_values:
    print(f"Training SVM model with RBF kernel and C={c}...")
    model = SVC(kernel='rbf', C=c, gamma=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_c.append(accuracy)
    print(f"RBF kernel with C={c} accuracy: {accuracy}")

# Plot the results for C values
plt.plot(c_values, accuracies_c, marker='o')
plt.xlabel('C value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs C value for RBF Kernel')
plt.xscale('log')
plt.show()

# Define the gamma values to test
gamma_values = [0.01, 0.1, 1, 10, 100]

# Store the accuracy results for gamma values
accuracies_gamma = []

# Train and evaluate the model for each gamma value
for gamma in gamma_values:
    print(f"Training SVM model with RBF kernel and gamma={gamma}...")
    model = SVC(kernel='rbf', C=1, gamma=gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_gamma.append(accuracy)
    print(f"RBF kernel with gamma={gamma} accuracy: {accuracy}")

# Plot the results for gamma values
plt.plot(gamma_values, accuracies_gamma, marker='o')
plt.xlabel('Gamma value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Gamma value for RBF Kernel')
plt.xscale('log')
plt.show()

############################################################################################################
# testing for poly kernel with different degree values
# Define the degree values to test
degree_values = [2, 3, 4, 5, 6]

# Store the accuracy results for degree values
accuracies_degree = []

# Train and evaluate the model for each degree value
for degree in degree_values:
    print(f"Training SVM model with poly kernel and degree={degree}...")
    model = SVC(kernel='poly', C=1, degree=degree)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_degree.append(accuracy)
    print(f"Poly kernel with degree={degree} accuracy: {accuracy}")

# Plot the results for degree values
plt.plot(degree_values, accuracies_degree, marker='o')
plt.xlabel('Degree value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Degree value for Poly Kernel')
plt.show()

############################################################################################################
