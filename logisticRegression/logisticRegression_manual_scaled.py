import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

#################################################################################
# UTILITY FUNCTIONS REQUIRED FOR THIS CODE TO RUN PROPERLY
#################################################################################

# Implementation of sigmoid function
def Sigmoid(x):
    # Use numpy to clip extreme values
    x = np.clip(x, -500, 500)  # Prevent overflow
    return 1.0 / (1.0 + np.exp(-x))

# Implementation of prediction function
def Prediction(theta, x):
    z = np.dot(theta, x)
    return Sigmoid(z)

# Implementation of cost function
def Cost_Function(X, Y, theta, m):
    predictions = Prediction(theta, X.T)  # Transpose X for dot product
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)  # Clip predictions to avoid log(0)
    cost = -np.mean(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))
    return cost

# Execute gradient updates over thetas
def Gradient_Descent(X, Y, theta, m, alpha):
    predictions = Prediction(theta, X.T)  # Transpose X for dot product
    errors = predictions - Y
    gradient = np.dot(X.T, errors) / m
    theta -= alpha * gradient
    return theta

#################################################################################
# END UTILITY FUNCTIONS
#################################################################################

# Load training data
file_path = "../datasets/cleaned_loan_data.csv"
data = pd.read_csv(file_path)

# Remove target column
X = data.drop(columns=["loan_status_1"])
y = data["loan_status_1"]

# Convert all data to numeric
X = X.astype(float)

# Split the data into training and testing sets
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data to improve convergence
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

# Convert scaled NumPy arrays back to DataFrames for consistency
trainX = pd.DataFrame(trainX, columns=X.columns)
testX = pd.DataFrame(testX, columns=X.columns)

# Initialize parameters
theta = np.zeros(trainX.shape[1])  # Adjusted for the correct number of features
alpha = 0.1  # Learning rate
max_iteration = 1000  # Maximum iterations

m = len(trainY)  # Number of training samples

# Perform gradient descent
for x in range(max_iteration):
    batch_size = 60
    indices = np.random.choice(m, batch_size, replace=False)
    X_batch = trainX.iloc[indices].to_numpy()  # Convert to NumPy for dot products
    Y_batch = trainY.iloc[indices].to_numpy()
    theta = Gradient_Descent(X_batch, Y_batch, theta, batch_size, alpha)
    if x % 200 == 0:
        print(f"Iteration {x}, Cost: {Cost_Function(X_batch, Y_batch, theta, batch_size)}")

# Apply the model to the testing samples
yHat = []
for i in range(len(testX)):
    xi = testX.iloc[i].to_numpy()
    prediction = round(Prediction(theta, xi))
    yHat.append(prediction)

# Calculate confusion matrix
def func_calConfusionMatrix(predY, trueY):
    predY = np.array(predY).astype(int)
    trueY = np.array(trueY).astype(int)

    A = ((predY == 1) & (trueY == 1)).sum()  # True Positives
    B = ((predY == 0) & (trueY == 1)).sum()  # False Negatives
    C = ((predY == 1) & (trueY == 0)).sum()  # False Positives
    D = ((predY == 0) & (trueY == 0)).sum()  # True Negatives

    return {
        'accuracy': (A + D) / (A + B + C + D),
        'classA_prediction': A / (A + C) if (A + C) != 0 else 0,
        'classA_recall': A / (A + B) if (A + B) != 0 else 0,
        'classB_prediction': D / (B + D) if (B + D) != 0 else 0,
        'classB_recall': D / (C + D) if (C + D) != 0 else 0
    }

# Evaluation metrics
values = func_calConfusionMatrix(yHat, testY)
print("Accuracy: ", values['accuracy'])
print("Class A Prediction: ", values['classA_prediction'])
print("Class A Recall: ", values['classA_recall'])
print("Class B Prediction: ", values['classB_prediction'])
print("Class B Recall: ", values['classB_recall'])

