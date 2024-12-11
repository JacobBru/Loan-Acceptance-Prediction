import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


sys.stdout = open('output2.txt', 'w')
# Set the Matplotlib backend
plt.switch_backend('TkAgg')

# Load the data
data = pd.read_csv('cleaned_loan_data.csv')

# Preprocess the data
X = data.drop('loan_status_1', axis=1)
y = data['loan_status_1']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def build_and_train_model_v1(layers, dropout_rate, regularization, learning_rate, epochs):
    model = Sequential()
    for units in layers:
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(regularization)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
    return model, history

def build_and_train_model_v2(layers, dropout_rate, regularization, learning_rate, epochs):
    model = Sequential()
    for units in layers:
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(regularization)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
    return model, history

def build_and_train_model_v3(layers, dropout_rate, regularization, learning_rate, epochs):
    model = Sequential()
    for units in layers:
        model.add(Dense(units, activation='tanh', kernel_regularizer=l2(regularization)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
    return model, history

def build_and_train_model_v4(layers, dropout_rate, regularization, learning_rate, epochs):
    model = Sequential()
    for units in layers:
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(regularization)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0, callbacks=[reduce_lr])
    return model, history

def build_and_train_model_v5(layers, dropout_rate, regularization, learning_rate, epochs):
    model = Sequential()
    for units in layers:
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(regularization)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0, callbacks=[early_stopping])
    return model, history

# Define different configurations
configurations = [
    {'layers': [64, 32], 'dropout_rate': 0.5, 'regularization': 0.001, 'learning_rate': 0.001, 'epochs': 10},
    {'layers': [128, 64, 32], 'dropout_rate': 0.3, 'regularization': 0.001, 'learning_rate': 0.001, 'epochs': 20},
    {'layers': [64, 32], 'dropout_rate': 0.2, 'regularization': 0.01, 'learning_rate': 0.0001, 'epochs': 30},
    {'layers': [128, 64, 32], 'dropout_rate': 0.4, 'regularization': 0.001, 'learning_rate': 0.0005, 'epochs': 15},
    {'layers': [256, 128, 64], 'dropout_rate': 0.5, 'regularization': 0.0001, 'learning_rate': 0.001, 'epochs': 20},
    {'layers': [64, 64, 32], 'dropout_rate': 0.3, 'regularization': 0.01, 'learning_rate': 0.0001, 'epochs': 30},
    {'layers': [128, 128], 'dropout_rate': 0.4, 'regularization': 0.005, 'learning_rate': 0.001, 'epochs': 25},
    {'layers': [32, 16], 'dropout_rate': 0.2, 'regularization': 0.01, 'learning_rate': 0.0001, 'epochs': 40},
    {'layers': [256, 128, 64, 32], 'dropout_rate': 0.3, 'regularization': 0.0005, 'learning_rate': 0.0005, 'epochs': 50},
    {'layers': [512, 256, 128], 'dropout_rate': 0.4, 'regularization': 0.0001, 'learning_rate': 0.001, 'epochs': 30},
]

# List of model building functions
model_builders = [
    build_and_train_model_v1,
    build_and_train_model_v2,
    build_and_train_model_v3,
    build_and_train_model_v4,
    build_and_train_model_v5
]

best_model = None
best_val_accuracy = 0
best_config = None
best_history = None
best_model_builder = None

# Train and evaluate each configuration with each model builder
for model_builder in model_builders:
    for config in configurations:
        model, history = model_builder(**config)
        val_accuracy = max(history.history['val_accuracy'])
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f'Model Builder: {model_builder.__name__}')
        print(f'Configuration: {config}')
        print(f'Validation Accuracy: {val_accuracy}')
        print(f'Test Loss: {loss}, Test Accuracy: {accuracy}\n')
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model
            best_config = config
            best_history = history
            best_model_builder = model_builder

# Print the best configuration and model builder
print(f'Best Model Builder: {best_model_builder.__name__}')
print(f'Best Configuration: {best_config}')

# Evaluate the best model on the test data
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f'Best Model - Loss: {loss}, Accuracy: {accuracy}')

# Generate predictions on the test data
y_pred = best_model.predict(X_test).flatten()

# Threshold for binary classification
y_pred_classes = (y_pred > 0.5).astype(int)

# Create the data_images_2 directory if it doesn't exist
if not os.path.exists('data_images_2'):
    os.makedirs('data_images_2')

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.savefig('data_images_2/confusion_matrix.png')
plt.show()

# 2. Histogram of Predicted Probabilities by Class
y_pred_class_0 = y_pred[y_test == 0]
y_pred_class_1 = y_pred[y_test == 1]

plt.figure(figsize=(10, 6))
plt.hist(y_pred_class_0, bins=np.linspace(0, 1, 50), alpha=0.6, label='Actual Class 0', color='blue')
plt.hist(y_pred_class_1, bins=np.linspace(0, 1, 50), alpha=0.6, label='Actual Class 1', color='orange')
plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Boundary (0.5)')
plt.title('Predicted Probabilities by Actual Class')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('data_images_2/predicted_probabilities_histogram.png')
plt.show()

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('data_images_2/roc_curve.png')
plt.show()

# 4. Scatter Plot for a Random Subset
sample_indices = np.random.choice(len(y_test), size=1000, replace=False)
y_test_sample = y_test.iloc[sample_indices]
y_pred_sample = y_pred[sample_indices]

plt.figure(figsize=(8, 6))
plt.scatter(y_test_sample, y_pred_sample, alpha=0.5, label='Predicted Probabilities')
plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Boundary (0.5)')
plt.title('Predicted Probabilities vs Actual Labels (Subset)')
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Probabilities')
plt.legend()
plt.savefig('data_images_2/predicted_probabilities_scatter.png')
plt.show()

# 5. Accuracy and Loss vs. Epoch
# Plot training & validation accuracy values for the best model
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(best_history.history['accuracy'])
plt.plot(best_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values for the best model
plt.subplot(1, 2, 2)
plt.plot(best_history.history['loss'])
plt.plot(best_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.savefig('data_images_2/accuracy_loss_vs_epoch.png')
plt.show()

sys.stdout.close()
