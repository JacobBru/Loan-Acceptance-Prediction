# This file is just using matplotlib to create the visuals for the paper
import matplotlib.pyplot as plt

#################################################################
# Accuracy vs C Values
#################################################################

# Values
c_values = [0.1, 0.01, 0.001]
accuracies = [0.8840, 0.8825, 0.8741]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(c_values, accuracies, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)

# Add labels, title, and grid
plt.xlabel("C Value (Regularization Parameter)", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Accuracy vs C Value for Logistic Regression", fontsize=14)
plt.grid(True)

# Highlight data points
for i, txt in enumerate(accuracies):
    plt.annotate(f"{txt:.2f}", (c_values[i], accuracies[i]), textcoords="offset points", xytext=(0, 10), ha='center')

# Show or save the plot
plt.tight_layout()
plt.savefig("accuracy_vs_c.png")  # Saves the plot as an image file
plt.show()

#################################################################
# Accuracy vs Solver
#################################################################

# Values
solvers = ["libfgs", "saga", "liblinear"]
accuracies = [0.8840, 0.8838, 0.8840]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(solvers, accuracies, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)

# Add labels, title, and grid
plt.xlabel("Solver", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Accuracy vs Solver for Logistic Regression", fontsize=14)
plt.grid(True)

# Highlight data points
for i, txt in enumerate(accuracies):
    plt.annotate(f"{txt:.2f}", (solvers[i], accuracies[i]), textcoords="offset points", xytext=(0, 10), ha='center')

# Show or save the plot
plt.tight_layout()
plt.savefig("accuracy_vs_solver.png")  # Saves the plot as an image file
plt.show()

#################################################################
# Accuracy vs Regularizer
#################################################################

# Values
regularizers = ["Default", "L1", "L2"]
accuracies = [0.8840, 0.8846, 0.8840]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(regularizers, accuracies, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)

# Add labels, title, and grid
plt.xlabel("Regularizer", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Accuracy vs Regularizer for Logistic Regression", fontsize=14)
plt.grid(True)

# Highlight data points
for i, txt in enumerate(accuracies):
    plt.annotate(f"{txt:.2f}", (regularizers[i], accuracies[i]), textcoords="offset points", xytext=(0, 10), ha='center')

# Show or save the plot
plt.tight_layout()
plt.savefig("accuracy_vs_regularizer.png")  # Saves the plot as an image file
plt.show()
