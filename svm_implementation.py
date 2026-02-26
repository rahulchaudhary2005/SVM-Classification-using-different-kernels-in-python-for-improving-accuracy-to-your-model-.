# ================================
# SVM Classification Experiment
# ================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --------------------------------
# 1. Generate Dataset (300 Samples)
# --------------------------------

X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

# --------------------------------
# 2. Split Dataset (Train/Test)
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

# --------------------------------
# 3. SVM Models with Different Kernels
# --------------------------------

models = {
    "Linear Kernel": SVC(kernel="linear", C=1),
    "Polynomial Kernel": SVC(kernel="poly", degree=3, C=1),
    "RBF Kernel (gamma=0.1)": SVC(kernel="rbf", gamma=0.1, C=1),
    "RBF Kernel (gamma=1)": SVC(kernel="rbf", gamma=1, C=1),
    "RBF Kernel (gamma=10)": SVC(kernel="rbf", gamma=10, C=1),
    "RBF Kernel (C=10)": SVC(kernel="rbf", gamma=1, C=10)
}

# --------------------------------
# 4. Train & Evaluate Models
# --------------------------------

results = {}

print("SVM Classification Results\n")

for name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    results[name] = acc

    print(f"{name} Accuracy: {acc:.4f}")


# --------------------------------
# 5. Function to Plot Decision Boundary
# --------------------------------

def plot_decision_boundary(model, X, y, title):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")


# --------------------------------
# 6. Plot Decision Boundaries
# --------------------------------

plt.figure(figsize=(15, 10))

i = 1

for name, model in models.items():

    plt.subplot(2, 3, i)

    plot_decision_boundary(
        model,
        X_train,
        y_train,
        name
    )

    i += 1

plt.tight_layout()
plt.show()


# --------------------------------
# 7. Plot Accuracy Comparison
# --------------------------------

plt.figure(figsize=(10, 5))

names = list(results.keys())
accuracies = list(results.values())

plt.bar(names, accuracies)
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)

plt.title("SVM Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.show()


# --------------------------------
# 8. Conclusion Print
# --------------------------------

print("\nSummary:")

best_model = max(results, key=results.get)
best_accuracy = results[best_model]

print(f"Best Model: {best_model}")
print(f"Best Accuracy: {best_accuracy:.4f}")
