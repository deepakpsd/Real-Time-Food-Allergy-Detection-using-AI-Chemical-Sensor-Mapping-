import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle

# -----------------------------
# 1. Load dataset
# -----------------------------
file_path = "E:/Projects/4th Semester/Website New/dataset 3.xlsx"

data = pd.read_csv(file_path)

print("\nOriginal dataset shape:", data.shape)
print("Original columns:", list(data.columns))

# -----------------------------
# 2. Validate required columns
# -----------------------------
required_columns = ["value", "label"]
for col in required_columns:
    if col not in data.columns:
        print(f"Error: Required column '{col}' not found")
        exit()

data = data[["value", "label"]].copy()

# -----------------------------
# 3. Clean dataset
# -----------------------------
data.dropna(how="all", inplace=True)
data["label"] = data["label"].astype(str).str.strip().str.lower()
data["value"] = pd.to_numeric(data["value"], errors="coerce")

data.replace("", np.nan, inplace=True)
data.dropna(subset=["value", "label"], inplace=True)

allowed_labels = ["egg", "milk", "peanut", "safe"]
data = data[data["label"].isin(allowed_labels)]

data = data[(data["value"] >= 0) & (data["value"] <= 1023)]
data.drop_duplicates(inplace=True)

data = shuffle(data, random_state=42).reset_index(drop=True)

print("\nCleaned dataset shape:", data.shape)
print("Classes found:", data["label"].unique())

# -----------------------------
# 4. Class distribution
# -----------------------------
print("\nClass distribution:")
print(data["label"].value_counts())

# -----------------------------
# 5. Features
# -----------------------------
X = data[["value"]]
y = data["label"]

# -----------------------------
# 6. Train-Test Split (70/30)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------
# 7. Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 8. Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 9. Evaluation
# -----------------------------
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=3))

cm = confusion_matrix(y_test, y_pred, labels=allowed_labels)
print("Confusion Matrix:\n", cm)

# -----------------------------
# 10. Confusion Matrix Plot
# -----------------------------
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks(np.arange(len(allowed_labels)), allowed_labels)
plt.yticks(np.arange(len(allowed_labels)), allowed_labels)

for i in range(len(allowed_labels)):
    for j in range(len(allowed_labels)):
        plt.text(j, i, cm[i][j], ha="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()

# -----------------------------
# 11. Metrics Graphs
# -----------------------------
precision = precision_score(y_test, y_pred, average=None, labels=allowed_labels)
recall = recall_score(y_test, y_pred, average=None, labels=allowed_labels)

# Precision Graph
plt.figure()
plt.bar(allowed_labels, precision)
plt.title("Precision per Class")
plt.xlabel("Classes")
plt.ylabel("Precision")
plt.savefig("precision_graph.png")
plt.show()

# Recall Graph
plt.figure()
plt.bar(allowed_labels, recall)
plt.title("Recall per Class")
plt.xlabel("Classes")
plt.ylabel("Recall")
plt.savefig("recall_graph.png")
plt.show()

# Accuracy Graph
plt.figure()
plt.bar(["Accuracy"], [acc])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.savefig("accuracy_graph.png")
plt.show()

# Performance Comparison Graph
f1_scores = (2 * precision * recall) / (precision + recall + 1e-6)

x = np.arange(len(allowed_labels))

plt.figure()
plt.bar(x - 0.2, precision, width=0.2, label="Precision")
plt.bar(x, recall, width=0.2, label="Recall")
plt.bar(x + 0.2, f1_scores, width=0.2, label="F1 Score")

plt.xticks(x, allowed_labels)
plt.title("Performance Comparison")
plt.xlabel("Classes")
plt.ylabel("Score")
plt.legend()

plt.savefig("performance_comparison.png")
plt.show()

# -----------------------------
# 12. Save Model
# -----------------------------
joblib.dump(model, "allergy_model.pkl")

print("\nModel saved as allergy_model.pkl")
print("Graphs saved successfully!")