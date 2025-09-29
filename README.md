# task-4-alml

# Task 4: Logistic Regression Binary Classification
# Dataset: Breast Cancer Wisconsin (built-in from sklearn)
# Tools: pandas, numpy, matplotlib, seaborn, scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score
)

# 1. Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset shape:", X.shape)
print("Target classes:", data.target_names)

# 2. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Logistic Regression
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)

# 5. Predictions
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

# 6. Evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve & AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 7. Threshold tuning example
custom_threshold = 0.3
y_pred_custom = (y_pred_prob >= custom_threshold).astype(int)
print("\nWith threshold =", custom_threshold)
print("Precision:", precision_score(y_test, y_pred_custom))
print("Recall:", recall_score(y_test, y_pred_custom))

# 8. Sigmoid function visualization
z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))
plt.plot(z, sigmoid, label="Sigmoid Function")
plt.axvline(0, color='r', linestyle='--', label="Threshold at 0")
plt.xlabel("z (Linear output)")
plt.ylabel("Sigmoid(z)")
plt.title("Sigmoid Function")
plt.legend()
plt.show()

"""
Interview Questions (Answers):

1. Logistic vs Linear Regression:
   - Linear regression predicts continuous values.
   - Logistic regression predicts probabilities for classification.

2. Sigmoid function:
   - Maps any real number into range (0,1).
   - Formula: 1 / (1 + exp(-z)).

3. Precision vs Recall:
   - Precision = TP / (TP + FP) → correctness of positive predictions.
   - Recall = TP / (TP + FN) → ability to find all positives.

4. ROC-AUC:
   - ROC shows trade-off between TPR and FPR at various thresholds.
   - AUC is the area under ROC, measuring model’s discrimination power.

5. Confusion Matrix:
   - Table showing counts of TP, FP, TN, FN.

6. Imbalanced classes:
   - Model may favor majority class.
   - Use resampling, class weights, or metrics like ROC-AUC, F1.

7. Choosing threshold:
   - Depends on problem.
   - If false negatives are costly → lower threshold (higher recall).
   - If false positives are costly → higher threshold (higher precision).

8. Logistic Regression for multi-class:
   - Yes, using One-vs-Rest (OvR) or Softmax (multinomial).
"""
