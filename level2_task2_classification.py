"""
Codveda Data Science Internship - Level 2, Task 2: Classification with Logistic Regression
Author: Thapelo Oatile Tlhomelang

Goal: Build a classifier to predict a categorical outcome (species of flowers).

Dataset: the classic Iris dataset (built into scikit-learn, no download
required) - 150 samples, 3 species, 4 numeric features (sepal/petal length
and width). It's the standard teaching dataset for exactly this kind of task.

Steps performed:
1. Preprocess the data (feature scaling).
2. Train and evaluate a Logistic Regression model.
3. Evaluate using accuracy, precision, recall, and a confusion matrix
   (the multiclass equivalent of an ROC curve view).
4. Compare Logistic Regression against Random Forest and SVM classifiers.
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42


def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    target_names = iris.target_names
    return X, y, target_names


def preprocess(X_train, X_test):
    """Feature scaling - important for Logistic Regression and SVM, which are
    sensitive to feature magnitude (unlike tree-based models)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def evaluate_model(model, X_train, X_test, y_train, y_test, name, target_names):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro")
    recall = recall_score(y_test, predictions, average="macro")
    f1 = f1_score(y_test, predictions, average="macro")

    print(f"\n{name}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (macro avg)")
    print(f"  Recall:    {recall:.4f} (macro avg)")
    print(f"  F1-score:  {f1:.4f} (macro avg)")

    return {
        "model": name, "accuracy": accuracy,
        "precision": precision, "recall": recall, "f1": f1,
    }


if __name__ == "__main__":
    X, y, target_names = load_data()
    print(f"Loaded Iris dataset: {len(X)} samples, {X.shape[1]} features, "
          f"{len(target_names)} species: {list(target_names)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    results = []

    # Logistic Regression - the primary model for this task.
    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    results.append(evaluate_model(
        log_reg, X_train_scaled, X_test_scaled, y_train, y_test,
        "Logistic Regression", target_names
    ))

    log_reg_preds = log_reg.predict(X_test_scaled)
    print("\nLogistic Regression - Detailed classification report:")
    print(classification_report(y_test, log_reg_preds, target_names=target_names))

    print("Confusion matrix (rows = actual, columns = predicted):")
    cm = confusion_matrix(y_test, log_reg_preds)
    print(pd.DataFrame(cm, index=target_names, columns=target_names))

    # Comparison models.
    results.append(evaluate_model(
        RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
        X_train_scaled, X_test_scaled, y_train, y_test,
        "Random Forest", target_names
    ))
    results.append(evaluate_model(
        SVC(kernel="rbf", random_state=RANDOM_STATE),
        X_train_scaled, X_test_scaled, y_train, y_test,
        "SVM (RBF kernel)", target_names
    ))

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    results_df.to_csv("classification_model_comparison.csv", index=False)

    print("\n" + "=" * 50)
    print("Model comparison (best accuracy first):")
    print(results_df.to_string(index=False))
