import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# load the data from the FRED data collected earlier
df = pd.read_csv("data/FRED_data.csv")

print(df.head())
print(df.info())

# features
X = df.drop(columns=["recession", "observation_date"])

# target
y = df["recession"]

print(y.value_counts())

# we split the data chronologically due to it being a time-series economic data.
# we will train on old data and test on new data which simulates real world forecasting
split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# use logistic regression as our first test as a strong baseline for classification
log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)

# the data has a very large amount of non recession months sop we need to make sure to account for this because the
# accuracy of the model can be very misleading. If the model predicts non recession every month it would achieve a high
# accuracy

print("Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, log_pred))
print("Precision:", precision_score(y_test, log_pred, zero_division=0))
print("Recall:", recall_score(y_test, log_pred, zero_division=0))
print("F1:", f1_score(y_test, log_pred, zero_division=0))

print(classification_report(y_test, log_pred, zero_division=0))

tscv = TimeSeriesSplit(n_splits=5)

cv_accuracy, cv_precision, cv_recall, cv_f1 = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_fold_train = X.iloc[train_idx]
    X_fold_test = X.iloc[test_idx]
    y_fold_train = y.iloc[train_idx]
    y_fold_test = y.iloc[test_idx]

    fold_scaler = StandardScaler()
    X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
    X_fold_test_scaled = fold_scaler.transform(X_fold_test)

    rf_cv = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight="balanced_subsample",
        random_state=42,
    )
    rf_cv.fit(X_fold_train_scaled, y_fold_train)
    y_pred = rf_cv.predict(X_fold_test_scaled)

    cv_accuracy.append(accuracy_score(y_fold_test, y_pred))
    cv_precision.append(precision_score(y_fold_test, y_pred, zero_division=0))
    cv_recall.append(recall_score(y_fold_test, y_pred, zero_division=0))
    cv_f1.append(f1_score(y_fold_test, y_pred, zero_division=0))

print("Random Forest Results (5-Fold Time-Series Cross-Validation)")
print("Average Accuracy: ", round(np.mean(cv_accuracy), 4))
print("Average Precision:", round(np.mean(cv_precision), 4))
print("Average Recall:   ", round(np.mean(cv_recall), 4))
print("Average F1:       ", round(np.mean(cv_f1), 4))