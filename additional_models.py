import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

#Load data
df = pd.read_csv("data/FRED_data.csv")
df = df.sort_values("observation_date").reset_index(drop=True)

#Features and target
X = df.drop(columns=["recession", "observation_date"])
y = df["recession"]

#Chronological Train/ Test Split (80/20)
split_index = int(len(df) * 0.8)

x_train = X[:split_index]
x_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("Train size:", len(x_train))
print("Test size: ", len(x_test))

#Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled  = scaler.transform(x_test)

#-----------------------------------
#Evaluation Function
def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n" + "*" * 50)
    print(model_name)
    print("*" * 50)
    print("Accuracy: ", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
    print("Recall:   ", round(recall_score(y_test, y_pred, zero_division=0), 4))
    print("F1:       ", round(f1_score(y_test, y_pred, zero_division=0), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

#-------------------------------
# Train/ Test Split Results
#Decision tree
dt_model = DecisionTreeClassifier(class_weight="balanced",
                                  max_depth=3,
                                  min_samples_leaf=5,
                                  min_samples_split=10,
                                  random_state=42)
evaluate_model("Decision Tree Train/Test Results", dt_model, X_train_scaled, X_test_scaled, y_train, y_test)
# The Decision Tree caught 1 out of the 2 recession months, so recall = 0.5.
# But it still predicted many non-recession months as recession which resulted in 89 false positives
# accuracy and precision are low due to the false positives
# Decision Tree is better than kNN at detecting recession months, but it creates many false alarms


#kNN
knn_model = KNeighborsClassifier(n_neighbors=7)
evaluate_model("kNN Train/Test Results", knn_model, X_train_scaled, X_test_scaled, y_train, y_test)
#Initially n_neighbors was set to 5
#Increasing it to 7 reduced false positives(from 32 to 15), which then improved accuracy
# higher accuracy than the Decision Tree because it correctly predicted
# many non-recession months. But it still missed both actual recession months.

#------------------------------------
# Time-Series Cross-Validation
# The goal is to avoid leakage from future data into the training set

tscv = TimeSeriesSplit(n_splits=5)

def cross_validate_model(model_name, model, X, y):
    cv_accuracy, cv_precision, cv_recall, cv_f1 = [], [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_fold_train = X.iloc[train_idx]
        X_fold_test = X.iloc[test_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_test = y.iloc[test_idx]

        fold_scaler = StandardScaler()
        X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
        X_fold_test_scaled = fold_scaler.transform(X_fold_test)

        model.fit(X_fold_train_scaled, y_fold_train)
        y_pred = model.predict(X_fold_test_scaled)

        cv_accuracy.append(accuracy_score(y_fold_test, y_pred))
        cv_precision.append(precision_score(y_fold_test, y_pred, zero_division=0))
        cv_recall.append(recall_score(y_fold_test, y_pred, zero_division=0))
        cv_f1.append(f1_score(y_fold_test, y_pred, zero_division=0))

    print("\n" + "*" * 50)
    print(model_name + " Cross-Validation Results")
    print("*" * 50)
    print("Average Accuracy: ", round(np.mean(cv_accuracy), 4))
    print("Average Precision:", round(np.mean(cv_precision), 4))
    print("Average Recall:   ", round(np.mean(cv_recall), 4))
    print("Average F1:       ", round(np.mean(cv_f1), 4))

cross_validate_model("Decision Tree", dt_model, X, y)
cross_validate_model("kNN", knn_model, X, y)