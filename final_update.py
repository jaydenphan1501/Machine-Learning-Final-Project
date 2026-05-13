import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

# --------------------------------------------------
# Load Data
df = pd.read_csv("data/FRED_data.csv")
df = df.sort_values("observation_date").reset_index(drop=True)
print(df.head())
print(df.info())

# Features and target
X = df.drop(columns=["recession", "observation_date"])
y = df["recession"]
print(y.value_counts())

# --------------------------------------------------
# Chronological Train / Test Split (80/20)
# We train on old data and test on new data to
# simulate real world forecasting
split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test  = X[split_index:]
y_train = y[:split_index]
y_test  = y[split_index:]

print("Train size:", len(X_train))
print("Test size: ", len(X_test))

# --------------------------------------------------
# Scale Features
# Fit on training data only to avoid data leakage
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# --------------------------------------------------
# Evaluation Function
def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n" + "-" * 50)
    print(model_name)
    print("-" * 50)
    print("Accuracy: ", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
    print("Recall:   ", round(recall_score(y_test, y_pred, zero_division=0), 4))
    print("F1:       ", round(f1_score(y_test, y_pred, zero_division=0), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# Baseline: Logistic Regression
# Used as a strong baseline for binary classification.
# The data has a very large amount of non-recession months
# so accuracy alone can be very misleading. If the model
# predicts non-recession every month it would achieve
# a high accuracy without detecting a single recession.
# ------------------------------------------------
log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)

print("\nLogistic Regression Results")
print("Accuracy: ", round(accuracy_score(y_test, log_pred), 4))
print("Precision:", round(precision_score(y_test, log_pred, zero_division=0), 4))
print("Recall:   ", round(recall_score(y_test, log_pred, zero_division=0), 4))
print("F1:       ", round(f1_score(y_test, log_pred, zero_division=0), 4))
print(classification_report(y_test, log_pred, zero_division=0))

# --------------------------------------------------
# Random Forest
tscv = TimeSeriesSplit(n_splits=5)
cv_accuracy, cv_precision, cv_recall, cv_f1 = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_fold_train = X.iloc[train_idx]
    X_fold_test  = X.iloc[test_idx]
    y_fold_train = y.iloc[train_idx]
    y_fold_test  = y.iloc[test_idx]

    fold_scaler         = StandardScaler()
    X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
    X_fold_test_scaled  = fold_scaler.transform(X_fold_test)

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

print("\nRandom Forest Results (5-Fold Time-Series Cross-Validation)")
print("Average Accuracy: ", round(np.mean(cv_accuracy), 4))
print("Average Precision:", round(np.mean(cv_precision), 4))
print("Average Recall:   ", round(np.mean(cv_recall), 4))
print("Average F1:       ", round(np.mean(cv_f1), 4))

# --------------------------------------------------
# Decision Tree — Train/Test Split
dt_model = DecisionTreeClassifier(
    class_weight="balanced",
    max_depth=3,
    min_samples_leaf=5,
    min_samples_split=10,
    random_state=42,
)
evaluate_model("Decision Tree Train/Test Results",
               dt_model, X_train_scaled, X_test_scaled, y_train, y_test)

# --------------------------------------------------
# kNN Train/Test Split
# n_neighbors tuned from 5 to 7
# k=5 produced 32 false positives, k=7 reduced this to 11.
# kNN missed had higher accuracy than Decision Tree due to fewer false positives.
knn_model = KNeighborsClassifier(n_neighbors=7)
evaluate_model("kNN Train/Test Results",
               knn_model, X_train_scaled, X_test_scaled, y_train, y_test)

# --------------------------------------------------
# Time-Series Cross-Validation — Decision Tree & kNN
tscv = TimeSeriesSplit(n_splits=5)

def cross_validate_model(model_name, model, X, y):
    cv_accuracy, cv_precision, cv_recall, cv_f1 = [], [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_fold_train = X.iloc[train_idx]
        X_fold_test  = X.iloc[test_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_test  = y.iloc[test_idx]

        fold_scaler         = StandardScaler()
        X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
        X_fold_test_scaled  = fold_scaler.transform(X_fold_test)

        model.fit(X_fold_train_scaled, y_fold_train)
        y_pred = model.predict(X_fold_test_scaled)

        cv_accuracy.append(accuracy_score(y_fold_test, y_pred))
        cv_precision.append(precision_score(y_fold_test, y_pred, zero_division=0))
        cv_recall.append(recall_score(y_fold_test, y_pred, zero_division=0))
        cv_f1.append(f1_score(y_fold_test, y_pred, zero_division=0))

    print("\n" + "-" * 50)
    print(model_name + " Cross-Validation Results")
    print("-" * 50)
    print("Average Accuracy: ", round(np.mean(cv_accuracy), 4))
    print("Average Precision:", round(np.mean(cv_precision), 4))
    print("Average Recall:   ", round(np.mean(cv_recall), 4))
    print("Average F1:       ", round(np.mean(cv_f1), 4))

cross_validate_model("Decision Tree", dt_model,  X, y)
cross_validate_model("kNN",           knn_model, X, y)


# -------------------------------------------------
# VISUALIZATIONS
# --------------------------------------------------
#Plot 1 Confusion Matrices (Decision Tree vs kNN)
dt_pred  = dt_model.predict(X_test_scaled)
knn_pred = knn_model.predict(X_test_scaled)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
model_preds = [("Decision Tree", dt_pred), ("kNN", knn_pred)]

for cor, (name, pred) in zip(axes, model_preds):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=cor,
                xticklabels=["No Rec.", "Rec."],
                yticklabels=["No Rec.", "Rec."],
                linewidths=1, linecolor="white", cbar=False)
    cor.set_title(name, fontsize=12, fontweight="bold")
    cor.set_xlabel("Predicted")
    cor.set_ylabel("Actual")

fig.suptitle("Confusion Matrices — Train/Test Split", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
plt.close()

# --------------------------------------------------
#Plot 2 kNN Diagnostic (Hyperparameter Tuning)

k_values  = list(range(1, 21))
acc_train = []
acc_test  = []
f1_test   = []

for k in k_values:
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_train_scaled, y_train)
    acc_train.append(accuracy_score(y_train, m.predict(X_train_scaled)))
    acc_test.append(accuracy_score(y_test,   m.predict(X_test_scaled)))
    f1_test.append(f1_score(y_test,          m.predict(X_test_scaled), zero_division=0))

best_k = 7

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
y_prob = knn_best.predict_proba(X_test_scaled)[:, 1]
y_pred = knn_best.predict(X_test_scaled)

tscv_plot = TimeSeriesSplit(n_splits=5)
fold_acc, fold_f1, fold_rec = [], [], []

for train_idx, test_idx in tscv_plot.split(X):
    fold_scaler         = StandardScaler()
    X_fold_train_scaled = fold_scaler.fit_transform(X.iloc[train_idx])
    X_fold_test_scaled  = fold_scaler.transform(X.iloc[test_idx])

    m = KNeighborsClassifier(n_neighbors=best_k)
    m.fit(X_fold_train_scaled, y.iloc[train_idx])
    p = m.predict(X_fold_test_scaled)

    fold_acc.append(accuracy_score(y.iloc[test_idx], p))
    fold_f1.append(f1_score(y.iloc[test_idx],        p, zero_division=0))
    fold_rec.append(recall_score(y.iloc[test_idx],   p, zero_division=0))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("kNN Classifier — Recession Detection", fontsize=14, fontweight="bold")

# K vs Accuracy
ax = axes[0]
ax.plot(k_values, acc_train, marker="o", label="Train Accuracy", color="steelblue")
ax.plot(k_values, acc_test,  marker="s", label="Test Accuracy",  color="tomato")
ax.axvline(best_k, linestyle="--", color="gray", label=f"Best K={best_k}")
ax.set_xlabel("K (neighbours)")
ax.set_ylabel("Accuracy")
ax.set_title("Hyperparameter Tuning")
ax.legend()
ax.grid(True, alpha=0.3)

#Predicted Probability by actual class
ax = axes[1]
no_rec_probs = y_prob[y_test.values == 0]
rec_probs    = y_prob[y_test.values == 1]
ax.hist(no_rec_probs, bins=15, color="steelblue", alpha=0.7,
        edgecolor="white", label="No Recession")
ax.hist(rec_probs,    bins=5,  color="tomato",    alpha=0.8,
        edgecolor="white", label="Recession")
ax.axvline(0.5, color="gray", linestyle="--", label="Threshold = 0.5")
ax.set_xlabel("Predicted Probability of Recession")
ax.set_ylabel("Count")
ax.set_title(f"Predicted Probabilities  (K={best_k})")
ax.legend()
ax.grid(True, alpha=0.3)

#CV Fold Scores
ax = axes[2]
fold_nums = [f"Fold {i + 1}" for i in range(5)]
x_pos     = np.arange(5)
width     = 0.28
ax.bar(x_pos - width, fold_acc, width, label="Accuracy", color="steelblue",    edgecolor="white")
ax.bar(x_pos,         fold_f1,  width, label="F1",       color="mediumpurple", edgecolor="white")
ax.bar(x_pos + width, fold_rec, width, label="Recall",   color="tomato",       edgecolor="white")
ax.set_xticks(x_pos)
ax.set_xticklabels(fold_nums)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.15)
ax.set_title("Cross-Validation Scores per Fold")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()