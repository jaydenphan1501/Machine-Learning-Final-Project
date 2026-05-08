import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
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
