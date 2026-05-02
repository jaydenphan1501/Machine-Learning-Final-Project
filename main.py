import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv("data/recession_data.csv")
df = df.drop("Unnamed: 0", axis=1)
print("Dataset loaded successfully.")
print("Columns:", df.columns.tolist())
print("Dataset shape:", df.shape)

# Separate features and target
X = df.drop("Recession", axis=1)
y = df["Recession"]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)
print("Target counts:")
print(y.value_counts())

#data was found imbalanced after those lines above were run
#that means there are more "no recession" than "recession"
#focus on Recall, F1-score, and confusion matrix, not just
#accuracy alone

#Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train counts:")
print(y_train.value_counts())
print("y_test counts:")
print(y_test.value_counts())

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data split and scaling completed.")

# Train Logistic Regression model
log_model = LogisticRegression(max_iter=1000, class_weight="balanced")

log_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = log_model.predict(X_test_scaled)

# Evaluate model
print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

#high accuracy, but the result is misleading
#the dataset is highly imbalanced, with many more no-recession cases than recession cases.
#predicted every test sample as no recession, so it correctly classified most
#normal periods but failed to detect any recession cases.
#Since recall and F1-score for the recession class were 0, accuracy alone is not a
#reliable metric.

#we should try other models like Decision Tree, Random Forest, and KNN and probably use
#a different dataset. This is just a draft i did for fun