# ==============================
# AI Financial Fraud Detection
# ==============================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------
# 1. Load Dataset
# ------------------------------

print("Loading dataset...")

# safer CSV loading
data = pd.read_csv("creditcard.csv", on_bad_lines='skip')

print("Dataset loaded successfully!\n")

print("First 5 rows:")
print(data.head())

# ------------------------------
# 2. Dataset Information
# ------------------------------

print("\nDataset Info:\n")
print(data.info())

print("\nMissing Values:\n")
print(data.isnull().sum())

# ------------------------------
# 3. Fraud vs Normal Distribution
# ------------------------------

print("\nTransaction Distribution:\n")
print(data['Class'].value_counts())

plt.figure()
sns.countplot(x='Class', data=data)
plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Count")
plt.show()

# ------------------------------
# 4. Feature Scaling
# ------------------------------

scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])

# ------------------------------
# 5. Define Features & Target
# ------------------------------

X = data.drop('Class', axis=1)
y = data['Class']

# ------------------------------
# 6. Train-Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ------------------------------
# 7. Train Model
# ------------------------------

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed!")

# ------------------------------
# 8. Predictions
# ------------------------------

y_pred = model.predict(X_test)

# ------------------------------
# 9. Model Evaluation
# ------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ------------------------------
# 10. Confusion Matrix
# ------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------------------
# 11. Example Fraud Prediction
# ------------------------------

print("\nTesting Example Transaction...")

sample = X_test.iloc[0].values.reshape(1, -1)

prediction = model.predict(sample)

if prediction[0] == 0:
    print("Result: Normal Transaction")
else:
    print("Result: Fraudulent Transaction Detected!")