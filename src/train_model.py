import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
data = pd.read_csv("data/churn.csv")

print("Dataset Loaded Successfully")
print(data.head())

# Drop customerID
if "customerID" in data.columns:
    data = data.drop("customerID", axis=1)

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

# Handle missing values
data = data.fillna(0)

# Convert categorical variables using One-Hot Encoding
data = pd.get_dummies(data)

# Separate features and target
X = data.drop("Churn_Yes", axis=1)
y = data["Churn_Yes"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

print("Model Training Completed")

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save model
# Save model
joblib.dump(model, "model/churn_model.pkl")

# Save feature columns
joblib.dump(X.columns, "model/feature_columns.pkl")

print("Model and feature columns saved successfully")