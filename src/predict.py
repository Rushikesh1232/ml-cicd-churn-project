import joblib
import pandas as pd

# Load model
model = joblib.load("model/churn_model.pkl")

# Load training columns
feature_columns = joblib.load("model/feature_columns.pkl")


def predict_churn(input_data):

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # Apply same encoding
    df = pd.get_dummies(df)

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure same column order
    df = df[feature_columns]

    prediction = model.predict(df)

    if prediction[0] == 1:
        return "Customer will churn"
    else:
        return "Customer will stay"


if __name__ == "__main__":

    sample_customer = {
        "SeniorCitizen": 0,
        "tenure": 5,
        "MonthlyCharges": 70,
        "TotalCharges": 350
    }

    result = predict_churn(sample_customer)

    print(result)