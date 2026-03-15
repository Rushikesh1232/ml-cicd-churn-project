from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import joblib
import pandas as pd

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

model = joblib.load("model/churn_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict_form")
def predict_form(
    SeniorCitizen: int = Form(...),
    tenure: int = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...)
):

    data = {
        "SeniorCitizen": SeniorCitizen,
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    prediction = model.predict(df)

    if prediction[0] == 1:
        return {"Prediction": "Customer will churn"}
    else:
        return {"Prediction": "Customer will stay"}