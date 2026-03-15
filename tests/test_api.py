import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_prediction():

    sample_data = {
        "SeniorCitizen": 0,
        "tenure": 10,
        "MonthlyCharges": 50,
        "TotalCharges": 500
    }

    response = client.post("/predict_form", data=sample_data)

    assert response.status_code == 200