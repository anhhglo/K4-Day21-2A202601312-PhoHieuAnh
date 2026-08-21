import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.serve import create_app


class FakeModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, rows):
        assert len(rows[0]) == 10
        return np.array([self.prediction])


def test_module_exposes_an_app_without_starting_model_loading():
    from src.serve import app

    assert isinstance(app, FastAPI)


def test_health_and_low_income_score():
    app = create_app(lambda: FakeModel(0))

    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        response = client.post("/score", json={"features": [0.0] * 10})

    assert response.status_code == 200
    assert response.json() == {"prediction": 0, "label": "thu_nhap_thap"}


def test_score_returns_high_income_label():
    app = create_app(lambda: FakeModel(1))

    with TestClient(app) as client:
        response = client.post("/score", json={"features": [1.0] * 10})

    assert response.status_code == 200
    assert response.json() == {"prediction": 1, "label": "thu_nhap_cao"}


def test_score_rejects_requests_without_exactly_ten_features():
    app = create_app(lambda: FakeModel(0))

    with TestClient(app) as client:
        response = client.post("/score", json={"features": [0.0] * 9})

    assert response.status_code == 400
