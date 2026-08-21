import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage
from pydantic import BaseModel, field_validator


MODEL_OBJECT = "artifacts/current/model.joblib"
LABELS = {0: "thu_nhap_thap", 1: "thu_nhap_cao"}


class ScoreRequest(BaseModel):
    features: list[float]

    @field_validator("features", mode="before")
    @classmethod
    def validate_numeric_features(cls, features: object) -> object:
        if not isinstance(features, list) or any(
            isinstance(feature, bool) or not isinstance(feature, (int, float))
            for feature in features
        ):
            raise ValueError("features must contain only JSON numbers")
        return features


def download_model() -> Path:
    bucket_name = os.environ["ARTIFACT_BUCKET"]
    model_path = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))
    temp_path = model_path.with_name(f"{model_path.name}.tmp")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        storage.Client().bucket(bucket_name).blob(MODEL_OBJECT).download_to_filename(
            str(temp_path)
        )
        os.replace(temp_path, model_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return model_path


def load_runtime_model() -> Any:
    return joblib.load(download_model())


def create_app(model_loader: Callable[[], Any] = load_runtime_model) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model_loader()
        yield

    api = FastAPI(lifespan=lifespan)

    @api.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @api.post("/score")
    def score(req: ScoreRequest, request: Request):
        if len(req.features) != 10:
            raise HTTPException(status_code=400, detail="exactly 10 features are required")

        prediction = int(request.app.state.model.predict([req.features])[0])
        return {"prediction": prediction, "label": LABELS[prediction]}

    return api


app = create_app()
