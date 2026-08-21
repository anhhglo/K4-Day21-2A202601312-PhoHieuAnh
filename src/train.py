import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score


EXPERIMENT_NAME = "adult-income"


def _load_and_validate(data_path, eval_path):
    train_frame = pd.read_csv(data_path)
    eval_frame = pd.read_csv(eval_path)

    for dataset_name, frame in (("training", train_frame), ("evaluation", eval_frame)):
        if "target" not in frame:
            raise ValueError(f"{dataset_name} data must contain a 'target' column")

    train_features = train_frame.drop(columns="target")
    eval_features = eval_frame.drop(columns="target")
    if list(train_features.columns) != list(eval_features.columns):
        raise ValueError("training and evaluation feature columns must match")

    for dataset_name, target in (
        ("training", train_frame["target"]),
        ("evaluation", eval_frame["target"]),
    ):
        if set(target) != {0, 1}:
            raise ValueError(f"{dataset_name} target must contain both classes 0 and 1")

    return train_features, train_frame["target"], eval_features, eval_frame["target"]


def _configure_experiment():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=Path("mlartifacts").resolve().as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    x_train, y_train, x_eval, y_eval = _load_and_validate(data_path, eval_path)

    model = GradientBoostingClassifier(**params, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_eval)
    f1 = float(f1_score(y_eval, predictions))
    accuracy = float(accuracy_score(y_eval, predictions))

    _configure_experiment()
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("outputs/report.json").write_text(
        json.dumps({"f1_score": f1, "accuracy": accuracy}) + "\n",
        encoding="utf-8",
    )
    joblib.dump(model, "models/model.joblib")
    print(f"f1_score: {f1}")
    print(f"accuracy: {accuracy}")
    return f1


if __name__ == "__main__":
    with Path("params.yaml").open(encoding="utf-8") as params_file:
        train(yaml.safe_load(params_file))
