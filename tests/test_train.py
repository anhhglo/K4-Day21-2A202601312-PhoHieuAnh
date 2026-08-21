import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.train import train


FEATURE_COLUMNS = [f"feature_{index}" for index in range(10)]
PARAMS = {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2}


def _two_class_frame(rows=200):
    generator = np.random.default_rng(42)
    features = generator.normal(size=(rows, len(FEATURE_COLUMNS)))
    frame = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    frame["target"] = (features[:, 0] + features[:, 1] > 0).astype(int)
    return frame


def _write_datasets(tmp_path, frame=None):
    frame = _two_class_frame() if frame is None else frame
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "holdout.csv"
    frame.iloc[:150].to_csv(train_path, index=False)
    frame.iloc[150:].to_csv(eval_path, index=False)
    return train_path, eval_path


def test_train_writes_metrics_report_and_serialized_model(tmp_path, monkeypatch):
    train_path, eval_path = _write_datasets(tmp_path)
    monkeypatch.chdir(tmp_path)

    f1 = train(PARAMS, data_path=str(train_path), eval_path=str(eval_path))

    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0
    report = json.loads(Path("outputs/report.json").read_text())
    assert set(report) == {"f1_score", "accuracy"}
    assert all(isinstance(value, float) for value in report.values())
    assert joblib.load("models/model.joblib").n_features_in_ == 10


@pytest.mark.parametrize("dataset_name", ["training", "evaluation"])
def test_train_rejects_dataset_without_target_column(tmp_path, dataset_name):
    train_path, eval_path = _write_datasets(tmp_path)
    path_without_target = train_path if dataset_name == "training" else eval_path
    pd.read_csv(path_without_target).drop(columns="target").to_csv(
        path_without_target, index=False
    )

    with pytest.raises(
        ValueError, match=rf"^{dataset_name} data must contain a 'target' column$"
    ):
        train(PARAMS, data_path=str(train_path), eval_path=str(eval_path))


def test_train_rejects_mismatched_feature_columns(tmp_path):
    train_path, eval_path = _write_datasets(tmp_path)
    eval_frame = pd.read_csv(eval_path).rename(columns={"feature_9": "different"})
    eval_frame.to_csv(eval_path, index=False)

    with pytest.raises(
        ValueError, match="^training and evaluation feature columns must match$"
    ):
        train(PARAMS, data_path=str(train_path), eval_path=str(eval_path))


def test_train_rejects_one_class_training_target(tmp_path):
    train_path, eval_path = _write_datasets(tmp_path)
    train_frame = pd.read_csv(train_path)
    train_frame["target"] = 0
    train_frame.to_csv(train_path, index=False)

    with pytest.raises(
        ValueError, match="^training target must contain both classes 0 and 1$"
    ):
        train(PARAMS, data_path=str(train_path), eval_path=str(eval_path))
