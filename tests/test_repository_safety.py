from pathlib import Path
from subprocess import run


REQUIRED_IGNORES = {
    ".venv/",
    ".secrets/",
    ".dvc/config.local",
    "data/*.csv",
    "models/",
    "outputs/",
    "mlflow.db",
    "mlartifacts/",
    "sa-key.json",
    "*.pem",
    "*.key",
}

DVC_DATASET_NEGATIONS = {
    "!data/train_batch1.csv",
    "!data/holdout.csv",
    "!data/train_batch2.csv",
}

DATASET_IGNORES = {
    "/train_batch1.csv",
    "/holdout.csv",
    "/train_batch2.csv",
}


def test_sensitive_and_generated_files_are_ignored():
    rules = {
        line.strip()
        for line in Path(".gitignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert REQUIRED_IGNORES <= rules
    assert DVC_DATASET_NEGATIONS <= rules


def test_dvc_managed_dataset_csvs_remain_ignored():
    rules = {
        line.strip()
        for line in Path("data/.gitignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert DATASET_IGNORES <= rules


def test_all_dataset_csvs_remain_ignored():
    paths = [
        "data/new.csv",
        "data/train_batch1.csv",
        "data/holdout.csv",
        "data/train_batch2.csv",
    ]
    result = run(["git", "check-ignore", *paths], check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == set(paths)
