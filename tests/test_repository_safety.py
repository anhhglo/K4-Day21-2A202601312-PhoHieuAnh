from pathlib import Path


REQUIRED_IGNORES = {
    ".venv/",
    ".secrets/",
    ".dvc/config.local",
    "models/",
    "outputs/",
    "mlflow.db",
    "mlartifacts/",
    "sa-key.json",
    "*.pem",
    "*.key",
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


def test_dvc_managed_dataset_csvs_remain_ignored():
    rules = {
        line.strip()
        for line in Path("data/.gitignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert DATASET_IGNORES <= rules
