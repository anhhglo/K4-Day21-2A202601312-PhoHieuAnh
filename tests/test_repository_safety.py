from pathlib import Path


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


def test_sensitive_and_generated_files_are_ignored():
    rules = {
        line.strip()
        for line in Path(".gitignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert REQUIRED_IGNORES <= rules
