import os
import tempfile
from pathlib import Path

import pandas as pd

from prepare_data import CANONICAL_COLUMNS


def _read_canonical_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != CANONICAL_COLUMNS:
        raise ValueError(f"{path.name} does not use the canonical schema")
    return frame


def append_batch(path1: str | Path, path2: str | Path) -> None:
    """Atomically append an equal-sized second batch exactly once."""
    batch1_path = Path(path1)
    batch2_path = Path(path2)
    batch1 = _read_canonical_csv(batch1_path)
    batch2 = _read_canonical_csv(batch2_path)

    if len(batch1) == 2 * len(batch2) and batch1.iloc[-len(batch2) :].reset_index(
        drop=True
    ).equals(batch2.reset_index(drop=True)):
        raise ValueError("train_batch1 already contains batch2")
    if len(batch1) != len(batch2):
        raise ValueError("train_batch1 and train_batch2 must have equal row counts")

    appended = pd.concat([batch1, batch2], ignore_index=True)
    expected_rows = len(batch1) + len(batch2)
    if len(appended) != expected_rows:
        raise ValueError("appended train_batch1 row count is incorrect")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=f".{batch1_path.stem}-", dir=batch1_path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        appended.to_csv(temporary_file, index=False)

    try:
        os.replace(temporary_path, batch1_path)
    finally:
        temporary_path.unlink(missing_ok=True)
