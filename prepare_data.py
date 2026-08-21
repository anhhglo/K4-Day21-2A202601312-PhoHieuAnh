from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

RAW_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "target",
]

FEATURE_NAMES = [
    "age",
    "workclass",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_COLUMNS = [
    "workclass", "marital_status", "occupation", "relationship", "sex"
]
CANONICAL_COLUMNS = FEATURE_NAMES + ["target"]
PRODUCTION_ROW_COUNT = 45_222
HOLDOUT_SIZE = 500


def prepare_frames(raw_train: pd.DataFrame, raw_test: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Adult Census frames into the canonical model schema."""
    combined = pd.concat([raw_train, raw_test], ignore_index=True)
    combined = combined.replace(r"^\s*\?\s*$", np.nan, regex=True).dropna()
    combined["target"] = (
        combined["target"]
        .astype(str)
        .str.strip()
        .str.rstrip(".")
        .eq(">50K")
        .astype(int)
    )

    for column in CATEGORICAL_COLUMNS:
        values = sorted(combined[column].astype(str).str.strip().unique())
        combined[column] = combined[column].astype(str).str.strip().map(
            {value: index for index, value in enumerate(values)}
        )

    return combined.loc[:, CANONICAL_COLUMNS].reset_index(drop=True)


def split_frames(
    frame: pd.DataFrame, holdout_size: int = HOLDOUT_SIZE, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle once, reserve a holdout, then divide the remainder equally."""
    if holdout_size < 0 or holdout_size >= len(frame):
        raise ValueError("holdout_size must be smaller than the frame")

    shuffled = frame.sample(frac=1, random_state=random_state)
    holdout = shuffled.iloc[:holdout_size]
    remaining = shuffled.iloc[holdout_size:]
    if len(remaining) % 2:
        raise ValueError("remaining rows must divide into equal training batches")

    midpoint = len(remaining) // 2
    return remaining.iloc[:midpoint], holdout, remaining.iloc[midpoint:]


def _download_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_train = pd.read_csv(TRAIN_URL, names=RAW_COLUMNS, skipinitialspace=True)
    raw_test = pd.read_csv(
        TEST_URL, names=RAW_COLUMNS, skipinitialspace=True, skiprows=1
    )
    return raw_train, raw_test


def main() -> None:
    raw_train, raw_test = _download_frames()
    cleaned = prepare_frames(raw_train, raw_test)
    if len(cleaned) != PRODUCTION_ROW_COUNT:
        raise ValueError(
            f"expected {PRODUCTION_ROW_COUNT} cleaned rows, found {len(cleaned)}"
        )

    batch1, holdout, batch2 = split_frames(cleaned)
    output_directory = Path("data")
    output_directory.mkdir(exist_ok=True)
    batch1.to_csv(output_directory / "train_batch1.csv", index=False)
    holdout.to_csv(output_directory / "holdout.csv", index=False)
    batch2.to_csv(output_directory / "train_batch2.csv", index=False)

    print(f"train_batch1 rows: {len(batch1)}")
    print(f"holdout rows: {len(holdout)}")
    print(f"train_batch2 rows: {len(batch2)}")
    print(f"positive-class rate: {cleaned['target'].mean():.1%}")


if __name__ == "__main__":
    main()
