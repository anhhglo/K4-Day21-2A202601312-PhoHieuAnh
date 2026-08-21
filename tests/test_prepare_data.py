import pandas as pd
import pytest

from append_batch import append_batch
from prepare_data import FEATURE_NAMES, RAW_COLUMNS, prepare_frames, split_frames


def _raw_train_fixture():
    return pd.DataFrame(
        [
            [39, " Private ", 77516, "Bachelors", 13, "Never-married", "Adm-clerical", "Not-in-family", "White", "Male", 2174, 0, 40, "United-States", ">50K"],
            [50, " Government ", 83311, "Bachelors", 13, "Married-civ-spouse", "Exec-managerial", "Husband", "White", "Male", 0, 0, 13, "United-States", "<=50K"],
            [38, " ? ", 215646, "HS-grad", 9, "Divorced", "Handlers-cleaners", "Not-in-family", "White", "Male", 0, 0, 40, "United-States", "<=50K"],
        ],
        columns=RAW_COLUMNS,
    )


def _raw_test_fixture():
    return pd.DataFrame(
        [
            [53, "Private", 234721, "11th", 7, "Married-civ-spouse", "Handlers-cleaners", "Husband", "Black", "Male", 0, 0, 40, "United-States", ">50K."],
        ],
        columns=RAW_COLUMNS,
    )


def _canonical_frame(rows):
    return pd.DataFrame(
        {
            "age": list(range(20, 20 + rows)),
            "workclass": [0] * rows,
            "education_num": [9] * rows,
            "marital_status": [0] * rows,
            "occupation": [0] * rows,
            "relationship": [0] * rows,
            "sex": [0] * rows,
            "capital_gain": [0] * rows,
            "capital_loss": [0] * rows,
            "hours_per_week": [40] * rows,
            "target": [index % 2 for index in range(rows)],
        }
    )


def test_prepare_frames_encodes_categories_alphabetically():
    cleaned = prepare_frames(_raw_train_fixture(), _raw_test_fixture())

    assert list(cleaned.columns) == FEATURE_NAMES + ["target"]
    assert set(cleaned["target"]) == {0, 1}
    assert cleaned["workclass"].tolist() == [1, 0, 1]


def test_split_frames_uses_fixed_holdout_and_equal_batches():
    frame = _canonical_frame(10)

    batch1, holdout, batch2 = split_frames(
        frame, holdout_size=2, random_state=42
    )

    assert (len(batch1), len(holdout), len(batch2)) == (4, 2, 4)
    assert set(batch1.index).isdisjoint(holdout.index)
    assert set(batch2.index).isdisjoint(holdout.index)


def test_append_batch_concatenates_once_and_rejects_duplicate_append(tmp_path):
    batch1_path = tmp_path / "train_batch1.csv"
    batch2_path = tmp_path / "train_batch2.csv"
    _canonical_frame(4).to_csv(batch1_path, index=False)
    _canonical_frame(4).assign(age=lambda frame: frame["age"] + 100).to_csv(
        batch2_path, index=False
    )

    append_batch(batch1_path, batch2_path)

    appended = pd.read_csv(batch1_path)
    assert len(appended) == 8
    assert appended["age"].tolist() == [20, 21, 22, 23, 120, 121, 122, 123]
    with pytest.raises(ValueError, match="^train_batch1 already contains batch2$"):
        append_batch(batch1_path, batch2_path)


def test_append_batch_rejects_identical_batch_files(tmp_path):
    batch1_path = tmp_path / "train_batch1.csv"
    batch2_path = tmp_path / "train_batch2.csv"
    batch = _canonical_frame(4)
    batch.to_csv(batch1_path, index=False)
    batch.to_csv(batch2_path, index=False)

    with pytest.raises(ValueError, match="^train_batch1 already contains batch2$"):
        append_batch(batch1_path, batch2_path)

    assert len(pd.read_csv(batch1_path)) == 4


def test_append_batch_rejects_contained_rows_in_a_different_order(tmp_path):
    batch1_path = tmp_path / "train_batch1.csv"
    batch2_path = tmp_path / "train_batch2.csv"
    batch1 = _canonical_frame(4)
    batch2 = batch1.sample(frac=1, random_state=7).reset_index(drop=True)
    batch1.to_csv(batch1_path, index=False)
    batch2.to_csv(batch2_path, index=False)

    with pytest.raises(ValueError, match="^train_batch1 already contains batch2$"):
        append_batch(batch1_path, batch2_path)

    assert len(pd.read_csv(batch1_path)) == 4


def test_append_batch_main_uses_default_paths_and_rejects_second_run(
    tmp_path, monkeypatch
):
    from append_batch import main

    data_directory = tmp_path / "data"
    data_directory.mkdir()
    batch1_path = data_directory / "train_batch1.csv"
    batch2_path = data_directory / "train_batch2.csv"
    _canonical_frame(4).to_csv(batch1_path, index=False)
    _canonical_frame(4).assign(age=lambda frame: frame["age"] + 100).to_csv(
        batch2_path, index=False
    )
    monkeypatch.chdir(tmp_path)

    main()

    assert len(pd.read_csv(batch1_path)) == 8
    with pytest.raises(ValueError, match="^train_batch1 already contains batch2$"):
        main()
