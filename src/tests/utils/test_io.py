import numpy as np
import pandas as pd

from mlutils.utils.io import find_git_root, read_local_data, remove_high_null_features, split_train_test


def test_read_local_data_classify(monkeypatch, sample_csv_classify, sample_dataset_classify):
    # Mock pd.read_csv to return the sample CSV DataFrame
    monkeypatch.setattr("pandas.read_csv", lambda path: sample_csv_classify.copy())

    X, y = read_local_data(sample_dataset_classify)

    # Check that 'customerID' is now the index
    assert X.index.name == "customerID"

    # No custom cleaning happening
    assert X.loc[2, "TotalCharges"] == " "  # was empty, replaced with MonthlyCharges
    assert X.loc[4, "TotalCharges"] == " "

    # Check if target values were mapped correctly
    assert y[1] == 0
    assert y[2] == 1

    # Check dimensions
    assert X.shape[0] == 4
    assert "Churn" not in X.columns


def test_read_local_data_regression(monkeypatch, sample_csv_regression, sample_dataset_regression):
    # Mock pd.read_csv to return the sample CSV DataFrame
    monkeypatch.setattr("pandas.read_csv", lambda path: sample_csv_regression.copy())

    X, y = read_local_data(sample_dataset_regression)

    # Check that 'customerID' is now the index
    assert X.index.name == "ID"

    # Check no null value in y
    assert np.sum(y.isna()) == 0

    # Check if target values were mapped correctly
    assert y[1] == 1100
    assert y[2] == 1200

    # Check dimensions
    assert X.shape[0] == 4
    assert "Price" not in X.columns
    assert "Useless" not in X.columns


def test_split_train_test():
    # Create mock data
    X = pd.DataFrame({"A": range(10), "B": range(10, 20)})
    y = pd.Series([0, 1] * 5)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2


def test_find_git_root(tmp_path, monkeypatch):
    project = tmp_path / "project"
    git_dir = project / ".git"
    subdir = project / "subdir"

    git_dir.mkdir(parents=True)
    subdir.mkdir()

    # Change current working directory to subdir
    monkeypatch.chdir(subdir)

    result = find_git_root()

    assert result == project


def test_remove_high_null_features_basic_case():
    data = {"A": [1, None, None, None], "B": [1, 2, 3, 4], "C": [None, None, None, None], "D": [1, None, 3, 4]}
    df = pd.DataFrame(data)

    result = remove_high_null_features(df, threshold=0.5)

    # Columns A and C should be dropped (75% and 100% nulls respectively)
    assert "A" not in result.columns
    assert "C" not in result.columns
    assert "B" in result.columns
    assert "D" in result.columns
    assert result.shape[1] == 2  # Only B and D should remain


def test_remove_high_null_features_all_below_threshold():
    df = pd.DataFrame({"A": [1, 2, None], "B": [4, 5, 6]})
    result = remove_high_null_features(df, threshold=0.8)
    assert result.equals(df)  # No column should be removed


def test_remove_high_null_features_all_above_threshold():
    df = pd.DataFrame({"A": [None, None, None], "B": [None, 1, None]})
    result = remove_high_null_features(df, threshold=0.3)
    # Both should be dropped
    assert result.empty
    assert result.shape[1] == 0


def test_remove_high_null_features_empty_df():
    df = pd.DataFrame()
    result = remove_high_null_features(df)
    assert result.empty


def test_remove_high_null_features_no_nulls():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = remove_high_null_features(df)
    assert result.equals(df)
