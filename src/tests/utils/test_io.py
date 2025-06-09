import numpy as np
import pandas as pd

from mlutils.utils.io import read_local_data, split_train_test  # Replace with actual module name


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
