from io import StringIO

import pandas as pd
import pytest

from mlutils.utils.io import read_local_data, split_train_test  # Replace with actual module name


@pytest.fixture
def sample_csv():
    data = StringIO(
        """customerID,TotalCharges,MonthlyCharges,Churn
1,29.85,29.85,No
2, ,56.95,Yes
3,1889.5,53.85,No
4, ,42.30,Yes
"""
    )
    return pd.read_csv(data)


def test_read_local_data(monkeypatch, sample_csv):
    # Mock pd.read_csv to return the sample CSV DataFrame
    monkeypatch.setattr("pandas.read_csv", lambda path: sample_csv.copy())

    X, y = read_local_data("dummy_path.csv", target_col="Churn")

    # Check that 'customerID' is now the index
    assert X.index.name == "customerID"

    # Check if TotalCharges was cleaned properly
    assert X.loc[2, "TotalCharges"] == 56.95  # was empty, replaced with MonthlyCharges
    assert X.loc[4, "TotalCharges"] == 42.30

    # Check if target values were mapped correctly
    assert y[1] == 0
    assert y[2] == 1

    # Check dimensions
    assert X.shape[0] == 4
    assert "Churn" not in X.columns


def test_split_train_test():
    # Create mock data
    X = pd.DataFrame({"A": range(10), "B": range(10, 20)})
    y = pd.Series([0, 1] * 5)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2
    # Check if the split preserves the index
    assert list(X_train.index.union(X_test.index)) == list(X.index)
