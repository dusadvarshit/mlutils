from io import StringIO
from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture
def sample_csv_classify():
    data = StringIO(
        """customerID,TotalCharges,MonthlyCharges,Churn
1,29.85,29.85,No
2, ,56.95,Yes
3,1889.5,53.85,No
4, ,42.30,Yes
"""
    )
    return pd.read_csv(data)


@pytest.fixture
def sample_dataset_classify(sample_csv_classify):
    dataset = MagicMock()
    dataset.target_col = "Churn"
    dataset.path = "/tmp"
    dataset.label_encoding = {"Yes": 1, "No": 0}
    dataset.index_col = "customerID"

    return dataset


@pytest.fixture
def sample_csv_regression():
    data = StringIO(
        """ID,TotalRooms,KitchenSize,Useless,Price
1,3,29.85,,1100
2,4,56.95,,1200
3,5,53.85,,900
4, ,42.30,1,4500
"""
    )
    return pd.read_csv(data)


@pytest.fixture
def sample_dataset_regression(sample_csv_regression):
    dataset = MagicMock()
    dataset.target_col = "Price"
    dataset.path = "/tmp"
    dataset.label_encoding = None
    dataset.index_col = "ID"

    return dataset
