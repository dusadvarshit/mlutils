import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlutils.automl.train import (
    build_pipeline,
    build_preprocessor,
    cardinal_handling,
    find_high_cardinality_features,
    split_features_by_type,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({"num1": [1, 2, 3, np.nan], "num2": [10, 20, 30, 40], "cat1": ["a", "b", "a", np.nan], "cat2": ["x", "y", "z", "x"], "high_card": [f"user_{i}" for i in [1, 2, 3, 4]]})


def test_find_high_cardinality_features(sample_df):
    assert find_high_cardinality_features(sample_df, threshold=2) == ["cat2", "high_card"]
    assert find_high_cardinality_features(sample_df, threshold=4) == []
    assert find_high_cardinality_features(sample_df, threshold=1) == ["cat1", "cat2", "high_card"]


def test_split_features_by_type(sample_df):
    num_cols, cat_cols = split_features_by_type(sample_df)
    assert set(num_cols) == {"num1", "num2"}
    assert set(cat_cols) == {"cat1", "cat2", "high_card"}


def test_cardinal_handling(sample_df):
    handled_df = cardinal_handling(sample_df.copy(), ["high_card"])
    assert np.isclose(handled_df["high_card"].sum(), 1.0)
    assert handled_df["high_card"].dtype == float


def test_build_preprocessor(sample_df):
    num_cols, cat_cols = split_features_by_type(sample_df)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    transformed = preprocessor.fit_transform(sample_df)
    # Check shape (numerical scaled + one-hot encoded + passthrough)
    expected_cols = (
        len(num_cols)  # scaled
        + sum(sample_df[cat].nunique(dropna=False) for cat in cat_cols)  # one-hot encoded
    )
    assert transformed.shape[0] == sample_df.shape[0]
    assert transformed.shape[1] >= expected_cols  # allow passthrough


def test_build_pipeline_balanced(sample_df):
    num_cols, cat_cols = split_features_by_type(sample_df)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    model = LogisticRegression()
    pipe = build_pipeline(preprocessor, model, imbalanced=False)
    assert isinstance(pipe, Pipeline)
    pipe.fit(sample_df, [0, 1, 0, 1])  # dummy target
    preds = pipe.predict(sample_df)
    assert len(preds) == len(sample_df)


def test_build_pipeline_imbalanced(sample_df):
    num_cols, cat_cols = split_features_by_type(sample_df)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    model = LogisticRegression()
    pipe = build_pipeline(preprocessor, model, imbalanced=True)
    assert isinstance(pipe, imbpipeline)
    pipe.fit(sample_df, [0, 1, 0, 1])
    preds = pipe.predict(sample_df)
    assert len(preds) == len(sample_df)
