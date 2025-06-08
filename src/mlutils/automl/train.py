import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from optuna.integration import OptunaSearchCV
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from mlutils.automl.eval import get_all_metrics  #
from mlutils.utils.config import (
    MLFLOW_TRACKING_URL,
    binary_cols,
    categorical_cols,
    categorical_cols_mapping,
    ohe_cols,
)  #
from mlutils.utils.io import split_train_test  #

mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)


def build_preprocessor() -> ColumnTransformer:
    """
    Creates a ColumnTransformer to preprocess categorical features.

    Applies one-hot encoding to nominal columns, ordinal encoding to binary columns,
    and ordinal encoding with category mappings to ordinal columns.

    Returns:
        ColumnTransformer: Transformer applying the specified encodings and returning a pandas DataFrame.

    Note:
        Assumes `ohe_cols`, `binary_cols`, `categorical_cols`, and `categorical_cols_mapping` are defined globally.
    """

    ## Setting Encoders
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore").set_output(transform="pandas")
    ordinal_binary = OrdinalEncoder().set_output(transform="pandas")
    ordinal_cat = OrdinalEncoder(categories=categorical_cols_mapping).set_output(transform="pandas")

    preprocessor = ColumnTransformer(
        transformers=[
            ("ohe", ohe, ohe_cols),
            ("binary", ordinal_binary, binary_cols),
            ("cat", ordinal_cat, categorical_cols),
        ]
    ).set_output(transform="pandas")

    return preprocessor


def build_pipeline(preprocessor: ColumnTransformer, model: BaseEstimator, imbalanced: bool = True) -> Pipeline:
    """
    Build a machine learning pipeline with optional SMOTE oversampling.

    Args:
        preprocessor: Transformer for preprocessing input data.
        model: Estimator for classification or regression.
        imbalanced (bool): If True, include SMOTE for imbalanced data handling.

    Returns:
        Pipeline: Configured sklearn Pipeline object.
    """

    if imbalanced:
        smote = SMOTE(random_state=42)
        clf = imbpipeline(steps=[("preprocess", preprocessor), ("smote", smote), ("model", model)])
    else:
        clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    return clf


def model_tune(X: pd.DataFrame, y: pd.Series, model_name: str, model: BaseEstimator, param_grid: dict, search_algo: str, mlflow_expt_name: str):
    """
    Tunes a machine learning model using the specified search algorithm and logs results to MLflow.

    Args:
        X (pd.DataFrame or np.ndarray): Feature data.
        y (pd.Series or np.ndarray): Target labels.
        model_name (str): Name of the model run.
        model: Scikit-learn compatible model instance.
        param_grid (dict): Hyperparameter search space.
        search_algo (str): Search strategy ("grid" or "bayesian").
        mlflow_expt_name (str): Name of the MLflow experiment.

    Returns:
        None
    """

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    mlflow.set_experiment(mlflow_expt_name)
    with mlflow.start_run(run_name=model_name):
        print(f"---{model_name}----")

        pipeline = build_pipeline(build_preprocessor(), model)

        if search_algo == "bayesian":
            searcher = OptunaSearchCV(pipeline, param_grid, cv=5, scoring="recall", n_trials=50, n_jobs=-1)
        elif search_algo == "grid":
            searcher = GridSearchCV(pipeline, param_grid, cv=5, scoring="recall", refit=True)

        mlflow.set_tag("developer", "Varshit Dusad")
        mlflow.set_tag("model_name", model_name)

        searcher.fit(X_train, y_train)
        mlflow.log_params(searcher.best_params_)

        y_predict = searcher.predict(X_test)
        best_estimator = searcher.best_estimator_

        scores = cross_validate(
            best_estimator,
            X_train,
            y_train,
            cv=5,
            scoring=[
                "accuracy",
                "precision",
                "recall",
                "f1",
                "jaccard",
                "mutual_info_score",
            ],
        )
        scores = {"train_cross_val_" + metric: np.mean(value) for metric, value in scores.items()}
        mlflow.log_metrics(scores)

        ## Log Model
        mlflow.sklearn.log_model(searcher.best_estimator_, "model")

        ## Classification Report
        report = classification_report(y_test, y_predict, output_dict=True)

        mlflow.log_dict(report, artifact_file="classification_report.json")

        ## All metrics
        metrics = get_all_metrics(y_test, y_predict)
        mlflow.log_metrics(metrics)

    mlflow.end_run()
