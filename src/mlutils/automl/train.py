import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from optuna.integration import OptunaSearchCV
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlutils.automl.eval import ClassificationMetrics, RegressionMetrics
from mlutils.utils.config import MLFLOW_TRACKING_URL  #
from mlutils.utils.io import split_train_test  #

mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)


def find_high_cardinality_features(df: pd.DataFrame, threshold: int = 10) -> list:
    """
    Identifies high cardinality features in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        threshold (int): The threshold for high cardinality features. Default is 10.

    Returns:
        list: A list of column names that are considered high cardinality.
    """
    return [col for col in df.columns if df[col].nunique() > threshold]


def build_preprocessor(num_cols, categorical_cols) -> ColumnTransformer:
    """
    Creates a ColumnTransformer to preprocess categorical features.

    Applies numerical transformations (imputation and scaling) and categorical transformations (imputation and one-hot encoding).

    Returns:
        ColumnTransformer: Transformer applying the specified encodings and returning a pandas DataFrame.
    """

    ## Setting Encoders
    numerical_transform = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),  # Impute missing values with the median
            ("scaler", StandardScaler()),  # Apply StandardScaler
        ]
    )

    categorical_transform = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),  # Impute missing values with constant 'Missing'
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),  # Apply StandardScaler
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transform, num_cols),
            ("cat", categorical_transform, categorical_cols),
        ],
        remainder="passthrough",
    ).set_output(transform="pandas")

    return preprocessor


def build_pipeline(preprocessor: ColumnTransformer, model: BaseEstimator, imbalanced: bool = False) -> Pipeline:
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


def split_features_by_type(df: pd.DataFrame) -> tuple:
    """
    Splits DataFrame columns into numerical and categorical based on their data types.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        tuple: Two list containing numerical and categorical column names.
    """
    numerical_types = ["int64", "float64"]
    categorical_types = ["object", "category"]

    num_cols = list(df.select_dtypes(include=numerical_types).columns)
    categorical_cols = list(df.select_dtypes(include=categorical_types).columns)

    return num_cols, categorical_cols


def cardinal_handling(X: pd.DataFrame, high_cardinality_features) -> pd.DataFrame:
    """
    Temporary function to handle high cardinality features using frequency encoding.
    This is a placeholder and should be replaced with a more robust solution to avoid data leakage.

    Args:
        X (pd.DataFrame): Input DataFrame with features.

    Returns:
        pd.DataFrame: DataFrame with high cardinality features encoded.
    """

    for col in high_cardinality_features:
        freq_encoding = X[col].value_counts(normalize=True).to_dict()
        X[col] = X[col].map(freq_encoding)

    return X


def model_tune(X: pd.DataFrame, y: pd.Series, model_name: str, model: BaseEstimator, param_grid: dict, search_algo: str, mlflow_expt_name: str, cross_val_scoring: list, evaluation_type: str = "classification", imbalanced: bool = False) -> None:
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
        evaluation_type (str): Type of evaluation ("classification" or "regression"). Default is "classification".

    Returns:
        None
    """
    high_cardinality_features = find_high_cardinality_features(X, threshold=5)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    X_train = cardinal_handling(X_train, high_cardinality_features)
    X_test = cardinal_handling(X_test, high_cardinality_features)

    num_cols, categorical_cols = split_features_by_type(X_train)

    mlflow.set_experiment(mlflow_expt_name)
    with mlflow.start_run(run_name=model_name):
        print(f"---{model_name}----")

        pipeline = build_pipeline(build_preprocessor(num_cols, categorical_cols), model, imbalanced)

        if search_algo == "optuna_search":
            searcher = OptunaSearchCV(pipeline, param_grid, cv=5, scoring="recall", n_trials=50, n_jobs=-1)
        elif search_algo == "grid_search":
            searcher = GridSearchCV(pipeline, param_grid, cv=5, scoring="recall", refit=True)
        elif search_algo == "random_search":
            searcher = RandomizedSearchCV(pipeline, param_grid, cv=5, scoring="recall", refit=True)

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
            scoring=cross_val_scoring,
        )
        scores = {"train_cross_val_" + metric: np.mean(value) for metric, value in scores.items()}
        mlflow.log_metrics(scores)

        ## Log Model
        mlflow.sklearn.log_model(searcher.best_estimator_, "model")

        if evaluation_type == "regression":
            ## Regression Metrics
            metrics_class = RegressionMetrics(y_test, y_predict)

        ## Classification Report
        elif evaluation_type == "classification":
            ## Classification Report as JSON
            report = classification_report(y_test, y_predict, output_dict=True)
            mlflow.log_dict(report, artifact_file="classification_report.json")

            metrics_class = ClassificationMetrics(y_test, y_predict)

        ## All metrics
        metrics = metrics_class.get_all_metrics()
        mlflow.log_metrics(metrics)

    mlflow.end_run()
