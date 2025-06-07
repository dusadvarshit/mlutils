import os

import optuna
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

load_dotenv()

if os.environ["ENV"] == "DEV":
    MLFLOW_TRACKING_URL = "http://localhost:8000" if os.name == "nt" else "http://mlflow:8000"
elif os.environ["ENV"] == "PROD":
    MLFLOW_TRACKING_URL = "http://mlflow.TestCluster.local:8000"

MLFLOW_ROOT_FOLDER = "../../../mlflow"

MLFLOW_ARTIFACTS = "../../../mlflow/artifacts"

binary_classification_models_param_grid = {
    "LogisticRegression": {
        "model": LogisticRegression(),
        "param_grid": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
            "model__max_iter": [10, 100, 1000],
        },
    },
    "GradientBoostingClassifier": {
        "model": GradientBoostingClassifier(),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5],
        },
    },
    "AdaBoostClassifier": {
        "model": AdaBoostClassifier(),
        "param_grid": {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.01, 0.1, 1],
        },
    },
    "BaggingClassifier": {
        "model": BaggingClassifier(),
        "param_grid": {
            "model__n_estimators": [10, 50, 100],
            "model__max_samples": [0.5, 1.0],
            "model__max_features": [0.5, 1.0],
        },
    },
    "SVC": {
        "model": SVC(),
        "param_grid": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf", "poly"],
            "model__gamma": ["scale", "auto"],
        },
    },
    "DecisionTreeClassifier": {
        "model": DecisionTreeClassifier(),
        "param_grid": {
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
    },
    "KNeighborsClassifier": {
        "model": KNeighborsClassifier(),
        "param_grid": {
            "model__n_neighbors": [3, 5, 10],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"],
        },
    },
    "GaussianNB": {
        "model": GaussianNB(),
        "param_grid": {
            # No major hyperparameters to tune, included for completeness
        },
    },
    "XGBClassifier": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5, 7],
        },
    },
    "LGBMClassifier": {
        "model": LGBMClassifier(),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 5, 7],
        },
    },
    "RandomForestClassifier": {
        "model": RandomForestClassifier(),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
    },
}

binary_classification_models_param_grid_optuna = {
    "LogisticRegression": {
        "model": LogisticRegression(),
        "param_grid": {
            "model__C": optuna.distributions.FloatDistribution(0.01, 10.0, log=True),
            "model__penalty": optuna.distributions.CategoricalDistribution(["l2"]),
            "model__solver": optuna.distributions.CategoricalDistribution(["lbfgs", "liblinear"]),
            "model__max_iter": optuna.distributions.IntDistribution(10, 1000, log=True),
        },
    },
    "GradientBoostingClassifier": {
        "model": GradientBoostingClassifier(),
        "param_grid": {"model__n_estimators": optuna.distributions.IntDistribution(100, 200), "model__learning_rate": optuna.distributions.FloatDistribution(0.01, 0.2, log=True), "model__max_depth": optuna.distributions.IntDistribution(3, 5)},
    },
    "AdaBoostClassifier": {"model": AdaBoostClassifier(), "param_grid": {"model__n_estimators": optuna.distributions.IntDistribution(50, 100), "model__learning_rate": optuna.distributions.FloatDistribution(0.01, 1.0, log=True)}},
    "BaggingClassifier": {"model": BaggingClassifier(), "param_grid": {"model__n_estimators": optuna.distributions.IntDistribution(10, 100), "model__max_samples": optuna.distributions.FloatDistribution(0.5, 1.0), "model__max_features": optuna.distributions.FloatDistribution(0.5, 1.0)}},
    "SVC": {
        "model": SVC(probability=True),  # Enable probability for ROC AUC
        "param_grid": {"model__C": optuna.distributions.FloatDistribution(0.1, 10.0, log=True), "model__kernel": optuna.distributions.CategoricalDistribution(["linear", "rbf", "poly"]), "model__gamma": optuna.distributions.CategoricalDistribution(["scale", "auto"])},
    },
    "DecisionTreeClassifier": {
        "model": DecisionTreeClassifier(),
        "param_grid": {"model__max_depth": optuna.distributions.CategoricalDistribution([None, 5, 10]), "model__min_samples_split": optuna.distributions.IntDistribution(2, 5), "model__min_samples_leaf": optuna.distributions.IntDistribution(1, 2)},
    },
    "KNeighborsClassifier": {
        "model": KNeighborsClassifier(),
        "param_grid": {"model__n_neighbors": optuna.distributions.IntDistribution(3, 10), "model__weights": optuna.distributions.CategoricalDistribution(["uniform", "distance"]), "model__metric": optuna.distributions.CategoricalDistribution(["euclidean", "manhattan"])},
    },
    "GaussianNB": {
        "model": GaussianNB(),
        "param_grid": {},  # No hyperparameters to tune
    },
    "XGBClassifier": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),  # print(classification_report(y_test, y_predict))
        "param_grid": {
            "model__learning_rate": optuna.distributions.FloatDistribution(0.01, 0.1, log=True),
            "model__n_estimators": optuna.distributions.IntDistribution(100, 500),
            "model__max_depth": optuna.distributions.IntDistribution(3, 7),
            "model__min_child_weight": optuna.distributions.IntDistribution(1, 5),
            "model__subsample": optuna.distributions.FloatDistribution(0.6, 1.0),
            "model__colsample_bytree": optuna.distributions.FloatDistribution(0.6, 1.0),
            "model__gamma": optuna.distributions.FloatDistribution(0, 0.3),
            "model__reg_alpha": optuna.distributions.FloatDistribution(0, 0.1),
            "model__reg_lambda": optuna.distributions.FloatDistribution(1, 2),
        },
    },
    "LGBMClassifier": {"model": LGBMClassifier(verbose=-1), "param_grid": {"model__n_estimators": optuna.distributions.IntDistribution(100, 200), "model__learning_rate": optuna.distributions.FloatDistribution(0.01, 0.1, log=True), "model__max_depth": optuna.distributions.IntDistribution(3, 7)}},
    "RandomForestClassifier": {
        "model": RandomForestClassifier(),
        "param_grid": {
            "model__n_estimators": optuna.distributions.IntDistribution(100, 200),
            "model__max_depth": optuna.distributions.CategoricalDistribution([None, 5, 10]),
            "model__min_samples_split": optuna.distributions.IntDistribution(2, 5),
            "model__min_samples_leaf": optuna.distributions.IntDistribution(1, 2),
        },
    },
}

ohe_cols = ["PaymentMethod"]
binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
categorical_cols = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
]
categorical_cols_mapping = [
    ["No phone service", "No", "Yes"],
    ["No", "DSL", "Fiber optic"],
    *([["No internet service", "No", "Yes"]] * 6),
    ["Month-to-month", "One year", "Two year"],
]
