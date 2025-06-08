import copy
import os

import optuna
from dotenv import load_dotenv

load_dotenv()


MLFLOW_TRACKING_URL = "http://localhost:5000" if os.name == "nt" else "http://mlflow:5000"


def clean_model_params(optimization_type: str, tup: tuple):
    if optimization_type in ["grid_search", "random_search"]:
        # For grid search or random search, return the parameter directly
        return tup[1]

    elif optimization_type == "optuna_search":
        # For optuna, convert the model param grid to an Optuna distribution
        d_type, values = tup

        if d_type == "int":
            return optuna.distributions.IntDistribution(tup[1][0], tup[1][-1])
        elif d_type == "float":
            return optuna.distributions.FloatDistribution(tup[1][0], tup[1][-1], log=True)
        elif d_type == "categorical":
            return optuna.distributions.CategoricalDistribution(tup[1])


def param_grid_fix(models_param_grid, optimization_type: str):
    new_models_param_grid = copy.deepcopy(models_param_grid)
    for key, value in new_models_param_grid.items():
        new_models_param_grid[key] = {"param_grid": value["param_grid"], "model": value["model"]}

        for hyperparam, hyperparam_value in value["param_grid"].items():
            new_models_param_grid[key]["param_grid"][hyperparam] = clean_model_params(optimization_type, hyperparam_value)

    return new_models_param_grid


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
