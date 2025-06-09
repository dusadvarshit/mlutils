from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from mlutils.utils.io import MyDataset

regressors_param_grid = {
    "LinearRegression": {
        "model": LinearRegression(),
        "param_grid": {},  # No hyperparameters to tune for basic LinearRegression
    },
    "Ridge": {"model": Ridge(), "param_grid": {"model__alpha": ("float", [0.01, 0.1, 1.0, 10.0]), "model__solver": ("categorical", ["auto", "svd", "cholesky", "lsqr"])}},
    "Lasso": {"model": Lasso(), "param_grid": {"model__alpha": ("float", [0.01, 0.1, 1.0, 10.0]), "model__max_iter": ("int", [1000, 5000])}},
    "GradientBoostingRegressor": {"model": GradientBoostingRegressor(), "param_grid": {"model__n_estimators": ("int", [100, 200]), "model__learning_rate": ("float", [0.01, 0.1, 0.2]), "model__max_depth": ("int", [3, 5])}},
    "RandomForestRegressor": {"model": RandomForestRegressor(), "param_grid": {"model__n_estimators": ("int", [100, 200]), "model__max_depth": ("int", [5, 10, None]), "model__min_samples_split": ("int", [2, 5]), "model__min_samples_leaf": ("int", [1, 2])}},
    "AdaBoostRegressor": {"model": AdaBoostRegressor(), "param_grid": {"model__n_estimators": ("int", [50, 100]), "model__learning_rate": ("float", [0.01, 0.1, 1.0])}},
    "BaggingRegressor": {"model": BaggingRegressor(), "param_grid": {"model__n_estimators": ("int", [10, 50, 100]), "model__max_samples": ("float", [0.5, 1.0]), "model__max_features": ("float", [0.5, 1.0])}},
    "DecisionTreeRegressor": {"model": DecisionTreeRegressor(), "param_grid": {"model__max_depth": ("int", [3, 5, 10]), "model__min_samples_split": ("int", [2, 5]), "model__min_samples_leaf": ("int", [1, 2])}},
    "KNeighborsRegressor": {"model": KNeighborsRegressor(), "param_grid": {"model__n_neighbors": ("int", [3, 5, 10]), "model__weights": ("categorical", ["uniform", "distance"]), "model__metric": ("categorical", ["euclidean", "manhattan"])}},
    "XGBRegressor": {"model": XGBRegressor(), "param_grid": {"model__n_estimators": ("int", [100, 200]), "model__learning_rate": ("float", [0.01, 0.1, 0.2]), "model__max_depth": ("int", [3, 5, 7])}},
    "LGBMRegressor": {"model": LGBMRegressor(), "param_grid": {"model__n_estimators": ("int", [100, 200]), "model__learning_rate": ("float", [0.01, 0.1]), "model__max_depth": ("int", [3, 5, 7])}},
}

dataset = MyDataset(
    name="regression_dataset",
    path="C:\\Users\\dusad\\Documents\\Projects\\agnei_consulting\\mlutils\\data\datasets\\denkuznetz\\housing-prices-regression\\versions\\1\\real_estate_dataset.csv",
    target_col="Price",
    type="regression",
    optimization_type="grid_search",
    param_grid_map=regressors_param_grid,
    cross_val_scoring=["neg_mean_squared_error", "r2", "neg_mean_absolute_error", "explained_variance"],
    index_col="ID",
)
