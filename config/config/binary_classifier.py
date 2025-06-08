from sklearn.ensemble import (
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression

models_param_grid = {
    "LogisticRegression": {
        "model": LogisticRegression(),
        "param_grid": {
            "model__C": ("float", [0.01, 0.1, 1, 10]),
            "model__penalty": ("categorical", ["l2"]),
            "model__solver": ("categorical", ["lbfgs", "liblinear"]),
            "model__max_iter": ("int", [10, 100, 1000]),
        },
    },
    "GradientBoostingClassifier": {
        "model": GradientBoostingClassifier(),
        "param_grid": {
            "model__n_estimators": ("int", [100, 200]),
            "model__learning_rate": ("float", [0.01, 0.1, 0.2]),
            "model__max_depth": ("int", [3, 5]),
        },
    },
    # "AdaBoostClassifier": {
    #     "model": AdaBoostClassifier(),
    #     "param_grid": {
    #         "model__n_estimators": ('int', [50, 100]),
    #         "model__learning_rate": ('float', [0.01, 0.1, 1]),
    #     },
    # },
    # "BaggingClassifier": {
    #     "model": BaggingClassifier(),
    #     "param_grid": {
    #         "model__n_estimators": ('int', [10, 50, 100]),
    #         "model__max_samples": ('float', [0.5, 1.0]),
    #         "model__max_features": ('float', [0.5, 1.0]),
    #     },
    # },
    # "DecisionTreeClassifier": {
    #     "model": DecisionTreeClassifier(),
    #     "param_grid": {
    #         "model__max_depth": ('int', [1, 5, 10]),
    #         "model__min_samples_split": ('int', [2, 5]),
    #         "model__min_samples_leaf": ('int', [1, 2]),
    #     },
    # },
    # "KNeighborsClassifier": {
    #     "model": KNeighborsClassifier(),
    #     "param_grid": {
    #         "model__n_neighbors": ('int', [3, 5, 10]),
    #         "model__weights": ('categorical', ["uniform", "distance"]),
    #         "model__metric": ('categorical', ["euclidean", "manhattan"]),
    #     },
    # },
    # "GaussianNB": {
    #     "model": GaussianNB(),
    #     "param_grid": {
    #     },
    # },
    # "XGBClassifier": {
    #     "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    #     "param_grid": {
    #         "model__n_estimators": ('int', [100, 200]),
    #         "model__learning_rate": ('float', [0.01, 0.1, 0.2]),
    #         "model__max_depth": ('int', [3, 5, 7]),
    #     },
    # },
    # "LGBMClassifier": {
    #     "model": LGBMClassifier(),
    #     "param_grid": {
    #         "model__n_estimators": ('int', [100, 200]),
    #         "model__learning_rate": ('float', [0.01, 0.1]),
    #         "model__max_depth": ('int', [3, 5, 7]),
    #     },
    # },
    # "RandomForestClassifier": {
    #     "model": RandomForestClassifier(),
    #     "param_grid": {
    #         "model__n_estimators": ('int', [100, 200]),
    #         "model__max_depth": ('int', [1, 5, 10]),
    #         "model__min_samples_split": ('int', [2, 5]),
    #         "model__min_samples_leaf": ('int', [1, 2]),
    #     },
    # },
}
