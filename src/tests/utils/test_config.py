import optuna
import pytest

from mlutils.utils.config import clean_model_params, param_grid_fix  # Replace 'your_module' with actual module name

# Test clean_model_params


@pytest.mark.parametrize(
    "optimization_type, tup, expected",
    [
        ("grid_search", (None, [1, 2, 3]), [1, 2, 3]),
        ("random_search", (None, ["a", "b"]), ["a", "b"]),
    ],
)
def test_clean_model_params_basic(optimization_type, tup, expected):
    assert clean_model_params(optimization_type, tup) == expected


def test_clean_model_params_optuna_int():
    result = clean_model_params("optuna_search", ("int", (1, 10)))
    assert isinstance(result, optuna.distributions.IntDistribution)
    assert result.low == 1
    assert result.high == 10


def test_clean_model_params_optuna_float():
    result = clean_model_params("optuna_search", ("float", (0.001, 0.1)))
    assert isinstance(result, optuna.distributions.FloatDistribution)
    assert result.low == 0.001
    assert result.high == 0.1


def test_clean_model_params_optuna_categorical():
    result = clean_model_params("optuna_search", ("categorical", ["linear", "rbf"]))
    assert isinstance(result, optuna.distributions.CategoricalDistribution)
    assert result.choices == ("linear", "rbf")


# Test param_grid_fix


def test_param_grid_fix_grid_search():
    models_param_grid = {
        "svc": {
            "model": "SVC",
            "param_grid": {
                "C": (None, [1, 10, 100]),
                "kernel": (None, ["linear", "rbf"]),
            },
        }
    }

    fixed = param_grid_fix(models_param_grid, "grid_search")

    assert fixed["svc"]["param_grid"]["C"] == [1, 10, 100]
    assert fixed["svc"]["param_grid"]["kernel"] == ["linear", "rbf"]
    assert fixed["svc"]["model"] == "SVC"


def test_param_grid_fix_optuna():
    models_param_grid = {
        "svc": {
            "model": "SVC",
            "param_grid": {
                "C": ("float", (0.01, 10)),
                "kernel": ("categorical", ["linear", "rbf"]),
                "degree": ("int", (2, 5)),
            },
        }
    }

    fixed = param_grid_fix(models_param_grid, "optuna_search")

    assert isinstance(fixed["svc"]["param_grid"]["C"], optuna.distributions.FloatDistribution)
    assert fixed["svc"]["param_grid"]["C"].low == 0.01
    assert fixed["svc"]["param_grid"]["C"].high == 10

    assert isinstance(fixed["svc"]["param_grid"]["degree"], optuna.distributions.IntDistribution)
    assert fixed["svc"]["param_grid"]["degree"].low == 2
    assert fixed["svc"]["param_grid"]["degree"].high == 5

    assert isinstance(fixed["svc"]["param_grid"]["kernel"], optuna.distributions.CategoricalDistribution)
    assert fixed["svc"]["param_grid"]["kernel"].choices == ("linear", "rbf")
