import numpy as np
import pytest

from mlutils.automl.eval import ClassificationMetrics, RegressionMetrics


@pytest.fixture
def classifier_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_pred_proba = np.array([0.2, 0.8, 0.4, 0.1, 0.7])

    classifier = ClassificationMetrics(y_true, y_pred, y_pred_proba)
    return classifier


@pytest.fixture
def regression_metrics():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    reg_metrics = RegressionMetrics(y_true, y_pred)
    return reg_metrics


class TestClassifierMetrics:
    def test_calculate_accuracy(self, classifier_metrics):
        assert classifier_metrics.calculate_accuracy() == pytest.approx(0.8)

    def test_calculate_precision(self, classifier_metrics):
        assert classifier_metrics.calculate_precision() == pytest.approx(1.0)

    def test_calculate_recall(self, classifier_metrics):
        assert classifier_metrics.calculate_recall() == pytest.approx(0.666666, rel=1e-2)

    def test_calculate_f1(self, classifier_metrics):
        assert classifier_metrics.calculate_f1() == pytest.approx(0.8)

    def test_calculate_roc_auc(self, classifier_metrics):
        assert classifier_metrics.calculate_roc_auc() == pytest.approx(0.83, rel=1e-3)

    def test_calculate_confusion_matrix(self, classifier_metrics):
        cm = classifier_metrics.calculate_confusion_matrix()
        expected_cm = np.array([[2, 0], [1, 2]])
        assert np.array_equal(cm, expected_cm)

    def test_get_all_metrics_classification(self, classifier_metrics):
        metrics = classifier_metrics.get_all_metrics()
        metrics = {key: np.round(value, 2) for key, value in metrics.items()}

        assert metrics["accuracy"] == pytest.approx(0.8)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(0.67)
        assert metrics["f1"] == pytest.approx(0.8)
        assert metrics["roc_auc"] == pytest.approx(0.83)


class TestRegressionMetrics:
    def test_calculate_mse(self, regression_metrics):
        assert regression_metrics.calculate_mse() == pytest.approx(0.375)

    def test_calculate_rmse(self, regression_metrics):
        assert regression_metrics.calculate_rmse() == pytest.approx(0.612372, rel=1e-6)

    def test_calculate_mae(self, regression_metrics):
        assert regression_metrics.calculate_mae() == pytest.approx(0.5)

    def test_calculate_r2(self, regression_metrics):
        assert regression_metrics.calculate_r2() == pytest.approx(0.948608, rel=1e-6)

    def test_calculate_mape(self, regression_metrics):
        assert regression_metrics.calculate_mape() == pytest.approx(0.33, rel=1e-6)

    def test_calculate_msle(self, regression_metrics):
        assert regression_metrics.calculate_msle() == pytest.approx(0.128, rel=1e-3)

    def test_get_all_metrics(self, regression_metrics):
        metrics = regression_metrics.get_all_metrics()
        metrics = {key: np.round(value, 6) for key, value in metrics.items()}

        assert metrics["mse"] == pytest.approx(0.375)
        assert metrics["rmse"] == pytest.approx(0.612372)
        assert metrics["mae"] == pytest.approx(0.5)
        assert metrics["r2"] == pytest.approx(0.948608)
        assert metrics["mape"] == pytest.approx(0.33)
        assert metrics["msle"] == pytest.approx(0.128, rel=1e-3)
