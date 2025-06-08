import numpy as np
import pytest

from mlutils.automl.eval import calculate_accuracy, calculate_confusion_matrix, calculate_f1, calculate_precision, calculate_recall, calculate_roc_auc, get_all_metrics


@pytest.fixture
def sample_data():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_pred_proba = np.array([0.2, 0.8, 0.4, 0.1, 0.7])
    return y_true, y_pred, y_pred_proba


def test_calculate_accuracy(sample_data):
    y_true, y_pred, _ = sample_data
    assert calculate_accuracy(y_true, y_pred) == pytest.approx(0.8)


def test_calculate_precision(sample_data):
    y_true, y_pred, _ = sample_data
    assert calculate_precision(y_true, y_pred) == pytest.approx(1.0)


def test_calculate_recall(sample_data):
    y_true, y_pred, _ = sample_data
    assert calculate_recall(y_true, y_pred) == pytest.approx(0.666666, rel=1e-2)


def test_calculate_f1(sample_data):
    y_true, y_pred, _ = sample_data
    assert calculate_f1(y_true, y_pred) == pytest.approx(0.8)


def test_calculate_roc_auc(sample_data):
    y_true, _, y_pred_proba = sample_data
    assert calculate_roc_auc(y_true, y_pred_proba) == pytest.approx(1.0)


def test_calculate_confusion_matrix(sample_data):
    y_true, y_pred, _ = sample_data
    cm = calculate_confusion_matrix(y_true, y_pred)
    expected_cm = np.array([[2, 0], [1, 2]])
    assert np.array_equal(cm, expected_cm)


def test_get_all_metrics(sample_data):
    y_true, y_pred, y_pred_proba = sample_data
    metrics = get_all_metrics(y_true, y_pred, y_pred_proba)
    metrics = {key: np.round(value, 2) for key, value in metrics.items()}

    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.67)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["roc_auc"] == pytest.approx(1.0)
