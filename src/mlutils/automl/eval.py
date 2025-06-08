import numpy as np
import pandas as pd
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, precision_score, r2_score, recall_score, roc_auc_score, root_mean_squared_error

from mlutils.utils.logger import CustomLogger  #

logger = CustomLogger("Eval").get_logger()


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ClassificationMetrics:
    """Class for computing common classification metrics."""

    y_true: pd.Series | np.ndarray
    y_pred: pd.Series | np.ndarray
    y_pred_proba: pd.Series | np.ndarray | None = None

    def calculate_accuracy(self) -> float:
        """Calculate accuracy score."""
        _accuracy = accuracy_score(self.y_true, self.y_pred)
        logger.info("Accuracy: {_accuracy}")
        return _accuracy

    def calculate_precision(self) -> float:
        """Calculate precision score."""
        _precision = precision_score(self.y_true, self.y_pred)
        logger.info("Precision: {_precision}")
        return _precision

    def calculate_recall(self) -> float:
        """Calculate recall score."""
        _recall = recall_score(self.y_true, self.y_pred)
        logger.info("Recall: {_recall}")
        return _recall

    def calculate_f1(self) -> float:
        """Calculate F1 score."""
        _f1_score = f1_score(self.y_true, self.y_pred)
        logger.info("F1 Score: {_f1_score}")
        return _f1_score

    def calculate_roc_auc(self) -> float:
        """Calculate ROC AUC score."""
        _roc_auc_score = np.round(roc_auc_score(self.y_true, self.y_pred), 2)
        logger.info("ROC AUC Score: {_roc_auc_score}")
        return _roc_auc_score

    def calculate_confusion_matrix(self) -> np.ndarray:
        """Calculate confusion matrix."""
        _confusion_matrix = confusion_matrix(self.y_true, self.y_pred)
        logger.info(f"Confusion Matrix: {_confusion_matrix}")
        return _confusion_matrix

    def get_all_metrics(self) -> dict[str, float]:
        """Calculate all classification metrics."""
        metrics = {
            "accuracy": self.calculate_accuracy(),
            "precision": self.calculate_precision(),
            "recall": self.calculate_recall(),
            "f1": self.calculate_f1(),
        }

        if self.y_pred_proba is not None:
            metrics["roc_auc"] = self.calculate_roc_auc()

        logger.info(f"All metrics: {metrics}")
        return metrics


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RegressionMetrics:
    """Class for computing common regression metrics."""

    y_true: pd.Series | np.ndarray
    y_pred: pd.Series | np.ndarray

    def calculate_mae(self) -> float:
        """Calculate Mean Absolute Error (MAE)."""
        _mae = mean_absolute_error(self.y_true, self.y_pred)
        logger.info(f"MAE: {_mae}")
        return _mae

    def calculate_mse(self) -> float:
        """Calculate Mean Squared Error (MSE)."""
        _mse = (root_mean_squared_error(self.y_true, self.y_pred)) ** 2
        logger.info(f"MSE: {_mse}")
        return _mse

    def calculate_rmse(self) -> float:
        """Calculate Root Mean Squared Error (RMSE)."""
        _rmse = root_mean_squared_error(self.y_true, self.y_pred)
        logger.info(f"RMSE: {_rmse}")
        return _rmse

    def calculate_r2(self) -> float:
        """Calculate R^2 (coefficient of determination)."""
        _r2 = r2_score(self.y_true, self.y_pred)
        logger.info(f"R² Score: {_r2}")
        return _r2

    def calculate_mape(self) -> float:
        """Calculate Mean Absolute Percentage Error (MAPE)."""
        _mape = np.round(mean_absolute_percentage_error(self.y_true, self.y_pred), 2)
        logger.info(f"MAPE: {_mape}")
        return _mape

    def calculate_msle(self) -> float:
        """Calculate Mean Squared Logarithmic Error (MSLE)."""
        _msle = mean_squared_log_error(self.y_true, self.y_pred)
        logger.info(f"MSLE: {_msle}")
        return _msle

    def get_all_metrics(self) -> dict[str, float]:
        """Calculate all regression metrics."""
        metrics = {
            "mae": self.calculate_mae(),
            "mse": self.calculate_mse(),
            "rmse": self.calculate_rmse(),
            "r2": self.calculate_r2(),
            "mape": self.calculate_mape(),
            "msle": self.calculate_msle(),
        }
        logger.info(f"All regression metrics: {metrics}")
        return metrics
