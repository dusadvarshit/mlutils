from unittest.mock import MagicMock, patch

from mlutils.utils.mlflow_utils import fetch_model


def test_fetch_model_success():
    with patch("mlutils.utils.mlflow_utils.mlflow") as mock_class:
        mock_class.tracking.MlflowClient.return_value = MagicMock()
        mock_class.set_tracking_uri.return_value = None
        mock_class.sklearn.load_model.return_value = MagicMock()

        client = mock_class.tracking.MlflowClient()
        client.get_latest_versions.return_value = [MagicMock()]
        client.get_latest_versions.return_value[0].version.return_value = 1

        model = fetch_model("test-model")

        assert isinstance(model, MagicMock)


def test_fetch_model_failure():
    with patch("mlutils.utils.mlflow_utils.mlflow") as mock_class:
        mock_class.tracking.MlflowClient.return_value = MagicMock()
        mock_class.set_tracking_uri.return_value = None

        mock_client_instance = mock_class.tracking.MlflowClient()
        mock_client_instance.get_latest_versions.side_effect = Exception("MLflow error")

        model = fetch_model("nonexistent-model")

        assert model is None
