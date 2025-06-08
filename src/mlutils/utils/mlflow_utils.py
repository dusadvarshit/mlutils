import mlflow
import mlflow.tracking

from mlutils.utils.config import MLFLOW_TRACKING_URL
from mlutils.utils.logger import CustomLogger

mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)

logger = CustomLogger("mlflow-utils").get_logger()


def fetch_model(model_name: str):
    """
    Fetch the model in staging from MLflow model registry.

    Args:
        model_name (str): Name of the registered model

    Returns:
        The loaded model from staging
    """
    client = mlflow.tracking.MlflowClient()

    # Get the latestmodel version
    try:
        # Fetch all versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")

        # Find the latest version by version number
        latest_version = max(versions, key=lambda v: int(v.version))

        # Get the run ID or artifact URI if needed
        logger.info(f"Latest version: {latest_version.version}")
        logger.info(f"Status: {latest_version.current_stage}")
        logger.info(f"Artifact URI: {latest_version.source}")

        # Load the model
        model = mlflow.sklearn.load_model(latest_version.source)
        return model

    except Exception as e:
        logger.warning(f"Error fetching {model_name} model: {str(e)}")
        return None
