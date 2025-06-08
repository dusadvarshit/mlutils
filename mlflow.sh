set -a
source .env
mlflow server \
  --backend-store-uri $POSTGRES_DB \
  --default-artifact-root s3://expt/mlflow/ \
  --host 0.0.0.0 \
  --port 5000

set +a

## For local testing, you can use SQLite and a local directory for artifacts
## Uncomment the following lines and comment the first line to run MLflow UI locally

# mkdir -p ~/mlruns
# mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
