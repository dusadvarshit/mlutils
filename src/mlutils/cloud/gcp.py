import os

from google.cloud import storage

# Path to your service account key
SERVICE_ACCOUNT_JSON = os.environ["GCP_SERVICE_ACCOUNT_JSON"]

# GCS bucket and file info
BUCKET_NAME = "expt-mandrakebio"
DESTINATION_BLOB_NAME = "folder/your_file.txt"  # GCS path
LOCAL_FILE_PATH = "a.txt"  # Local file to upload


def upload_blob(bucket_name: str, destination_blob_name: str, local_file_path: str) -> None:
    # Authenticate and create a client
    client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_JSON)

    # Get the bucket
    bucket = client.bucket(BUCKET_NAME)

    # Get the blob (the file in GCS)
    blob = bucket.blob(DESTINATION_BLOB_NAME)

    # Upload the new file — this will overwrite the existing one
    blob.upload_from_filename(LOCAL_FILE_PATH)

    print(f"File {LOCAL_FILE_PATH} uploaded to {DESTINATION_BLOB_NAME}.")
