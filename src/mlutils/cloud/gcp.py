import os
from datetime import timedelta as td

## Accessing .env file
from dotenv import load_dotenv
from google.api_core.exceptions import GoogleAPIError
from google.cloud import storage
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

load_dotenv()
####

# Path to your service account key
SERVICE_ACCOUNT_JSON = os.environ["GCP_SERVICE_ACCOUNT_JSON"]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class GCPStorage:
    client: storage.Client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_JSON)
    BUCKET_NAME: str = "expt-mandrakebio"

    def upload_file(self, upload_file: str, filename: str) -> None:
        """Uploads a file to the configured GCS bucket."""
        try:
            print("Uploading to GCP Storage!!!")
            bucket = self.client.bucket(self.BUCKET_NAME)
            blob = bucket.blob(filename)
            blob.upload_from_filename(upload_file)
            print(f"File '{upload_file}' uploaded as '{filename}'.")
        except GoogleAPIError as e:
            print(f"Error uploading file: {e}")

    def generate_presigned_url_for_object(self, gcs_filename: str, expiration: td | None = td(hours=1)) -> str | None:
        """Generates a signed URL for accessing a GCS object."""
        try:
            bucket = self.client.bucket(self.BUCKET_NAME)
            blob = bucket.blob(gcs_filename)
            url = blob.generate_signed_url(expiration=expiration)
            return url
        except Exception as e:
            print(f"Error generating signed URL: {e}")
            return None

    def create_bucket(self, bucket_name: str = None, location: str = "us-east1") -> None:
        """Creates a GCS bucket in the specified region."""
        try:
            name = bucket_name if bucket_name else self.BUCKET_NAME
            bucket = self.client.bucket(name)
            new_bucket = self.client.create_bucket(bucket, location=location)
            print(f"Bucket '{new_bucket.name}' created successfully in {location}.")
        except GoogleAPIError as e:
            print(f"Error creating bucket: {e}")

    def list_buckets(self) -> list:
        """Lists all GCS buckets under the current project."""
        try:
            buckets = list(self.client.list_buckets())
            names = [bucket.name for bucket in buckets]
            print("Buckets:", names)
            return names
        except GoogleAPIError as e:
            print(f"Error listing buckets: {e}")
            return []

    def delete_bucket(self, bucket_name: str) -> None:
        """Deletes the specified GCS bucket."""
        try:
            bucket = self.client.bucket(bucket_name)
            bucket.delete(force=True)
            print(f"Bucket '{bucket_name}' deleted successfully.")
        except GoogleAPIError as e:
            print(f"Error deleting bucket: {e}")

    def list_objects(self, bucket_name: str = None) -> list:
        """Lists objects stored in a specified GCS bucket."""
        try:
            name = bucket_name if bucket_name else self.BUCKET_NAME
            bucket = self.client.bucket(name)
            blobs = list(bucket.list_blobs())
            keys = [blob.name for blob in blobs]
            print(f"Objects in bucket '{name}':", keys)
            return keys
        except GoogleAPIError as e:
            print(f"Error listing objects: {e}")
            return []
