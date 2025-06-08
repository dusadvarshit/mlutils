import os
from contextlib import suppress

from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient

## Accessing .env file
from dotenv import load_dotenv

load_dotenv()
####

# Azure credentials and storage details
tenant_id = os.environ["AZURE_TENANT_ID"]
client_id = os.environ["AZURE_CLIENT_ID"]
client_secret = os.environ["AZURE_CLIENT_SECRET"]


# Authenticate using service principal
credential = ClientSecretCredential(tenant_id, client_id, client_secret)

# Azure Storage account details
storage_account_name = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
container_name = os.environ["AZURE_CONTAINER_NAME"]

# Create the BlobServiceClient using the storage account URL
blob_service_client = BlobServiceClient(account_url=f"https://{storage_account_name}.blob.core.windows.net/", credential=credential)

# Get container and blob client
container_client = blob_service_client.get_container_client(container_name)


def upload_file_to_blob(blob_name: str, local_file_path: str) -> None:
    # (Optional) Create container if not exists

    ## Checks if the container exists, if not, it creates it
    with suppress(Exception):
        container_client.create_container()

    # Upload file
    with open(local_file_path, "rb") as data:
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(data, overwrite=True)

    print(f"File '{local_file_path}' uploaded to blob '{blob_name}' in container '{container_name}'.")
