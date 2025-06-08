import boto3
from botocore.client import BaseClient

## Accessing .env file
from dotenv import load_dotenv
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

load_dotenv()
####


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class AWSStorage:
    s3: BaseClient = boto3.client("s3", region_name="us-east-2")
    BUCKET_NAME: str = "expt"

    def upload_file(self, upload_file: str, filename: str) -> None:
        """Uploads a file to the configured S3 bucket.

        Args:
            upload_file (str): Local file path to be uploaded.
            filename (str): Filename to be used as the key in S3.

        Returns:
            None
        """

        print("Uploading to AWS!!!")
        s3_key = f"{filename}"

        with open(upload_file, "rb") as fileobj:
            # Upload the file to S3
            self.s3.upload_fileobj(fileobj, self.BUCKET_NAME, s3_key)

        return None

    def generate_presigned_url_for_object(self, s3_filename: str, expiration: int = 3600 * 24 * 7) -> str | None:
        """Generates a presigned URL for accessing an S3 object.

        Args:
            s3_filename (str): The key of the S3 object.
            expiration (int, optional): Expiration time in seconds. Defaults to 7 days.

        Returns:
            str or None: A presigned URL string if successful, otherwise None.
        """

        try:
            s3_key = f"{s3_filename}"
            response = self.s3.generate_presigned_url("get_object", Params={"Bucket": self.BUCKET_NAME, "Key": s3_key}, ExpiresIn=expiration)

        except Exception as e:
            print(f"Error generating presigned URL: {e}")
            return None

        return response

    def create_bucket(self, bucket_name: str = None) -> None:
        """Creates an S3 bucket in the specified region.

        Args:
            bucket_name (str, optional): The name of the bucket to create.
                Defaults to the class's `BUCKET_NAME`.

        Returns:
            None
        """
        try:
            name = bucket_name if bucket_name else self.BUCKET_NAME
            self.s3.create_bucket(Bucket=name, CreateBucketConfiguration={"LocationConstraint": "us-east-2"})
            print(f"Bucket '{name}' created successfully.")
        except Exception as e:
            print(f"Error creating bucket: {e}")

    def list_buckets(self) -> list:
        """Lists all S3 buckets under the current AWS account.

        Returns:
            list: A list of bucket names.
        """
        try:
            response = self.s3.list_buckets()
            buckets = [bucket["Name"] for bucket in response.get("Buckets", [])]
            print("Buckets:", buckets)
            return buckets
        except Exception as e:
            print(f"Error listing buckets: {e}")
            return []

    def delete_bucket(self, bucket_name: str) -> None:
        """Deletes the specified S3 bucket.

        Args:
            bucket_name (str): The name of the bucket to delete.

        Returns:
            None
        """
        try:
            self.s3.delete_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' deleted successfully.")
        except Exception as e:
            print(f"Error deleting bucket: {e}")

    def list_objects(self, bucket_name: str = None) -> list:
        """Lists objects stored in a specified S3 bucket.

        Args:
            bucket_name (str, optional): The bucket to list objects from.
                Defaults to the class's `BUCKET_NAME`.

        Returns:
            list: A list of object keys (filenames).
        """
        try:
            name = bucket_name if bucket_name else self.BUCKET_NAME
            response = self.s3.list_objects_v2(Bucket=name)
            objects = [obj["Key"] for obj in response.get("Contents", [])]
            print(f"Objects in bucket '{name}':", objects)
            return objects
        except Exception as e:
            print(f"Error listing objects in bucket: {e}")
            return []
