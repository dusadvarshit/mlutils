import os

import boto3
import pytest
from moto import mock_aws

from mlutils.cloud.aws import AWSStorage  # Change this to your actual module path

REGION = "us-east-2"
BUCKET_NAME = "expt"
TEST_FILE_NAME = "test_file.txt"
S3_KEY = "uploaded_test_file.txt"


@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-2"


@pytest.fixture(scope="function")
def aws_storage(aws_credentials):
    """Provides a mocked AWSStorage instance with a pre-created bucket."""
    with mock_aws():
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket=BUCKET_NAME, CreateBucketConfiguration={"LocationConstraint": REGION})
        yield AWSStorage(s3=s3, BUCKET_NAME=BUCKET_NAME)


@pytest.fixture
def temp_file():
    """Creates a temporary file for upload testing."""
    with open(TEST_FILE_NAME, "w") as f:
        f.write("dummy content")
    yield TEST_FILE_NAME
    os.remove(TEST_FILE_NAME)


def test_upload_file(aws_storage, temp_file):
    aws_storage.upload_file(temp_file, S3_KEY)
    objects = aws_storage.list_objects()
    assert S3_KEY in objects


@mock_aws
def test_generate_presigned_url(aws_storage, temp_file):
    aws_storage.upload_file(temp_file, S3_KEY)
    url = aws_storage.generate_presigned_url_for_object(S3_KEY)
    assert isinstance(url, str)
    assert "https://" in url


@mock_aws
def test_create_bucket(aws_storage):
    aws_storage.create_bucket("new-bucket")
    buckets = aws_storage.list_buckets()
    assert "new-bucket" in buckets


@mock_aws
def test_list_buckets(aws_storage):
    buckets = aws_storage.list_buckets()
    assert BUCKET_NAME in buckets


@mock_aws
def test_delete_bucket(aws_storage):
    aws_storage.create_bucket(bucket_name="bucket-to-delete")

    aws_storage.delete_bucket("bucket-to-delete")

    result = aws_storage.list_buckets()
    assert "bucket-to-delete" not in result


@mock_aws
def test_list_objects_empty_bucket(aws_storage):
    objects = aws_storage.list_objects()
    assert objects == []
