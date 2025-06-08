import boto3

## Accessing .env file
from dotenv import load_dotenv

load_dotenv()
####

s3 = boto3.client("s3", region_name="us-east-2")


BUCKET_NAME = "expt"
FOLDER_NAME = "test_dump"


def upload_file_to_audios(fileobj, filename):
    print("Uploading to AWS!!!")
    s3_key = f"{FOLDER_NAME}/{filename}"

    s3.upload_fileobj(fileobj, BUCKET_NAME, s3_key)

    return None


def generate_presigned_url(s3_filename, expiration=3600 * 24 * 7):
    """
    Generate a presigned URL for an S3 object.

    :param s3_filename: The filename of the S3 object
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as a string
    """
    try:
        s3_key = f"{FOLDER_NAME}/{s3_filename}"
        response = s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET_NAME, "Key": s3_key}, ExpiresIn=expiration)

    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None

    return response
