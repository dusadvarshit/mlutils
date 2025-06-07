import kagglehub
from dotenv import load_dotenv

load_dotenv()


# Download latest version
def fetch_kaggle_dataset(dataset_name: str):
    """
    Fetch a Kaggle dataset by its name.

    :param dataset_name: The name of the Kaggle dataset to fetch.
    :path: Path to the downloaded dataset files.
    """
    path = kagglehub.dataset_download(dataset_name)
    return path
