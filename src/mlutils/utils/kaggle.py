import kagglehub


# Download latest version
def fetch_kaggle_dataset(dataset_name: str, path: str = None):
    """
    Fetch a Kaggle dataset by its name.

    :param dataset_name: The name of the Kaggle dataset to fetch.
    :param path: Optional path to save the dataset files.
    :return: Path to the downloaded dataset files.
    """
    return kagglehub.dataset_download(dataset_name, path=path)
