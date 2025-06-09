from pathlib import Path

import pandas as pd
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from sklearn.model_selection import train_test_split

from mlutils.utils.config import param_grid_fix


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MyDataset:
    path: str
    name: str
    target_col: str
    param_grid_map: dict
    type: str = "classification"  # Default to classification type
    cross_val_scoring = list = ["accuracy"]
    optimization_type: str = "grid_search"  # Default to grid search
    index_col: str | None = None  # Default to None, can be set to a specific column name
    imbalanced: bool = False  # Default to not imbalanced, only for classification
    binary: bool | None = True  # Default to not binary, only for classification
    label_encoding: dict | None = None  # Default to not using label encoder, only for classification

    def __post_init__(self):
        self.return_optimization_specific_param_grid()

    def display_basic_info(self):
        print("Dataset name", self.name)
        print("Dataset type", self.type)
        print("Dataset path", self.path)
        print("Target column", self.target_col)

        if self.type == "classification":
            print("Imbalanced:", self.imbalanced)
            print("Binary:", self.binary)

    def return_optimization_specific_param_grid(self) -> None:
        """
        Returns the parameter grid for a specific model.

        Returns:
            dict: The parameter grid for the specified optimization_type.
        """
        self.param_grid_map = param_grid_fix(self.param_grid_map, self.optimization_type)

        return None


def remove_high_null_features(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Removes features from the DataFrame that have a high percentage of null values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        threshold (float, optional): The threshold for the percentage of null values.
            Features with a percentage of null values greater than this threshold will be removed.
            Defaults to 0.5 (50%).

    Returns:
        pd.DataFrame: A DataFrame with features having high null values removed.
    """

    null_percentage = df.isnull().mean()
    features_to_remove = null_percentage[null_percentage > threshold].index
    return df.drop(columns=features_to_remove)


def auto_divide_categorical_variables(df: pd.DataFrame, high_cardinality_threshold: int = 10) -> dict:
    """
    Automatically detects and categorizes categorical variables in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        high_cardinality_threshold (int): Threshold for high cardinality features. Default is 10.

    Returns:
        dict: A dictionary containing lists of column names categorized as:
            - 'binary': Categorical variables with 2 unique values
            - 'one_hot': Categorical variables with 3-10 unique values
            - 'high_cardinality': Categorical variables with more than 10 unique values
    """

    categorical_vars = df.select_dtypes(include=["object", "category"]).columns.tolist()

    binary_vars = []
    one_hot_vars = []
    high_cardinality_vars = []

    for col in categorical_vars:
        n_unique = df[col].nunique()
        if n_unique == 2:
            binary_vars.append(col)
        elif n_unique <= high_cardinality_threshold:
            one_hot_vars.append(col)
        else:
            high_cardinality_vars.append(col)

    return {"binary": binary_vars, "one_hot": one_hot_vars, "high_cardinality": high_cardinality_vars}


def read_local_data(my_dataset: MyDataset, null_threshold: float = 0.5) -> tuple:
    """
    Reads data from a CSV file into a pandas DataFrame.
    Args:
        my_dataset (MyDataset): An instance of MyDataset containing the dataset information.
        null_threshold (float): The threshold for removing features with high null values. Defaults to 0.5.
    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the CSV file.
    """

    df = pd.read_csv(my_dataset.path)
    if my_dataset.index_col:
        df.set_index(my_dataset.index_col, inplace=True)

    df = remove_high_null_features(df, threshold=null_threshold)

    X = df.drop(my_dataset.target_col, axis=1)
    y = df[my_dataset.target_col]

    if my_dataset.label_encoding:
        y = y.map(my_dataset.label_encoding)

    return X, y


def split_train_test(X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2) -> tuple:
    """Splits the data into training and testing sets.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.

    Returns:
        tuple: A tuple containing the training and testing sets for the features and target variable, respectively.
               The tuple is in the order (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def find_git_root():
    start_path = Path.cwd()

    current = start_path
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return None
