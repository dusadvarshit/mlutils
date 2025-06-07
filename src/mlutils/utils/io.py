import pandas as pd
from sklearn.model_selection import train_test_split


def read_local_data(path: str, target_col: str) -> tuple:
    """
    Reads data from a CSV file into a pandas DataFrame.
    Args:
        path (str, optional): The path to the CSV file.
            Defaults to '../../../data/churn_data.csv'.
    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the CSV file.
    """

    df = pd.read_csv(path)
    df["TotalCharges"] = df["TotalCharges"].apply(lambda x: float(x) if x != " " else 0)
    df.loc[(df[df["TotalCharges"] == 0]).index, "TotalCharges"] = df.loc[(df[df["TotalCharges"] == 0]).index, "MonthlyCharges"]
    df.set_index("customerID", inplace=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    return X, y


def split_train_test(X: pd.DataFrame, y: pd.DataFrame) -> tuple:
    """Splits the data into training and testing sets.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.

    Returns:
        tuple: A tuple containing the training and testing sets for the features and target variable, respectively.
               The tuple is in the order (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def export_img():
    pass
