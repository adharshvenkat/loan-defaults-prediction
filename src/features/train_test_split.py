# Description: Script to split data into train and test sets

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def get_data() -> pd.DataFrame:
    """
    Get data from data/processed folder
    """

    print("Reading data from data/interim folder ...")
    
    df = pd.read_csv("../../data/interim/loan_default_processed.csv")

    print("total data size: ", df.shape)

    return df

def split_train_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split data into train and test sets
    """

    print("Splitting data into train and test sets ...")

    # Splitting data into train and test sets
    train, test = train_test_split(df, test_size=0.2, random_state=101)

    return train, test

def save_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """
    Save data to data/processed folder
    """

    # Saving train and test sets to data/processed folder
    df_train.to_csv("../../data/processed/train.csv", index=False)
    df_test.to_csv("../../data/processed/test.csv", index=False)

    print("train data size: ", df_train.shape)
    print("test data size: ", df_test.shape)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    df = get_data()
    train, test = split_train_test(df)
    save_data(train, test)