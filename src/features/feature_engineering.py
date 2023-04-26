# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pandas as pd

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def get_data():
    """
    Get data from data/raw folder
    """

    print("Reading data from data/interim folder ...")
    
    df = pd.read_csv("../../data/interim/loan_default_cleaned.csv")

    return df

def build_features(df):
    """
    Build features
    """

    print("Building features ...")
    
    # Creating a new column for with 5 income bins based on lowest and highest income
    df["income_bin"] = pd.cut(df["income"], bins=[0, 2000, 4000, 6000, 10000, 1000000], labels=["0-2k", "2k-4k", "4k-6k", "6k-10k", ">10k"])

    return df

def save_data(df):
    """
    Save data to data/processed folder
    """

    print("Saving data to data/processed folder ...")
    
    df.to_csv("../../data/processed/loan_default_processed.csv", index=False)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    df = get_data()

    df_features = build_features(df)

    save_data(df_features)
