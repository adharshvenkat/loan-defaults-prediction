# Description: Feature engineering for loan default prediction model

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pandas as pd
from sklearn.impute import KNNImputer

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def get_data() -> pd.DataFrame:
    """
    Get data from data/raw folder
    """

    print("Reading data from data/interim folder ...")
    
    df = pd.read_csv("../../data/interim/loan_default_cleaned.csv")

    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features
    """

    print("Building features ...")

    # Imputation of missing values using knn imputer for  "income", "property_value", "LTV", and "dtir1"
    for col in ["income", "property_value", "LTV", "dtir1"]:
        imputer = KNNImputer(n_neighbors=3)
        df[col] = imputer.fit_transform(df[[col]])

    # Replacing ">" and "<" in age column
    df["age"] = df["age"].str.replace(">", "greater_than_")
    df["age"] = df["age"].str.replace("<", "lesser_than_")
        
    # Creating a new column with 5 income bins based on lowest and highest income
    df["income_bin"] = pd.cut(df["income"], bins=[0, 2000, 4000, 6000, 10000, 1000000], labels=["0-2k", "2k-4k", "4k-6k", "6k-10k", "more_than_10k"])
    df.drop(columns=["income"], inplace=True)

    # Creating a new column with 5 property value bins based on lowest and highest property value
    df["property_value_bin"] = pd.cut(df["property_value"], bins=[0, 200000, 400000, 600000, 1000000, 100000000], labels=["0-200k", "200k-400k", "400k-600k", "600k-1M", "greater_than_1M"])
    df.drop(columns=["property_value"], inplace=True)

    # Vectorizing the categorical/ object columns
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype == "category":
            dummies = pd.get_dummies(
                df[col], prefix=col, prefix_sep="_", drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=col, inplace=True)

    return df

def save_data(df: pd.DataFrame) -> None:
    """
    Save data to data/processed folder
    """

    print("Saving data to data/interim folder ...")
    
    df.to_csv("../../data/interim/loan_default_processed.csv", index=False)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    df = get_data()

    df_features = build_features(df)

    save_data(df_features)
