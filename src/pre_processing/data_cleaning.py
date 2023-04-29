# Description: Script to clean data

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pandas as pd

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def get_data() -> pd.DataFrame:
    """
    Get data from data/raw folder
    """

    print("Reading data from data/raw folder ...")
    
    df = pd.read_csv("../../data/raw/loan_default.csv")

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data
    """

    print("Cleaning data ...")

    # Drop columns as mentioned in the EDA section
    df.drop(
            columns=[
                "ID",
                "year",
                "Interest_rate_spread",
                "rate_of_interest",
                "Upfront_charges",
                "Secured_by",
                "construction_type",
                "Security_Type",
                "Credit_Worthiness",
                "open_credit",
                "interest_only",
                "lump_sum_payment",
                "total_units",
            ],
            inplace=True,
        )
    
    # Drop rows with missing values for important columns
    df.dropna(
            axis=0,
            subset=[
                "loan_amount",
                "term",
                "Neg_ammortization",
                "age",
            ],
            inplace=True,
        )
        
    return df

def save_data(df: pd.DataFrame) -> None:
    """
    Save data to data/interim folder
    """

    print("Saving data to data/interim folder ...")
    
    df.to_csv("../../data/interim/loan_default_cleaned.csv", index=False)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    df = get_data()

    df_cleaned = clean_data(df)

    save_data(df_cleaned)
