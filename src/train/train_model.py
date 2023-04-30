# Description: Script to train all models

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pandas as pd
import random_forest, gradient_boosting_machine
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import fbeta_score, make_scorer


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def get_data() -> pd.DataFrame:
    """
    Get the canonical data from data/processed folder for training
    """

    print("Reading canonical data from data/processed folder and splittin features and target ...")
    
    df_train = pd.read_csv("../../data/processed/train.csv")
    df_test = pd.read_csv("../../data/processed/test.csv")

    # Separating features and target
    X_train = df_train.drop(columns=["Status"])
    y_train = df_train["Status"]

    X_test = df_test.drop(columns=["Status"])
    y_test = df_test["Status"]

    return X_train, y_train, X_test, y_test

def train_models(X_train: pd.DataFrame, y_train: pd.DataFrame, model) -> None:
    """
    Training the models
    """

    print(f"Training {model} model ...")

    # Training model
    scorer = {"AUC": "roc_auc", "F_score": make_scorer(fbeta_score, beta=1)}
    cv_split = ShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=101)
    best_cv_params, best_model = model.train(X_train, y_train, scorer, cv_split)

    return best_cv_params, best_model

def evaluate_models(X_test: pd.DataFrame, y_test: pd.DataFrame, best_model, best_params, model) -> None:
    """
    Evaluating the best models
    """

    print(f"Evaluating {model} model ...")

    # Evaluating model
    model.evaluate(best_params, best_model, X_test, y_test)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # Get data
    X_train, y_train, X_test, y_test = get_data()

    # List of available models
    # models = [random_forest, 
    #           gradient_boosting_machine]
    
    models = {random_forest: False, gradient_boosting_machine: True}

    # Performing training and evaluation for all models
    for model in models:
        if models[model]:
            best_cv_params, best_model = train_models(X_train, y_train, model)
            evaluate_models(X_test, y_test, best_model, best_cv_params, model)
        else:
            print(f"Skipping {model} model ...")
