# Description: Script to train a stacking ensemble model

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, roc_curve, roc_auc_score


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def train(X_train, y_train, scorer, cv_split):
    """
    Performing hyperparameter tuning and training the ensemble model
    """

    # Setting up hyperparemeter grid values
    lr_params_grid = {
        'C': loguniform(0.01, 100),
        'max_iter': np.arange(50, 700, 50),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'lbfgs']
    }

    base_model = LogisticRegression(n_jobs=-1, random_state=101, verbose=True)

    # Importing the best models from the previous scripts
    with open("../../models/random_forest.pickle", "rb") as f:
        rf_best_pipe = pickle.load(f)

    with open("../../models/adaboost.pickle", "rb") as f:
        adaboost_best_pipe = pickle.load(f)

    with open("../../models/gradient_boosting_machine.pickle", "rb") as f:
        gbm_best_pipe = pickle.load(f)

    estimators = [('rf', rf_best_pipe), ('adaboost', adaboost_best_pipe), ('gbm', gbm_best_pipe)]

    # Cross validation using RandomizedSearchCV
    lr_cv = RandomizedSearchCV(
        estimator=base_model, 
        param_distributions=lr_params_grid, 
        n_iter=15, 
        scoring=scorer, 
        refit="F_score",
        n_jobs=-1, # use all cores
        cv=cv_split, # number of folds
        return_train_score=True, # return training results
        verbose=3, # print progress
        random_state=101,
    )

    lr_cv.fit(X_train, y_train)
    
    ensemble_model = StackingClassifier(estimators=estimators, final_estimator=lr_cv.best_estimator_, cv=5)

    ensemble_model.fit(X_train, y_train)

    return lr_cv, ensemble_model


def evaluate(lr_cv, ensemble_best_pipe, X_test, y_test):
    """
    Evaluate the model with the best hyperparameters
    """

    # Evaluating the model with the best hyperparameters (create seperate function to share with other models)
    print(
        f"""
    --------------
    TUNING RESULTS
    --------------
    ESTIMATOR: {lr_cv.estimator}
    BEST SCORE: {lr_cv.best_score_:.2%}
    BEST PARAMS: {lr_cv.best_params_}
    TRAIN AUC: {lr_cv.cv_results_["mean_train_AUC"][lr_cv.best_index_]:.2%}
    TRAIN AUC SD: {lr_cv.cv_results_["std_train_AUC"][lr_cv.best_index_]:.2%}
    TEST AUC: {lr_cv.cv_results_["mean_test_AUC"][lr_cv.best_index_]:.2%}
    TEST AUC SD: {lr_cv.cv_results_["std_test_AUC"][lr_cv.best_index_]:.2%}
    TRAIN F_score: {lr_cv.cv_results_['mean_train_F_score'][lr_cv.best_index_]:.2%}
    TEST F_score: {lr_cv.cv_results_['mean_test_F_score'][lr_cv.best_index_]:.2%}  
    """
    )

    # Getting the predictin probabilities
    y_pred_prob = ensemble_best_pipe.predict_proba(X_test)[:, 1]

    # Getting the predictions
    y_pred = ensemble_best_pipe.predict(X_test)

    # Storing the report of the best model's pegbmormance
    report = {}
    report["accuracy"] = accuracy_score(y_test, y_pred)
    report["precision"] = precision_score(y_test, y_pred)
    report["recall"] = recall_score(y_test, y_pred)
    report["f05"] = fbeta_score(y_test, y_pred, beta=0.5)
    report["f1"] = f1_score(y_test, y_pred)
    report["f2"] = fbeta_score(y_test, y_pred, beta=2)
    report["cf_matrix"] = confusion_matrix(y_test, y_pred, normalize="all")
    report["auroc"] = roc_auc_score(y_test, y_pred_prob)
    report["roc"] = roc_curve(y_test, y_pred_prob)

    print(
        f"""
    -----------
    PERFORMANCE 
    -----------
    ACCURACY: {report["accuracy"]:.2%}
    PRECISION: {report["precision"]:.2%}
    RECALL: {report["recall"]:.2%}
    F05: {report["f05"]:.2%}
    F1: {report["f1"]:.2%}
    F2: {report["f2"]:.2%}
    ROC AUC: {report["auroc"]:.2%}
    """
    )

    print("CONFUSION MATRIX")
    print(report["cf_matrix"])

    print("ROC CURVE")
    print(report["roc"])

    # Saving the model
    with open("../../models/logistic_regression_ensemble.pickle", "wb") as f:
        pickle.dump(ensemble_best_pipe, f)