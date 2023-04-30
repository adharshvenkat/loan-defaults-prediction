# Description: Adaboost model training & evaluation

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, roc_curve, roc_auc_score
import pickle


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def train(X_train, y_train, scorer, cv_split):
    """
    Hyperparameter tuning using RandomizedSearchCV, and training the model
    """

    # Setting up hyperparemeter grid values
    adaboost_params_grid = {
        "adaboost__n_estimators": np.arange(50, 700, 50), 
        "adaboost__learning_rate": np.arange(0.1, 1.0, 0.05),
        "adaboost__algorithm": ["SAMME", "SAMME.R", "RealBoost"],
        "adaboost__estimator": [DecisionTreeClassifier(max_depth=d) for d in range(1,6)]
    }

    # Defining the base model
    adaboost = AdaBoostClassifier(random_state=101)
    
    # Setting up the pipeline
    adaboost_pipe = Pipeline([("adaboost", adaboost)])

    # Cross validation using RandomizedSearchCV
    adaboost_cv = RandomizedSearchCV(
        estimator=adaboost_pipe, 
        param_distributions=adaboost_params_grid, 
        n_iter=15, 
        scoring=scorer, 
        refit="F_score",
        n_jobs=-1, # use all cores
        cv=cv_split, # number of folds
        return_train_score=True, # return training results
        verbose=3, # print progress
        random_state=101,
    )

    # Fitting the model
    adaboost_cv.fit(X_train, y_train)

    # Getting the best model from cross validation
    adaboost_best_pipe = adaboost_cv.best_estimator_

    return adaboost_cv, adaboost_best_pipe

def evaluate(adaboost_cv, adaboost_best_pipe, X_test, y_test):
    """
    Evaluate the model with the best hyperparameters
    """

    # Evaluating the model with the best hyperparameters (create seperate function to share with other models)
    print(
        f"""
    --------------
    TUNING RESULTS
    --------------
    ESTIMATOR: {adaboost_cv.estimator}
    BEST SCORE: {adaboost_cv.best_score_:.2%}
    BEST PARAMS: {adaboost_cv.best_params_}
    TRAIN AUC: {adaboost_cv.cv_results_["mean_train_AUC"][adaboost_cv.best_index_]:.2%}
    TRAIN AUC SD: {adaboost_cv.cv_results_["std_train_AUC"][adaboost_cv.best_index_]:.2%}
    TEST AUC: {adaboost_cv.cv_results_["mean_test_AUC"][adaboost_cv.best_index_]:.2%}
    TEST AUC SD: {adaboost_cv.cv_results_["std_test_AUC"][adaboost_cv.best_index_]:.2%}
    TRAIN F_score: {adaboost_cv.cv_results_['mean_train_F_score'][adaboost_cv.best_index_]:.2%}
    TEST F_score: {adaboost_cv.cv_results_['mean_test_F_score'][adaboost_cv.best_index_]:.2%}  
    """
    )

    # Getting the predictin probabilities
    y_pred_prob = adaboost_best_pipe.predict_proba(X_test)[:, 1]

    # Getting the predictions
    y_pred = adaboost_best_pipe.predict(X_test)

    # Storing the report of the best model's peadaboostormance
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
    with open("../../models/adaboost.pickle", "wb") as f:
        pickle.dump(adaboost_best_pipe, f)
