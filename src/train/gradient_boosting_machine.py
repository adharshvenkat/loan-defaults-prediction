# Description: Gradient Boosting model training & evaluation

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import xgboost as xgb
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
    gbm_params_grid = {
        "gbm__n_estimators": np.arange(100, 700, 50), 
        "gbm__max_depth": np.arange(3, 10, 1), 
        "gbm__learning_rate": np.arange(0.01, 0.2, 0.01), 
        "gbm__reg_alpha": list(np.linspace(0, 1, 11)), # L1 regularization term
        "gbm__reg_lambda": list(np.linspace(0, 1, 11)), # L2 regularization term
    }

    # Defining the base model
    gbm = xgb.XGBClassifier(
        objective="binary:logistic", 
        booster="gbtree", 
        n_jobs=-1, # use all cores
        random_state=101, 
        use_label_encoder=False, # set to False as label is already numeric
        )
    
    # Setting up the pipeline
    gbm_pipe = Pipeline([("gbm", gbm)])

    # Cross validation using RandomizedSearchCV
    gbm_cv = RandomizedSearchCV(
        estimator=gbm_pipe, 
        param_distributions=gbm_params_grid, 
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
    gbm_cv.fit(X_train, y_train)

    # Getting the best model from cross validation
    gbm_best_pipe = gbm_cv.best_estimator_

    return gbm_cv, gbm_best_pipe

def evaluate(gbm_cv, gbm_best_pipe, X_test, y_test):
    """
    Evaluate the model with the best hyperparameters
    """

    # Evaluating the model with the best hyperparameters (create seperate function to share with other models)
    print(
        f"""
    --------------
    TUNING RESULTS
    --------------
    ESTIMATOR: {gbm_cv.estimator}
    BEST SCORE: {gbm_cv.best_score_:.2%}
    BEST PARAMS: {gbm_cv.best_params_}
    TRAIN AUC: {gbm_cv.cv_results_["mean_train_AUC"][gbm_cv.best_index_]:.2%}
    TRAIN AUC SD: {gbm_cv.cv_results_["std_train_AUC"][gbm_cv.best_index_]:.2%}
    TEST AUC: {gbm_cv.cv_results_["mean_test_AUC"][gbm_cv.best_index_]:.2%}
    TEST AUC SD: {gbm_cv.cv_results_["std_test_AUC"][gbm_cv.best_index_]:.2%}
    TRAIN F_score: {gbm_cv.cv_results_['mean_train_F_score'][gbm_cv.best_index_]:.2%}
    TEST F_score: {gbm_cv.cv_results_['mean_test_F_score'][gbm_cv.best_index_]:.2%}  
    """
    )

    # Getting the predictin probabilities
    y_pred_prob = gbm_best_pipe.predict_proba(X_test)[:, 1]

    # Getting the predictions
    y_pred = gbm_best_pipe.predict(X_test)

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
    with open("../../models/gradient_boosting_machine.pickle", "wb") as f:
        pickle.dump(gbm_best_pipe, f)
