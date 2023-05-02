# Description: Support Vector Machine model training & evaluation

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, roc_curve, roc_auc_score
import pickle


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def train(X_train, y_train, scorer, cv_split):
    """
    Hyperparameter tuning using RandomizedSearchCV, Principal Component Analysis,
    and training the model
    """

    # Setting up hyperparemeter grid values
    svm_params_grid = {
        'pca__n_components': np.arange(10, 40, 5),
        'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'svm__C': [0.01, 0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7))
    }

    # Defining the base model
    svm = SVC(max_iter=300, probability=True, random_state=101, verbose=True)

    # PCA and min-max scaling
    pca = PCA(random_state=101)
    mm_scaler = MinMaxScaler()
    
    # Setting up the pipeline
    svm_pipe = Pipeline([("mm_scaler", mm_scaler), ("pca", pca),("svm", svm)])

    # Cross validation using RandomizedSearchCV
    svm_cv = RandomizedSearchCV(
        estimator=svm_pipe, 
        param_distributions=svm_params_grid, 
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
    svm_cv.fit(X_train, y_train)

    # Getting the best model from cross validation
    svm_best_pipe = svm_cv.best_estimator_

    return svm_cv, svm_best_pipe

def evaluate(svm_cv, svm_best_pipe, X_test, y_test):
    """
    Evaluate the model with the best hyperparameters
    """

    # Evaluating the model with the best hyperparameters (create seperate function to share with other models)
    print(
        f"""
    --------------
    TUNING RESULTS
    --------------
    ESTIMATOR: {svm_cv.estimator}
    BEST SCORE: {svm_cv.best_score_:.2%}
    BEST PARAMS: {svm_cv.best_params_}
    TRAIN AUC: {svm_cv.cv_results_["mean_train_AUC"][svm_cv.best_index_]:.2%}
    TRAIN AUC SD: {svm_cv.cv_results_["std_train_AUC"][svm_cv.best_index_]:.2%}
    TEST AUC: {svm_cv.cv_results_["mean_test_AUC"][svm_cv.best_index_]:.2%}
    TEST AUC SD: {svm_cv.cv_results_["std_test_AUC"][svm_cv.best_index_]:.2%}
    TRAIN F_score: {svm_cv.cv_results_['mean_train_F_score'][svm_cv.best_index_]:.2%}
    TEST F_score: {svm_cv.cv_results_['mean_test_F_score'][svm_cv.best_index_]:.2%}  
    """
    )

    # Getting the predictin probabilities
    y_pred_prob = svm_best_pipe.predict_proba(X_test)[:, 1]

    # Getting the predictions
    y_pred = svm_best_pipe.predict(X_test)

    # Storing the report of the best model's pesvmormance
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
    with open("../../models/support_vector_machine.pickle", "wb") as f:
        pickle.dump(svm_best_pipe, f)
