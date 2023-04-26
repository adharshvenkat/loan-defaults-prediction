Loan Defaults Prediction
==============================

About this Project <a name="1"></a>
------------

Loans are a major source of revenue for banks, however, these are associated with high risk. A lot of the times, banks end up lending bad loans if the borrower defaults on the loan repayment. This may cause the banks to incur huge losses, potential Bankruptcy or even a global economic slowdown, as we have seen in the past (The 2007 - 2008 Financial Crisis). 

The aim of this project is to help banks mitigate this issue by providing a highly reliable and efficient Machine Learning solution that can help determine if any new borrower is likely to default or not. 


About the data <a name="2"></a>
------------

The dataset has been taken from Kaggle, titled [Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset). It consists of past data on loan borrowers and contains a lot of features such as borrower's income, gender, loan purpose, rate of interest, term, etc.


Project Organization <a name="3"></a>
------------

The repository is structured in the following hierarchy:


    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed
    │   ├── processed      <- The final, canonical data sets for modeling
    │   └── raw            <- The original, immutable data dump
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── data_cleaning.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── feature_engineering.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

