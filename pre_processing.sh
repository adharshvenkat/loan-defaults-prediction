#!/usr/bin/env python 


cd /home/loan-defaults-detection/src/pre_processing
python3 data_cleaning.py
cd /home/loan-defaults-detection/src/features
python3 feature_engineering.py
python3 train_test_split.py
