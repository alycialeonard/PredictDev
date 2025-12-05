#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: xgb.py
Author: Alycia Leonard
Date: 2025-12-05
Version: 1.0
Description: xgb modelling script for UPV dataset - run a single experiment
License: GNU GPL-3.0
Contact: alycia.leonard@eng.ox.ac.uk
"""

from functions import run_experiment
from scipy.stats import randint
import warnings
from xgboost import XGBClassifier

# Suppress tight layout warnings (inevitable due to SHAP code)
warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")

# --------- RUN THE EXPERIMENT ---------- #
def main():

    # Define target for prediction.
    tar = 'Which 5 items are most important to you in your daily life? Please indicate these in order of importance, starting with the most important_Electricity'

    # Define short-form of target to use in file saving
    tar_short = 'upv_electricity'

    # Define questions to drop from predictors
    to_drop = ['Which of the following items do you have access to in your daily life?',
               'Which 5 items are most important to you in your daily life? Please indicate these in order of importance, starting with the most important',
               'Given the chosen climate event - which 3 items are most useful to you?']

    # Define XGBoost classifier (without tuned params). Use_label_encoder=False and eval_metric to avoid deprecation warnings
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)

    clf_short = 'xgb'

    # Define param_dist to search (for hyperparameters)
    param_dist = {
        'classifier__n_estimators': randint(100, 1001),
        'classifier__max_depth': randint(3, 16),
        'classifier__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0],
        'classifier__gamma': [0, 0.1, 0.5, 1],
        'classifier__reg_alpha': [0, 0.1, 0.5, 1.0],
        'classifier__reg_lambda': [1, 5, 10]
    }

    # Run the experiment
    run_experiment(tar, tar_short, to_drop, clf, clf_short, param_dist)

if __name__ == "__main__":
    main()