#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: rf.py
Author: Alycia Leonard
Date: 2025-12-05
Version: 1.0
Description: rf modelling script for UPV dataset - run experiment batch using function
License: GNU GPL-3.0
Contact: alycia.leonard@eng.ox.ac.uk
"""

import pandas as pd
import os
import warnings
from functions import safe_list_parser, run_experiment
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# Suppress tight layout warnings (inevitable due to SHAP code)
warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")

# --------- RUN THE EXPERIMENTS ---------- #
def main():

    # Get dataframe of experiments to run
    cwd = os.getcwd()
    exp = pd.read_csv(os.path.join(cwd, 'data', 'experiments.csv'), low_memory=False)

    # Define classifier
    clf = RandomForestClassifier(random_state=42)
    clf_short = "rf"

    # Define parameters to test for hyperparameters
    param_dist = {
        'classifier__n_estimators': randint(200, 801),
        'classifier__max_depth': [None, 8, 12, 16, 20],
        'classifier__min_samples_leaf': randint(1, 6),
        'classifier__min_samples_split': randint(2, 11),
        'classifier__max_features': ['sqrt', 'log2', 0.2, 0.5],
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }

    # Cycle through rows in the dataframe, i.e., experiments to run
    for index, row in exp.iterrows():

        # Define target, short-form of target for filenames, questions to drop
        tar = row['target']
        tar_short = row['target_short']
        to_drop = safe_list_parser(row['stems_to_drop'])

        # Run the experiment
        run_experiment(tar, tar_short, to_drop, clf, clf_short, param_dist)


if __name__ == "__main__":
    main()