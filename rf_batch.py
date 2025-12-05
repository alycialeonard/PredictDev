#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: rf.py
Author: Alycia Leonard
Date: 2025-12-05
Version: 1.0
Description: rf modelling script for UPV dataset - run experiment batch using function from rf.py
License: GNU GPL-3.0
Contact: alycia.leonard@eng.ox.ac.uk
"""

import pandas as pd
import os
import warnings
from rf import rf_experiment
from functions import safe_list_parser

# Suppress tight layout warnings (inevitable due to SHAP code)
warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")

# --------- RUN THE EXPERIMENTS ---------- #
def main():
    # Get df of experiments to run
    cwd = os.getcwd()
    exp = pd.read_csv(os.path.join(cwd, 'data', 'experiments.csv'), low_memory=False)
    # Cycle through rows in the dataframe, i.e., experiments to run
    for index, row in exp.iterrows():
        # Define target for prediction.
        tar = row['target']
        # Define short-form of target to use in file saving
        tar_short = row['target_short']
        # Define questions to drop from predictors - parse safely to prevent list issue
        to_drop = safe_list_parser(row['stems_to_drop'])
        # Run the experiment
        rf_experiment(tar, tar_short, to_drop)


if __name__ == "__main__":
    main()