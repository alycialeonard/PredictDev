#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: plots.py
Author: Alycia Leonard
Date: 2025-11-26
Version: 1.0
Description: Compute stratified descriptive statistics to better understand the dataset.
License: GNU GPL-3.0
Contact: alycia.leonard@eng.ox.ac.uk
"""

import pandas as pd
import os
from functions import csv_to_list
import matplotlib.pyplot as plt

# Load dataset
cwd = os.getcwd()
df = pd.read_csv(os.path.join(cwd, 'data', "Kenya_UPV_Survey_Preprocessed_AllCols.csv"), low_memory=False)

# Define path to save results
plots_save_path = os.path.join(cwd, 'results', 'descriptive_statistics', 'plots')
os.makedirs(plots_save_path, exist_ok=True)
csv_save_path = os.path.join(cwd, 'results', 'descriptive_statistics', 'csvs')
os.makedirs(csv_save_path, exist_ok=True)

# Load lists of columns
cols_path = os.path.join(cwd, 'data', 'cols')
onehot_cols_annotation = csv_to_list(os.path.join(cols_path, "onehot_cols_annotation.csv"), 'onehot_cols_annotation')
onehot_cols_appliance_least = csv_to_list(os.path.join(cols_path, "onehot_cols_appliance_least.csv"), 'onehot_cols_appliance_least')
onehot_cols_appliance_most = csv_to_list(os.path.join(cols_path, "onehot_cols_appliance_most.csv"), 'onehot_cols_appliance_most')
onehot_cols_climate = csv_to_list(os.path.join(cols_path, "onehot_cols_climate.csv"), 'onehot_cols_climate')
onehot_cols_upv = csv_to_list(os.path.join(cols_path, "onehot_cols_upv.csv"), 'onehot_cols_upv')

# Put all these lists of columns in a list to cycle through, with short-names for plotting, in a tuple
column_groups = [("Annotations", onehot_cols_annotation),
                 ("Least valuable appliance", onehot_cols_appliance_least),
                 ("Most valuable appliance", onehot_cols_appliance_most),
                 ("Climate UPV items", onehot_cols_climate),
                 ("General UPV items", onehot_cols_upv),
                 ]


# Define demographics to study. Tuples: shorthand name (i.e., for plotting), question to use to stratify.
demographics = [("Gender", "What is your gender?"),
                ("Age", "Age (Range)"),
                ("Marital Status", "Marital Status"),
                ("Education", "What is the highest level of education you have completed?"),
                ("Occupation", "What is your main occupation?"),
                ("Household size", "Total household size"),
                ("Household Income", "What is the monthly total income of your household (KES)?")
                ]

# Get + plot stratified proportions by different demographics
for columns in column_groups:
    # Get proportions across the whole dataset
    proportions = df[columns[1]].mean().sort_values(ascending=False)
    # Save the results
    proportions.to_csv(os.path.join(csv_save_path, f"{columns[0]}_Prevalence.csv"))
    # Plot the top 10
    top10 = proportions.head(10)
    top10_cols = top10.index.tolist()
    fig, ax = plt.subplots()
    top10.T.plot(kind='bar', ax=ax)
    ax.set_ylabel("Proportion")
    ax.set_title(f"{columns[0]} proportions in dataset (Top 10)")
    plt.savefig(os.path.join(plots_save_path, f"{columns[0]}_Prevalence.png"), bbox_inches='tight')
    plt.close(fig)
    for demographic in demographics:
        # Get proportions by demographics
        stratified_proportions = df.groupby(demographic[1])[columns[1]].mean()
        # Save the results
        stratified_proportions.to_csv(os.path.join(csv_save_path, f"{columns[0]}_Prevalence_{demographic[0]}.csv"))
        # Plot the top 10
        stratified_top10 = stratified_proportions[top10_cols]
        fig, ax = plt.subplots()
        stratified_top10.T.plot(kind='bar', ax=ax)
        ax.set_ylabel("Proportion")
        ax.set_title(f"{columns[0]} stratified by {demographic[0]}")
        plt.savefig(os.path.join(plots_save_path, f"{columns[0]}_Prevalence_{demographic[0]}.png"), bbox_inches='tight')
        plt.close(fig)