#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: functions.py
Author: Alycia Leonard
Date: 2025-11-26
Version: 1.0
Description: Helper functions for various analyses in this repo.
License: GNU GPL-3.0
Contact: alycia.leonard@eng.ox.ac.uk
"""

import ast
import re
import pandas as pd
import numpy as np

def safe_list_parser(x):
    if isinstance(x, str):
        try: # Try to parse as a Python literal
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError): # Fall back to comma split
            return [item.strip() for item in x.split(',')]
    return x  # Leave non-strings untouched


# Normalize column names: strip whitespace, collapse multiple spaces, remove non-breaking spaces
def clean_colname(c):
    if pd.isna(c):
        return c
    c = str(c)
    c = c.replace('\xa0', ' ')                 # non-breaking space
    c = re.sub(r'\s+', ' ', c).strip()         # collapse spaces and strip
    return c


# Read a CSV file and convert it to a list. Assumes CSV has one column and a header.
def csv_to_list(path, header):
    df = pd.read_csv(path, delimiter=',')
    return df[header].to_list()


# Get item selection for a certain paragraph number from utterances data, by interviewee
def items_from_paragraph_number(df_utterances, paragraph_number, new_column_header):
    subset = df_utterances[['Paragraph Number', 'Item Name', 'Interview ID']].copy()
    # Get the entries associated with paragraph 8 (most valued appliance)
    subset = subset[subset['Paragraph Number'] == paragraph_number]
    # Drop duplicates so there's one row per interview ID
    subset = subset.drop_duplicates()
    # Drop paragraph number since we don't need it anymore
    subset = subset[['Item Name', 'Interview ID']]
    # Rename the item column to match the question text
    subset = subset.rename(columns={'Item Name': new_column_header})
    return subset


# Normalize shap values to a 2D array shap_for_plot with shape (n_samples, n_features)
def normalize_shap_values(shap_values):
    if isinstance(shap_values, list):
        # common case: list of arrays, length == n_classes
        if len(shap_values) == 2:
            shap_for_plot = shap_values[1] # positive class
            chosen = "list[1] (positive class)"
        else:
            shap_for_plot = shap_values[-1]  # fallback: last class
            chosen = f"list[{len(shap_values)-1}] (fallback last class)"
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 2:
            shap_for_plot = shap_values  # already (n_samples, n_features)
            chosen = "ndarray (2D)"
        elif shap_values.ndim == 3:
            # shape (n_samples, n_features, n_classes) -> pick positive class at last axis
            if shap_values.shape[2] >= 2:
                shap_for_plot = shap_values[..., 1]  # positive class
                chosen = "ndarray (3D) -> slice [:,:,1] (positive class)"
            else:
                # single class in last axis? fallback to first
                shap_for_plot = shap_values[..., 0]
                chosen = "ndarray (3D) -> slice [:,:,0] (fallback)"
        else:
            raise ValueError("Unexpected shap_values ndarray ndim="+str(shap_values.ndim))
    else:
        raise TypeError("Unexpected type for shap_values: " + str(type(shap_values)))
    return chosen, shap_for_plot