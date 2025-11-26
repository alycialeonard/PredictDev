#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: preprocess.py
Author: Alycia Leonard
Date: 2025-11-26
Version: 1.0
Description: This script prepares the data  for use in ML models.
    Assumes that the data are downloaded from https://zenodo.org/records/11242112
License: GNU GPL-3.0
Contact: alycia.leonard@eng.ox.ac.uk
"""

import pandas as pd
from functions import safe_list_parser, clean_colname, csv_to_list, items_from_paragraph_number
import os

# ----------- LOAD DATA ------------- #

# Get current working directory, data path, path to column files
cwd = os.getcwd()
data_path = os.path.join(cwd, 'data')
cols_path = os.path.join(data_path, "cols")

# Load survey and utterance datasets
df_u = pd.read_csv(os.path.join(data_path, "Kenya_UPV_Utterances.csv"), low_memory=False)
df_s = pd.read_csv(os.path.join(data_path, "Kenya_UPV_Survey.csv"), header=1, low_memory=False)

# Load lists of columns by type (derived in advance from inspection of the dataset)
categorical_cols = csv_to_list(os.path.join(cols_path, "categorical_cols.csv"), 'categorical_cols')
multi_label_cols = csv_to_list(os.path.join(cols_path, "multi_label_cols.csv"), 'multi_label_cols')
num_cols = csv_to_list(os.path.join(cols_path, "numeric_cols.csv"), 'numeric_cols')
meta_cols = csv_to_list(os.path.join(cols_path, "meta_cols.csv"), 'meta_cols')
text_cols = csv_to_list(os.path.join(cols_path, "text_cols.csv"), 'text_cols')
explode_cols = csv_to_list(os.path.join(cols_path, "explode_cols.csv"), 'explode')

# --------------- GENERAL SURVEY DATA CLEANING --------------- #

# Clean column names - strip whitespace, collapse multiple spaces, remove non-breaking spaces
df_s.columns = [clean_colname(c) for c in df_s.columns]
# Drop any columns where all values are NAs
df_s = df_s.dropna(axis=1, how='all').copy()
# Replace "Above 60000" in income column with the number 60000 so we can treat it as numeric.
df_s['What is the monthly total income of your household (KES)?'] = df_s['What is the monthly total income of your household (KES)?'].replace('Above 60000', 60000)
# Add number of males and number of females columns to get total household size column.
df_s["Total household size"] = df_s["Number of females in household"] + df_s["Number of males in household"]

# Clean up multi-label rows into pythonic lists
for col in multi_label_cols:
    # Turn comma-separated lists in cells into pythonic lists
    df_s[col] = df_s[col].apply(safe_list_parser)
    # Replace NaN or float with empty list to avoid TypeError
    df_s[col] = df_s[col].apply(lambda x: x if isinstance(x, (list, set, tuple)) else [])

# --------- EXPLODE COMMUNITY SERVICE COLUMNS IN SURVEY ---------- #

# Get the max length across all list-y community service rows
max_len = df_s['Community service'].apply(lambda x: len(x) if isinstance(x, list) else 0).max()

# Create exploded columns for multiple community service/ease answers - turn into categorical columns
for i in range(max_len):
    for c in explode_cols:
        # Explode the column
        df_s[f'{c}_{i+1}'] = df_s[f'{c}'].apply(lambda x: x[i] if isinstance(x, list) and i < len(x) else None)
        # Add the columns to the categorical list, since they act like categorical data now
        categorical_cols.extend([f'{c}_{i+1}'])

# Make annotations a pythonic list instead of separated by semicolons
df_u.loc[:, 'Annotations'] = df_u['Annotations'].str.split(';')

# ----------- MERGE DATA FROM UTTERANCES TO SURVEY ------------- #

# Get the appliances that people selected as what they valued most from the utterances (was omitted from survey file)
df_u_most = items_from_paragraph_number(df_u, 8, 'Of these appliances which is the most valuable?')
# Merge this back onto the survey dataset
df_s = df_s.merge(df_u_most, on='Interview ID', how='left')
# Add this to the categorical columns list
categorical_cols.extend(['Of these appliances which is the most valuable?'])

# Get the appliances that people selected as what they valued least from the utterances (was omitted from survey file)
df_u_least = items_from_paragraph_number(df_u, 9, 'Of these appliances which is the least valuable?')
# Merge this back onto the survey dataset
df_s = df_s.merge(df_u_least, on='Interview ID', how='left')
# Add this to the categorical columns list
categorical_cols.extend(['Of these appliances which is the least valuable?'])

# --------- MAKE BINARY COLUMNS FOR MULTI-SELECT COLUMNS --------- #

binary_cols = {}

# For each multi-label column 
for col in multi_label_cols:
    # Get the unique responses in that column
    items = unique_items = (df_s[col].explode().dropna().unique().tolist())
    # For each unique response
    for item in items:
        # Make a binary column 
        binary_cols[f"{col}_{item}"] = df_s[col].apply(lambda lst: 1 if item in lst else 0)

# Turn dict of columns into a DataFrame, join all at once to avoid fragmentation
binary_df = pd.DataFrame(binary_cols)
df_s = pd.concat([df_s, binary_df], axis=1)

# --------- MAKE BINARY COLUMNS CATEGORICAL COLUMNS ------- #

binary_frames = []

# For each categorical column
for c in categorical_cols:
    # Make dummy variable (one-hot)
    dummies = pd.get_dummies(df_s[c], prefix=c, dtype=int)
    # Save to binary_frames
    binary_frames.append(dummies)

# Concatenate all dummy columns at once
binary_df = pd.concat(binary_frames, axis=1)
df_s = pd.concat([df_s, binary_df], axis=1)

# --------- CREATE AND MERGE ANNOTATION ONE-HOT FEATURES FROM UTTERANCES --------- #

# Ensure Annotations are lists (defensive)
df_u['Annotations'] = df_u['Annotations'].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x]))

# Explode annotations, drop NAs, stringify and strip whitespace, remove empties
annotations_exploded = (df_u[['Interview ID', 'Annotations']].explode('Annotations').dropna(subset=['Annotations']))
annotations_exploded['Annotations'] = annotations_exploded['Annotations'].astype(str).str.strip()
annotations_exploded = annotations_exploded[annotations_exploded['Annotations'] != ""]

# Create one-hot encoded columns for the annotation values
dummies = annotations_exploded['Annotations'].str.get_dummies()
# Join the dummy matrix back to the exploded DataFrame, so each exploded row keeps its Interview ID
annotations_wth_dummies = annotations_exploded.join(dummies)
# Group by Interview ID and aggregate. Max gives 1 if annotation appears anywhere for this interview
annotation_dummies_by_interview = (annotations_wth_dummies.groupby('Interview ID')[dummies.columns].max())
# Prefix dummy columns with "annotation_"
annotation_dummies_by_interview = annotation_dummies_by_interview.add_prefix("annotation_")

# Merge into df_s
df_s = df_s.merge(annotation_dummies_by_interview.reset_index(), on='Interview ID', how='left')

# Fill NAs with 0 and cast to int
annotation_cols = [c for c in df_s.columns if c.startswith('annotation_')]
df_s[annotation_cols] = df_s[annotation_cols].fillna(0).astype(int)

# ---------- DROPS & SAVES --------- #

# Save original with all cols
df_s.to_csv(os.path.join(data_path, 'Kenya_UPV_Survey_Preprocessed_AllCols.csv'), index=False)

# Drop all the old cols which have been encoded
df_s = df_s.drop(columns=explode_cols+categorical_cols+text_cols+multi_label_cols+meta_cols)

# Save this version
df_s.to_csv(os.path.join(data_path, 'Kenya_UPV_Survey_Preprocessed_EncodedCols.csv'), index=False)
