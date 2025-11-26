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

