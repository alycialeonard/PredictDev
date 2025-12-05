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
import matplotlib.pyplot as plt
import os
import seaborn as sns
import shap
import joblib
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import json
from datetime import datetime
import sys


# --------- DATA PROCESSING HELPERS ------- #

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

#--------- PLOTTING FUNCTIONS ------- #

# Plot ROC & AUC (works if classifier exposes predict_proba)
# Takes model as sklearn pipeline, y_test, X_test, path to save in.
def plot_roc_auc(model, X_test, y_test, save_path):
    # Binarize labels for multiclass one-vs-rest
    classes = np.unique(y_test)
    n_classes = len(classes)
    try:
        y_score = model.predict_proba(X_test)  # shape (n_samples, n_classes)
        # If sklearn returns list for binary classifiers, convert
        if isinstance(y_score, list):
            # list of arrays for each class -> stack
            y_score = np.vstack([col[:, 1] if col.ndim == 2 else col for col in y_score]).T

        # binarize y_test
        y_test_bin = label_binarize(y_test, classes=classes)
        if y_test_bin.shape[1] == 1:
            # binary label_binarize may return shape (n_samples,1) -> squeeze to (n_samples,)
            y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

        # ROC per class
        fpr = dict();
        tpr = dict();
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # micro-average
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # macro-average AUC (interpolate)
        # first aggregate all fpr points
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.figure(figsize=(8, 6))
        plt.plot(fpr["micro"], tpr["micro"], label=f'micro (AUC = {roc_auc["micro"]:.2f})', linestyle=':', linewidth=2)
        plt.plot(fpr["macro"], tpr["macro"], label=f'macro (AUC = {roc_auc["macro"]:.2f})', linestyle='-.', linewidth=2)
        for i, cls in enumerate(classes):
            plt.plot(fpr[i], tpr[i], lw=1, label=f'{cls} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curves (one-vs-rest)')
        plt.legend(loc='lower right', fontsize='small')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("ROC plot skipped (predict_proba not available or error):", e)


# Plot precision-recall curve
def plot_precision_recall_curve(model, X_test, y_test, save_path):
    classes = np.unique(y_test)
    try:
        y_score = model.predict_proba(X_test)  # shape (n_samples, n_classes)
        y_test_bin = label_binarize(y_test, classes=classes)
        if y_test_bin.shape[1] == 1:
            # binary label_binarize may return shape (n_samples,1) -> squeeze to (n_samples,)
            y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
        plt.figure(figsize=(8, 6))
        ap = dict()
        for i, cls in enumerate(classes):
            precision_i, recall_i, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            ap[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])
            plt.plot(recall_i, precision_i, lw=1, label=f'{cls} (AP={ap[i]:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curves (one-vs-rest)')
        plt.legend(loc='lower left', fontsize='small')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("Precision-Recall plot skipped (predict_proba not available or error):", e)


# Plot confusion matrices (using counts)
def plot_confusion_matrix_count(model, X_test, y_test, save_path):
    y_pred_test = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test, labels=np.unique(y_test))
    cm_index = np.unique(y_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=cm_index, yticklabels=cm_index, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix (counts)')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# Plot confusion matrix (normalized by positive row)
def plot_confusion_matrix_norm(model, X_test, y_test, save_path):
    y_pred_test = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test, labels=np.unique(y_test))
    cm_index = np.unique(y_test)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=cm_index, yticklabels=cm_index, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix (normalsed by true row)')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# Plot precision, recall, and f1 per class.
def plot_precision_recall_f1(model, X_test, y_test, save_path):
    y_pred_test = model.predict(X_test)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_test, zero_division=0)
    metrics_df = pd.DataFrame({'class': np.unique(y_test), 'precision': precision, 'recall': recall, 'f1': f1, 'support': support}).set_index('class')
    ax = metrics_df[['precision', 'recall', 'f1']].plot(kind='bar', figsize=(10, 6))
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    plt.title('Precision / Recall / F1 per class')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# Plot vanilla feature importance
def plot_feature_importance(model, save_path):
    # Extract feature importances and feature names from best model
    preprocessed_features = (model.named_steps['preprocessing'].get_feature_names_out())
    importances = model.named_steps['classifier'].feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': preprocessed_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df.head(10))
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()