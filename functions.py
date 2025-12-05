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
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, precision_recall_fscore_support
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

# ---------- EXPERIMENT FUNCTION -------- #

def run_experiment(target, target_short, stems_to_drop, clf, clf_short, param_dist):

    # -------- DEFINE EXPERIMENT PATH & START LOGGING ---------- #
    print(f"Starting experiment: Predicting {target_short} with {clf_short}!")
    cwd = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = os.path.join(cwd, 'results', clf_short, target_short, timestamp)
    os.makedirs(experiment_path, exist_ok=True)
    log_file = os.path.join(experiment_path, "console_output.txt")
    print(f"Console output will log to {log_file}.")
    log_f = open(log_file, "w")
    sys.stdout = log_f
    sys.stderr = log_f

    # -------- DEFINE SAVE PATHS ------------#
    model_save_path = os.path.join(experiment_path, 'models')
    os.makedirs(model_save_path, exist_ok=True)
    plots_save_path = os.path.join(experiment_path, 'plots' )
    os.makedirs(plots_save_path, exist_ok=True)
    metrics_save_path = os.path.join(experiment_path, 'metrics')
    os.makedirs(metrics_save_path, exist_ok=True)

    # ----- PRINT EXPERIMENT SETTINGS TO LOG FILE ----- #
    print("Experiment settings:")
    print(f"- Target: {target}")
    print(f"- Target short name: {target_short}")
    print(f"- Question stems being dropped: {stems_to_drop}\n")

    # ------------ LOAD DATA ------------ #
    print("Experiment execution:\nLoading data...")
    data_path = os.path.join(cwd, 'data')
    df = pd.read_csv(os.path.join(data_path, "Kenya_UPV_Survey_Preprocessed_EncodedCols.csv"), low_memory=False)
    print(f"Data loaded from {os.path.join(data_path, "Kenya_UPV_Survey_Preprocessed_EncodedCols.csv")}!")
    num_cols = csv_to_list(os.path.join(data_path, "cols", "numeric_cols.csv"), 'numeric_cols')

    # --------- PREPARE DATA ------------- #
    print("Preparing data...")
    # Drop rows where target is missing & set target to y
    df = df.dropna(subset=[target]).copy()
    y = df[target]
    # Set predictors as X
    cols_to_drop = [c for c in df.columns if any(c.startswith(stem) for stem in stems_to_drop)]
    X = df.drop(columns=[target] + cols_to_drop)
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # -------- DEFINE CLASSIFICATION PIPELINE -------- #
    print("Defining classification pipeline & hyperparameter search space...")
    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols
            ),
        ],
        remainder='passthrough'  # already processed pass through
    )

    # Define Random Forest classifier pipeline
    clf_pipeline = Pipeline([('preprocessing', preprocessor), ('classifier', clf)])

    # ------- HYPERPARAMETER SEARCH ------- #
    # Define random search to execute: 100 random combinations, 5-fold validation, weighted f1
    random_search = RandomizedSearchCV(
        estimator=clf_pipeline,
        param_distributions=param_dist,
        n_iter=100,
        scoring='f1_weighted',
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        refit=True
    )
    # Run the search
    print("Running randomized hyperparameter search...")
    random_search.fit(X_train, y_train)
    print("- Best parameters:", random_search.best_params_)
    print("- Best CV weighted F1:", random_search.best_score_)
    # Get the best model pipeline from the search
    best_model = random_search.best_estimator_
    # Save the best model
    joblib.dump(best_model, os.path.join(model_save_path, 'rf.pkl'))

    # ------- EVALUATE MODEL -------- #
    # Evaluate the best model
    print("Performance of best model:")
    test_score = best_model.score(X_test, y_test)
    print(f"- Test accuracy with best params: {test_score:.3f}")
    train_score = best_model.score(X_train, y_train)
    print(f"- Train accuracy with best params: {train_score:.3f}")
    y_pred_test = best_model.predict(X_test)
    print("- Classification report:\n")
    print(classification_report(y_test, y_pred_test, zero_division=0))
    print("- Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))

    # ----------- SAVE METRICS -------- #
    print("Saving metrics...")
    # Convert classification report to dataframe & save as CSV
    report_dict = classification_report(y_test, y_pred_test, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(metrics_save_path, 'classification_report.csv'), index=True)

    # Convert confusion matrix to dataframe & save as CSV
    cm = confusion_matrix(y_test, y_pred_test)
    cm_df = pd.DataFrame(cm, index=best_model.classes_, columns=best_model.classes_)
    cm_df.to_csv(os.path.join(metrics_save_path, 'confusion_matrix.csv'))

    # Create summary CSV for experiment tracking
    summary = {
        'target': target,
        'target_short': target_short,
        'train_accuracy': float(train_score),
        'test_accuracy': float(test_score),
        'best_cv_weighted_f1': float(random_search.best_score_),
        'best_params': json.dumps(random_search.best_params_, default=str),
        'n_train': int(X_train.shape[0]),
        'n_test': int(X_test.shape[0]),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(metrics_save_path, 'metrics_summary.csv'), index=False)
    print(f"Performance metrics saved to {metrics_save_path}!")

    # -------- PLOT PERFORMANCE -------- #

    print("Plotting performance...")
    # Plot confusion matrix (counts)
    plot_confusion_matrix_count(best_model, X_test, y_test, os.path.join(plots_save_path, 'confusion_matrix_counts.png'))
    # Plot confusion matrix (normalised by true row)
    plot_confusion_matrix_norm(best_model, X_test, y_test, os.path.join(plots_save_path, 'confusion_matrix_normalized.png'))
    # Bar plot of precision / recall / f1 per class
    plot_precision_recall_f1(best_model, X_test, y_test, os.path.join(plots_save_path, 'precision_recall_f1_per_class.png'))
    # Plot ROC & AUC
    plot_roc_auc(best_model, X_test, y_test, os.path.join(plots_save_path, 'roc_curves.png'))
    # Plot precision-recall curve
    plot_precision_recall_curve(best_model, X_test, y_test, os.path.join(plots_save_path, 'precision_recall_curves.png'))
    # Extract & plot generic feature importance
    plot_feature_importance(best_model, os.path.join(plots_save_path, "feature_importance.png"))
    print(f"Performance plots saved to {plots_save_path}!")

    # ---------- GET SHAP VALUES ----------- #
    print("Getting SHAP values...")
    # Get preprocessor and classifier from best model
    preprocessor = best_model.named_steps['preprocessing']
    clf = best_model.named_steps['classifier']

    # Transform training data for SHAP
    X_train_transformed = preprocessor.transform(X_train)
    feature_names = preprocessor.get_feature_names_out(X_train.columns)
    X_shap = pd.DataFrame(X_train_transformed, columns=feature_names)
    n_samples = X_shap.shape[0]
    n_features = X_shap.shape[1]
    print(f"X_shap shape: samples={n_samples}, features={n_features}")

    # Use TreeExplainer on the classifier and compute SHAP values
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_shap)
    # shap_for_plot = shap_values[..., 1]  # positive class

    # Hand the fact that SHAP can output about a million different shapes.
    print("Raw SHAP type:", type(shap_values))
    # Case 1: One list per class
    if isinstance(shap_values, list):
        shap_for_plot = shap_values[1]  # positive class
    # Case 2: 3D array: (samples, features, classes)
    elif shap_values.ndim == 3 and shap_values.shape[2] >= 2:
        shap_for_plot = shap_values[:, :, 1]
    # Case 3: 2D array matching (samples, features)
    elif shap_values.ndim == 2 and shap_values.shape[1] > 1:
        shap_for_plot = shap_values
    # Case 4: SHAP returned only one value per sample (shape: (n,1)). Fill it across all features so code doesn't crash
    elif shap_values.ndim == 2 and shap_values.shape[1] == 1:
        shap_for_plot = np.repeat(shap_values, X_shap.shape[1], axis=1)
    else:
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

    # Get per-sample SHAP values (positive class) as DataFrame, add sample identifiers, save to CSV
    shap_df = pd.DataFrame(shap_for_plot, columns=X_shap.columns, index=X_shap.index)
    shap_df.insert(0, 'sample_index', shap_df.index)
    shap_df.to_csv(os.path.join(metrics_save_path, 'shap_values_per_sample.csv'), index=False)
    print(f"SHAP per sample saved to {os.path.join(metrics_save_path, 'shap_values_per_sample.csv')}!")

    # Get feature-level summary (mean absolute SHAP, std, rank) & save to csv
    mean_abs = np.nanmean(np.abs(shap_for_plot), axis=0)
    std_abs  = np.nanstd(np.abs(shap_for_plot), axis=0)
    feat_summary = pd.DataFrame({'feature': X_shap.columns, 'mean_abs_shap': mean_abs, 'std_abs_shap': std_abs})
    feat_summary = feat_summary.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    feat_summary['rank'] = feat_summary['mean_abs_shap'].rank(ascending=False, method='dense').astype(int)
    feat_summary.to_csv(os.path.join(metrics_save_path, 'shap_feature_summary.csv'), index=False)
    print(f"SHAP summary saved to {os.path.join(metrics_save_path, 'shap_feature_summary.csv')}!")

    print("Plotting SHAP values...")
    # Plot summary dot plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_for_plot, X_shap, max_display=20, show=False)
    plt.title(f"SHAP Summary (positive class)")
    plt.savefig(os.path.join(plots_save_path, "SHAP_summary_dot.png"), bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to {os.path.join(plots_save_path, "SHAP_summary_dot.png")}!")

    # ------- STOP LOGGING ------- #
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_f.close()
    print(f"Console log saved to {log_file}")