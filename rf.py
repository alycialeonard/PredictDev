#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: rf.py
Author: Alycia Leonard
Date: 2025-12-05
Version: 1.0
Description: rf modelling script for UPV dataset - run a single experiment
License: GNU GPL-3.0
Contact: alycia.leonard@eng.ox.ac.uk
"""

import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from functions import (csv_to_list, plot_roc_auc, plot_precision_recall_curve, plot_confusion_matrix_count,
                       plot_confusion_matrix_norm, plot_precision_recall_f1, plot_feature_importance)
import matplotlib.pyplot as plt
import joblib
import shap
import numpy as np
from scipy.stats import randint
import json
from datetime import datetime
import warnings
import sys

# Suppress tight layout warnings (inevitable due to SHAP code)
warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")


# --------- FUNCTION TO RUN A RANDOM FOREST EXPERIMENT -------- #
def rf_experiment(target, target_short, stems_to_drop):

    # -------- DEFINE EXPERIMENT PATH & START LOGGING ---------- #
    print("Starting experiment!")
    cwd = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = os.path.join(cwd, 'results', 'rf', target_short, timestamp)
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

    # Define Random Forest classifier (without parameters), pipeline
    clf = RandomForestClassifier(random_state=42)
    clf_pipeline = Pipeline([('preprocessing', preprocessor), ('classifier', clf)])

    # ------- HYPERPARAMETER SEARCH ------- #
    # Define hyperparameter distribution to search
    param_dist = {
        'classifier__n_estimators': randint(200, 801),
        'classifier__max_depth': [None, 8, 12, 16, 20],
        'classifier__min_samples_leaf': randint(1, 6),
        'classifier__min_samples_split': randint(2, 11),
        'classifier__max_features': ['sqrt', 'log2', 0.2, 0.5],
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }
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
    # Use TreeExplainer on the classifier and compute SHAP values
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_shap)
    shap_for_plot = shap_values[..., 1]  # positive class
    # Get top 20 features by mean abs val of shap
    mean_abs_shap = np.nanmean(np.abs(shap_for_plot), axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1]
    top_n = min(20, len(top_idx))

    # ---------- SAVE SHAP VALUES --------- #
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

    # --------- PLOT SHAP VALUES ---------- #
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



# --------- RUN THE EXPERIMENT ---------- #
def main():
    # Define target for prediction.
    tar = 'Which of the following items do you have access to in your daily life?_Motorcycle'
    # Define short-form of target to use in file saving
    tar_short = 'access_motorcycle'
    # Define questions to drop from predictors
    to_drop = ['Which of the following items do you have access to in your daily life?',
               'Which 5 items are most important to you in your daily life? Please indicate these in order of importance, starting with the most important',
               'Given the chosen climate event - which 3 items are most useful to you?']
    # Run the experiment
    rf_experiment(tar, tar_short, to_drop)


if __name__ == "__main__":
    main()