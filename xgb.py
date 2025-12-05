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


from xgboost import XGBClassifier
from scipy.stats import randint
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shap
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from functions import csv_to_list, plot_confusion_matrix_count, plot_confusion_matrix_norm, plot_precision_recall_f1, plot_roc_auc, plot_precision_recall_curve, plot_feature_importance
import json
from datetime import datetime
import sys

# Suppress tight layout warnings (inevitable due to SHAP code)
warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")

# ---------- EXPERIMENT FUNCTION -------- #

def run_xgb_experiment(target, target_short, stems_to_drop, clf, clf_short, param_dist):

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

    # Second split (train/validation)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train,stratify=y_train,test_size=0.15,random_state=42)

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
    # Define random search to execute: 100 random combinations, 5-fold validation
    pos_f1_scorer = make_scorer(f1_score, pos_label=1)
    # compute class imbalance weight & append to param distribution
    scale_pos = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    param_dist['classifier__scale_pos_weight'] = [scale_pos, scale_pos * 0.5, scale_pos * 2]

    random_search = RandomizedSearchCV(
        estimator=clf_pipeline,
        param_distributions=param_dist,
        n_iter=200,
        scoring= pos_f1_scorer, # Previously: 'f1_weighted',
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        refit=True
    )

    # Run the search
    print("Running randomized hyperparameter search...")
    random_search.fit(X_tr, y_tr)
    print("- Best parameters:", random_search.best_params_)
    print("- Best CV weighted F1:", random_search.best_score_)
    # Get the best estimator
    best_model = random_search.best_estimator_

    print("Final refit with early stopping... ")
    # Now create transformed X_tr/T_val for final refit with early stopping
    preproc = best_model.named_steps['preprocessing']
    clf = best_model.named_steps['classifier']
    X_tr_trans = preproc.transform(X_tr)
    X_val_trans = preproc.transform(X_val)  # transform the validation set too

    # Refit classifier with early stopping (directly on the XGB object)
    clf.set_params(verbosity=0)
    clf.fit(X_tr_trans, y_tr, eval_set=[(X_val_trans, y_val)],early_stopping_rounds=50, verbose=False)

    # Put classifier back into the pipeline if you want to save it
    best_model.named_steps['classifier'] = clf
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
    plot_confusion_matrix_count(best_model, X_test, y_test, os.path.join(plots_save_path, 'confusion_matrix_counts.png'))
    plot_confusion_matrix_norm(best_model, X_test, y_test, os.path.join(plots_save_path, 'confusion_matrix_normalized.png'))
    plot_precision_recall_f1(best_model, X_test, y_test, os.path.join(plots_save_path, 'precision_recall_f1_per_class.png'))
    plot_roc_auc(best_model, X_test, y_test, os.path.join(plots_save_path, 'roc_curves.png'))
    plot_precision_recall_curve(best_model, X_test, y_test, os.path.join(plots_save_path, 'precision_recall_curves.png'))
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

# --------- RUN THE EXPERIMENT ---------- #
def main():

    # Define target for prediction.
    tar = 'Which 5 items are most important to you in your daily life? Please indicate these in order of importance, starting with the most important_Electricity'

    # Define short-form of target to use in file saving
    tar_short = 'UPV_Electricity'

    # Define questions to drop from predictors
    to_drop = ['Which of the following items do you have access to in your daily life?',
               'Which 5 items are most important to you in your daily life? Please indicate these in order of importance, starting with the most important',
               'Given the chosen climate event - which 3 items are most useful to you?']

    # Define XGBoost classifier (without tuned params)
    clf = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)

    clf_short = 'xgb'

    # Define param_dist to search (for hyperparameters)
    param_dist = {
        'classifier__n_estimators': randint(100, 1001),
        'classifier__max_depth': randint(3, 4, 5),
        'classifier__learning_rate': [0.01, 0.03, 0.05, 0.1],
        'classifier__subsample': [0.6, 0.7, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.7, 0.8, 1.0],
        'classifier__gamma': [0, 0.1, 0.5, 1.0],
        'classifier__min_child_weight': [5, 10, 20],
        'classifier__reg_alpha': [0, 0.01, 0.1, 0.5],
        'classifier__reg_lambda': [0.5, 1, 5, 10]
    }

    # Run the experiment
    run_xgb_experiment(tar, tar_short, to_drop, clf, clf_short, param_dist)

if __name__ == "__main__":
    main()