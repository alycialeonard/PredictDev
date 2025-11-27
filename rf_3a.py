#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: plots.py
Author: Alycia Leonard
Date: 2025-11-26
Version: 1.0
Description: rf modelling script for UPV dataset - Experiment 3a
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

# EXPERIMENT 3a: Predict items given demographic, geographic, and gold value data.
# To run and record: python -u rf_3a.py 2>&1 | tee "results/rf_3a/annotation_multipurpose/output/rf_3a_$(date +"%Y%m%d_%H%M%S").txt"

# --------- DEFINE EXPERIMENT PARAMETERS ---------- #

# Define target for prediction.
target = 'annotation_Multipurpose'

# Define short-form of target to use in file saving
target_short = 'annotation_multipurpose'

# Define questions to drop from predictors
stems_to_drop = ['annotation_']
#['Which 5 items are most important to you in your daily life? Please indicate these in order of importance, starting with the most important',
#'Given the chosen climate event - which 3 items are most useful to you?']

# -------- DEFINE SAVE PATHS ------------#
cwd = os.getcwd()
model_save_path = os.path.join(cwd, 'results', 'rf_3a', target_short, 'models', 'rf_3a.pkl')
plots_save_path = os.path.join(cwd, 'results', 'rf_3a', target_short, 'plots' )

# ------------ LOAD DATA ------------ #

# Get current working directory, data path, load dataset and numeric columns list for preprocessor
data_path = os.path.join(cwd, 'data')
df = pd.read_csv(os.path.join(data_path, "Kenya_UPV_Survey_Preprocessed_EncodedCols.csv"), low_memory=False)
num_cols = csv_to_list(os.path.join(data_path, "cols", "numeric_cols.csv"), 'numeric_cols')

# --------- PREPARE DATA ------------- #

# Drop rows where target is missing & set target to y
df = df.dropna(subset=[target]).copy()
y = df[target]

# Set predictors as X
cols_to_drop = [c for c in df.columns if any(c.startswith(stem) for stem in stems_to_drop)]
X = df.drop(columns=[target] + cols_to_drop)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# -------- DEFINE CLASSIFICATION PIPELINE -------- #

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

# Define Random Forest classifier (without parameters)
clf = RandomForestClassifier(random_state=42)

# Use full pipeline with preprocessing inside
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
print("Running randomized search...")
random_search.fit(X_train, y_train)
print("- Best parameters:", random_search.best_params_)
print("- Best CV weighted F1:", random_search.best_score_)

# Get the best model pipeline from the search
best_model = random_search.best_estimator_
# Save the best model
joblib.dump(best_model, model_save_path)

# ------- EVALUATE MODEL & PLOT PERFORMANCE -------- #

# Evaluate the best model
test_score = best_model.score(X_test, y_test)
print(f"Test accuracy with best params: {test_score:.3f}")
train_score = best_model.score(X_train, y_train)
print(f"Train accuracy with best params: {train_score:.3f}")
y_pred_test = best_model.predict(X_test)
print("Classification report:\n")
print(classification_report(y_test, y_pred_test, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))

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

# ------- FEATURE IMPORTANCE --------- #

# Extract & plot generic feature importance
plot_feature_importance(best_model, os.path.join(plots_save_path, "feature_importance.png"))

# ---------- SHAP ----------- #

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
print(f"Top {top_n} features by mean |SHAP|:")
for r in range(top_n):
    j = top_idx[r]
    print(f" {r+1:02d}. {X_shap.columns[j]}  mean|shap|={mean_abs_shap[j]:.6e}")

# Plot summary dot plot, bar plot, dependence for top feature
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_for_plot, X_shap, max_display=20, show=False)
plt.title(f"SHAP Summary (positive class)")
plt.savefig(os.path.join(plots_save_path, "SHAP_summary_dot.png"), bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_for_plot, X_shap, plot_type='bar', max_display=20, show=False)
plt.title(f"SHAP Feature Importance (bar) — positive class")
plt.savefig(os.path.join(plots_save_path, "SHAP_summary_bar.png"), bbox_inches='tight')
plt.close()

if mean_abs_shap.size > 0:
    top_feature = X_shap.columns[top_idx[0]]
    plt.figure(figsize=(8, 6))
    try:
        shap.dependence_plot(top_feature, shap_for_plot, X_shap, show=False)
        plt.title(f"SHAP Dependence for {top_feature}")
        plt.savefig(os.path.join(plots_save_path, f"SHAP_dependence_top_feature.png"), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("Dependence plot error:", e)

