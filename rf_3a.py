#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: rf_3a.py
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
from functions import csv_to_list
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import numpy as np
from scipy.stats import randint

# EXPERIMENT 3a: Predict items given demographic, geographic, and gold value data.

# Get current working directory, data path
cwd = os.getcwd()
data_path = os.path.join(cwd, 'data')
# Load dataset
df = pd.read_csv(os.path.join(data_path, "Kenya_UPV_Survey_Preprocessed_EncodedCols.csv"), low_memory=False)
# Load numeric column list (for preprocessor)
num_cols = csv_to_list(os.path.join(data_path, "cols", "numeric_cols.csv"), 'numeric_cols')

# Define target for prediction. Starting with electricity - important to see if it can generally be predicted who will value electric service
target = 'Which 5 items are most important to you in your daily life? Please indicate these in order of importance, starting with the most important_Electricity'
# Drop rows where target is missing
df = df.dropna(subset=[target]).copy()
# Define target (y)
y = df[target]

# Define column stems for variables to drop from predictors
stems_to_drop = ['Which 5 items are most important to you in your daily life? Please indicate these in order of importance, starting with the most important',
                 'Given the chosen climate event - which 3 items are most useful to you?']
cols_to_drop = [c for c in df.columns if c.startswith(stems_to_drop[0]) or c.startswith(stems_to_drop[1])]
# Define predictors (X)
X = df.drop(columns=[target] + cols_to_drop)

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols
        ),
    ],
    remainder='passthrough'  # already processed multi-labels pass through
)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ------- HYPERPARAMETER SEARCH ------- #

# Define Random Forest classifier (without parameters)
clf = RandomForestClassifier(random_state=42)

# Use full pipeline with preprocessing inside
clf_pipeline = Pipeline([('preprocessing', preprocessor), ('classifier', clf)])

param_dist = {
    'classifier__n_estimators': randint(200, 801),      # sample between 200 and 800
    'classifier__max_depth': [None, 8, 12, 16, 20],
    'classifier__min_samples_leaf': randint(1, 6),     # 1..5
    'classifier__min_samples_split': randint(2, 11),   # 2..10
    'classifier__max_features': ['sqrt', 'log2', 0.2, 0.5],
    'classifier__class_weight': ['balanced', 'balanced_subsample']
}

random_search = RandomizedSearchCV(
    estimator=clf_pipeline,
    param_distributions=param_dist,
    n_iter=100,               # try 100 random combos
    scoring='f1_weighted',
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42,
    refit=True
)

print("Running randomized search...")
random_search.fit(X_train, y_train)
print("- Best parameters:", random_search.best_params_)
print("- Best CV weighted F1:", random_search.best_score_)

# Get the best model (pipeline) and evaluate it on the test set
best_model = random_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test accuracy with best params: {test_score:.3f}")
train_score = best_model.score(X_train, y_train)
print(f"Train accuracy with best params: {train_score:.3f}")
y_pred_test = best_model.predict(X_test)
print("Classification report:")
print(classification_report(y_test, y_pred_test, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))

# Save best model
joblib.dump(best_model, os.path.join(cwd, 'models', 'rf_3a.pkl'))

# Extract feature importances and feature names from best model
preprocessed_features = (best_model.named_steps['preprocessing'].get_feature_names_out())
importances = best_model.named_steps['classifier'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': preprocessed_features,'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df.head(10))
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(cwd, 'plots', 'rf_3a', "feature_importance.png"), bbox_inches='tight')
plt.close()

# Get preprocessor and classifier from best model
preprocessor = best_model.named_steps['preprocessing']
clf = best_model.named_steps['classifier']

# Transform training data (note: use X_train not X_test for the background)
X_train_transformed = preprocessor.transform(X_train)

# Get feature names
feature_names = preprocessor.get_feature_names_out(X_train.columns)

# Make dataframe for SHAP
X_train_pre = pd.DataFrame(X_train_transformed, columns=feature_names)

# Use TreeExplainer on the underlying classifier and compute SHAP values
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_train_pre)

# Normalize to a 2D array shap_for_plot with shape (n_samples, n_features) - can be organised multiple ways
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

print("SHAP array format:", chosen)

# Top features by mean |SHAP|
mean_abs_shap = np.nanmean(np.abs(shap_for_plot), axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1]
top_n = min(20, len(top_idx))
print(f"Top {top_n} features by mean |SHAP|:")
for r in range(top_n):
    j = top_idx[r]
    print(f" {r+1:02d}. {X_train_pre.columns[j]}  mean|shap|={mean_abs_shap[j]:.6e}")

# Plot summary dot plot, bar plot, dependence for top feature
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_for_plot, X_train_pre, max_display=20, show=False)
plt.title(f"SHAP Summary (positive class)")
plt.savefig(os.path.join(cwd, 'plots', 'rf_3a', "SHAP_summary_dot.png"), bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_for_plot, X_train_pre, plot_type='bar', max_display=20, show=False)
plt.title(f"SHAP Feature Importance (bar) — positive class")
plt.savefig(os.path.join(cwd, 'plots', 'rf_3a', "SHAP_summary_bar.png"), bbox_inches='tight')
plt.close()

if mean_abs_shap.size > 0:
    top_feature = X_train_pre.columns[top_idx[0]]
    plt.figure(figsize=(8, 6))
    try:
        shap.dependence_plot(top_feature, shap_for_plot, X_train_pre, show=False)
        plt.title(f"SHAP Dependence for {top_feature}")
        plt.savefig(os.path.join(cwd, 'plots', 'rf_3a', f"SHAP_dependence_top_feature.png"), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("Dependence plot error:", e)


#print("shap_for_plot.shape:", getattr(shap_for_plot, 'shape', None))

# Expected value handling (explainer.expected_value can be scalar or array/list)
# exp_val = getattr(explainer, "expected_value", None)
# if exp_val is not None:
#     try:
#         if hasattr(exp_val, "__len__") and len(exp_val) >= 2:
#             base_value = exp_val[1]  # expected value for positive class
#         else:
#             base_value = exp_val
#     except Exception:
#         base_value = exp_val
# else:
#     base_value = None
# print("explainer.expected_value (chosen):", base_value)

