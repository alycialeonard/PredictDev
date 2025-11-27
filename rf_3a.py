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
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from functions import csv_to_list
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import numpy as np
from scipy.stats import randint

# EXPERIMENT 3a: Predict items given demographic, geographic, and gold value data.
# To run + log: python -u plots.py 2>&1 | tee "results/rf_3a/electricity/output/rf_3a_$(date +"%Y%m%d_%H%M%S").txt"

# --------- DEFINE EXPERIMENT PARAMETERS ---------- #

# Define target for prediction.
# Starting with electricity - important to see if it can generally be predicted who will value electric service
target = 'Which 5 items are most important to you in your daily life? Please indicate these in order of importance, starting with the most important_Electricity'

# Define questions to drop from predictors
# Dropping other UPV responses (general and climate)
stems_to_drop = ['Which 5 items are most important to you in your daily life? Please indicate these in order of importance, starting with the most important',
                 'Given the chosen climate event - which 3 items are most useful to you?']

# Define paths to save results
cwd = os.getcwd()
model_save_path = os.path.join('results', 'rf_3a', 'electricity', 'models', 'rf_3a.pkl')
plots_save_path = os.path.join(cwd, 'results', 'rf_3a', 'electricity', 'plots')

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
cols_to_drop = [c for c in df.columns if c.startswith(stems_to_drop[0]) or c.startswith(stems_to_drop[1])]
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
    remainder='passthrough'  # already processed multi-labels pass through
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

# ------- EVALUATE MODEL -------- #

# Evaluate the best model
test_score = best_model.score(X_test, y_test)
print(f"Test accuracy with best params: {test_score:.3f}")
train_score = best_model.score(X_train, y_train)
print(f"Train accuracy with best params: {train_score:.3f}")
y_pred_test = best_model.predict(X_test)
print("Classification report:\n")
print(classification_report(y_test, y_pred_test, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))

# ------- PLOT PERFORMANCE ------- #

# Plot confusion matrix heatmap (counts)
cm = confusion_matrix(y_test, y_pred_test, labels=np.unique(y_test))
cm_index = np.unique(y_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=cm_index, yticklabels=cm_index, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion matrix (counts)')
plt.tight_layout()
plt.savefig(os.path.join(plots_save_path, 'confusion_matrix_counts.png'), bbox_inches='tight')
plt.close()

# Plot normalised confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=cm_index, yticklabels=cm_index, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion matrix (normalized by true row)')
plt.tight_layout()
plt.savefig(os.path.join(plots_save_path, 'confusion_matrix_normalized.png'), bbox_inches='tight')
plt.close()

# Bar plot of precision / recall / f1 per class
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_test, zero_division=0)
metrics_df = pd.DataFrame({'class': np.unique(y_test), 'precision': precision, 'recall': recall, 'f1': f1, 'support': support}).set_index('class')
ax = metrics_df[['precision', 'recall', 'f1']].plot(kind='bar', figsize=(10, 6))
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
plt.title('Precision / Recall / F1 per class')
plt.xticks(rotation=45, ha='right')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(plots_save_path, 'precision_recall_f1_per_class.png'), bbox_inches='tight')
plt.close()

# ROC & AUC (works if classifier exposes predict_proba)
# Binarize labels for multiclass one-vs-rest
classes = np.unique(y_test)
n_classes = len(classes)
try:
    y_score = best_model.predict_proba(X_test)  # shape (n_samples, n_classes)
    # If sklearn returns list for binary classifiers, convert
    if isinstance(y_score, list):
        # list of arrays for each class -> stack
        y_score = np.vstack([col[:, 1] if col.ndim==2 else col for col in y_score]).T

    # binarize y_test
    y_test_bin = label_binarize(y_test, classes=classes)
    if y_test_bin.shape[1] == 1:
        # binary label_binarize may return shape (n_samples,1) -> squeeze to (n_samples,)
        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

    # ROC per class
    fpr = dict(); tpr = dict(); roc_auc = dict()
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
    plt.savefig(os.path.join(plots_save_path, 'roc_curves.png'), bbox_inches='tight')
    plt.close()
except Exception as e:
    print("ROC plot skipped (predict_proba not available or error):", e)

# Precision - Recall curves and average precision
try:
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
    plt.savefig(os.path.join(plots_save_path, 'precision_recall_curves.png'), bbox_inches='tight')
    plt.close()
except Exception as e:
    print("Precision-Recall plot skipped (predict_proba not available or error):", e)

# ------- FEATURE IMPORTANCE --------- #

# Extract feature importances and feature names from best model
preprocessed_features = (best_model.named_steps['preprocessing'].get_feature_names_out())
importances = best_model.named_steps['classifier'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': preprocessed_features,'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df.head(10))
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(plots_save_path, "feature_importance.png"), bbox_inches='tight')
plt.close()

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

