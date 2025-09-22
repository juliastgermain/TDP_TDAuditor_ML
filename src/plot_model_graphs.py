#!/usr/bin/env python3
"""
plot_model_graphs.py

Reusable script for generating:
- Model calibration curves
- Feature weight barplots
- Venn diagram for scan overlap

Usage:
    python plot_model_graphs.py <predictions_file.tsv>
    (e.g., output of train_logit_model.py)
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from matplotlib_venn import venn2

def plot_calibration_curve(df, y_true_col='hit', y_pred_col='Predicted_Probability', title=None):
    y_proba = df[y_pred_col].values
    y_true = df[y_true_col].values
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title(title or 'Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_feature_weights(coef_file, top_n=20):
    # Assumes output from train_logit_model
    if isinstance(coef_file, str):
        coef_df = pd.read_csv(coef_file, sep='\t')
    else:
        coef_df = coef_file

    instrument_features = coef_df[coef_df['Feature'].str.startswith('Inst_')]
    dissociation_features = coef_df[coef_df['Feature'].str.startswith('Dissoc_')]
    other_features = coef_df[~coef_df['Feature'].str.startswith(('Inst_', 'Dissoc_'))]

    sns.set(style="whitegrid")
    plt.figure(figsize=(13, 9))
    plt.subplot(2, 1, 1)
    sns.barplot(data=instrument_features, x='Coefficient', y='Feature', palette='viridis')
    plt.title('Instrument Feature Weights')
    plt.subplot(2, 1, 2)
    sns.barplot(data=dissociation_features, x='Coefficient', y='Feature', palette='magma')
    plt.title('Dissociation Method Weights')
    plt.tight_layout()
    plt.show()
    print("\nTop Other Factors:")
    print(other_features.head(top_n)[["Feature", "Coefficient"]])

def plot_venn_for_file(df, file_name):
    subset_df = df[df["SourceFile"] == file_name]
    logit_set = set(subset_df[subset_df["Predicted_Probability"] > 0.5]["ScanNumber"])
    hit_set = set(subset_df[subset_df["hit"] == 1]["ScanNumber"])
    plt.figure(figsize=(6, 6))
    venn2([logit_set, hit_set], set_labels=('Logit > 0.5', 'Identified (Hit Report)'))
    plt.title(f"Scan Overlap for File:\n{file_name}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_model_graphs.py <predictions.tsv>")
        sys.exit(1)
    pred_file = sys.argv[1]
    df = pd.read_csv(pred_file, sep='\t')
    plot_calibration_curve(df, y_true_col='hit', y_pred_col='Predicted_Probability')
    # Save or load coefficients separately for feature plotting if needed.
    # plot_feature_weights('feature_coefficients.tsv')
    # For the Venn plot, specify a file name:
    # plot_venn_for_file(df, 'your_source_file_here')