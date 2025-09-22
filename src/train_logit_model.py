#!/usr/bin/env python3
"""
train_logit_model.py

Reusable script for training a logistic regression model with SMOTE,
feature engineering, and evaluating on train, test, and prediction splits.

Works for any dataset processed:
'tdauditor_topfd_with_instrument.tsv', 'tdauditor_flashdeconv_with_instrument.tsv',
'tdauditor_pro_with_instrument.tsv'.

Usage:
    python train_logit_model.py path_to_tsv

Outputs:
    - Saves predictions in a new file: <input>_predictions.tsv
    - Prints model stats and feature weights
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def onehot_encode(df):
    df = pd.get_dummies(df, columns=['mzMLDissociation'], prefix='dissoc')
    df = pd.get_dummies(df, columns=['Instrument'], prefix='inst')
    return df

def add_features(df):
    if 'LongestTag' in df and 'DeconvPeakCount' in df:
        for col in df.columns:
            if col.startswith('dissoc_'):
                df[f'{col}_x_LongestTag'] = df[col] * df['LongestTag']
            if col.startswith('inst_'):
                df[f'{col}_x_DeconvPeakCount'] = df[col] * df['DeconvPeakCount']
        df['IsLongTagReliable'] = ((df['LongestTag'] < 25) & (df['DeconvPeakCount'] < 250))
    return df

def train_model(file_path):
    print(f"\n[INFO] Loading {file_path} ...")
    df = pd.read_csv(file_path, delimiter="\t")
    df = df.dropna(subset=['hit'])
    df['Instrument'] = df['Instrument'].fillna('Orbitrap Eclipse')

    features = df.drop(columns=['hit', 'SourceFile', 'NativeID'], errors='ignore')
    X = features.copy()
    y = df["hit"]

    # Split filenames for proper group separation
    filenames = df["SourceFile"].unique()
    train_filenames, temp_filenames = train_test_split(filenames, test_size=0.3, random_state=42)
    test_filenames, predict_filenames = train_test_split(temp_filenames, test_size=0.5, random_state=42)

    train_df = df[df["SourceFile"].isin(train_filenames)]
    test_df = df[df["SourceFile"].isin(test_filenames)]
    predict_df = df[df["SourceFile"].isin(predict_filenames)]

    X_train, y_train = train_df[features.columns], train_df["hit"]
    X_test, y_test = test_df[features.columns], test_df["hit"]
    X_predict, y_predict_true = predict_df[features.columns], predict_df["hit"]

    for dataset in [X_train, X_test, X_predict]:
        if 'MatchedToDeconvolution' in dataset:
            dataset.loc[:, 'MatchedToDeconvolution'] = dataset['MatchedToDeconvolution'].astype(int)

    # Encode categorical variables
    X_train = onehot_encode(X_train)
    X_test = onehot_encode(X_test)
    X_predict = onehot_encode(X_predict)

    # Synchronize columns
    train_cols = X_train.columns
    X_test = X_test.reindex(columns=train_cols, fill_value=0)
    X_predict = X_predict.reindex(columns=train_cols, fill_value=0)

    # Feature engineering
    X_train = add_features(X_train)
    X_test = add_features(X_test)
    X_predict = add_features(X_predict)

    # Impute missing values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())
    X_predict = X_predict.fillna(X_train.mean())

    # Pipeline with SMOTE and scaling
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=0.01, random_state=42, max_iter=1000, class_weight='balanced'))
    ])

    # Cross-validation
    print("\n[INFO] Performing cross-validation...")
    groups = train_df['SourceFile']
    cv = GroupKFold(n_splits=5)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(pipeline, X_train, y_train, groups=groups, cv=cv, scoring=scoring, n_jobs=-1)
    for metric in scoring:
        mean_score = np.mean(cv_results[f'test_{metric}'])
        std_score = np.std(cv_results[f'test_{metric}'])
        print(f"CV {metric}: {mean_score:.3f} ± {std_score:.3f}")

    # Train model and collect coefficients
    pipeline.fit(X_train, y_train)
    feature_mapping = {
        **{f'dissoc_{name}': f'Dissoc_{name}' for name in df['mzMLDissociation'].unique() if pd.notnull(name)},
        **{f'inst_{name}': f'Inst_{name}' for name in df['Instrument'].unique() if pd.notnull(name)}
    }
    feature_names = [feature_mapping.get(col, col) for col in X_train.columns]
    final_model = pipeline.named_steps['model']

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": final_model.coef_[0]
    }).sort_values("Coefficient", key=lambda x: abs(x), ascending=False)

    print("\nTop 20 Most Influential Features:")
    print(coef_df.head(20)[["Feature", "Coefficient"]])

    # Evaluate
    X_test_processed = pipeline[:-1].transform(X_test)
    y_test_pred = pipeline.named_steps['model'].predict(X_test_processed)
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print("Classification Report (Test Set):\n", classification_report(y_test, y_test_pred))

    # Predict
    X_pred_processed = pipeline[:-1].transform(X_predict)
    y_pred_proba = pipeline.named_steps['model'].predict_proba(X_pred_processed)[:, 1]
    y_pred_adjusted = (y_pred_proba >= 0.5).astype(int)

    print("\n[INFO] Prediction Set Results (first 12 predictions):")
    print("Probabilities:", y_pred_proba[:12])
    print("Predicted Hits:", y_pred_adjusted[:12])
    print("True Hits:", y_predict_true.values[:12])

    # Save predictions
    predict_df['Predicted_Probability'] = y_pred_proba
    predict_df['Predicted_Hit'] = y_pred_adjusted
    out_file = file_path.replace('.tsv', '_predictions.tsv')
    predict_df.to_csv(out_file, sep='\t', index=False)
    print(f"\n[INFO] Saved prediction results to '{out_file}'.")

    # Also return for downstream graphs
    return coef_df, predict_df, y_predict_true

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_logit_model.py <input_file.tsv>")
        sys.exit(1)
    coef_df, predict_df, y_true = train_model(sys.argv[1])