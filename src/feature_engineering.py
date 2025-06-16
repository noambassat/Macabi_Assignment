import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split


def extract_last_week_paragraph(text):
    """
    Extract the last paragraph explicitly starting with 'שבוע'.
    Returns entire text as fallback if not found.
    """
    splits = re.split(r'(?=^\s*שבוע\s*\d+)', text, flags=re.MULTILINE)
    valid_splits = [split.strip() for split in splits if split.strip()]
    return valid_splits[-1] if valid_splits else text.strip()


def generate_features(df, text_col='clinical_sheet', target_col='Y', test_size=0.3, random_state=42):
    """
    Prepares and splits data into train/test with last-week paragraph extraction and stratified sampling.
    """
    df = df.copy()
    df['last_week_paragraph'] = df[text_col].apply(extract_last_week_paragraph)

    indicator_cols = [
        'match_diag_141', 'match_rasham_after', 'match_aspirin_after', 'match_pdf_after',
        'essential_hypertension_sum', 'pregnancy_hypertension_sum',
        'preeclampsia_sum', 'eclampsia_sum', 'labs_sum'
    ]

    stratify_key = df[indicator_cols].astype(str).agg('-'.join, axis=1)
    rare_groups = stratify_key.value_counts()[lambda x: x == 1].index
    stratify_key_fixed = stratify_key.replace(rare_groups, 'rare')

    columns_to_drop = indicator_cols + [
        'no_match', 'severity_level', 'diagnosis_source', 'hypertension_target'
    ]

    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify_key_fixed
    )

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test
