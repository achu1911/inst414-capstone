# fintech_roi_ai/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # <--- THIS IS MISSING

def stratified_split(df, target, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split dataset into train, validation, and test with stratification.
    """
    X = df.drop(columns=[target])
    y = df[target]

    # First split: train vs temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(test_size + val_size),
        stratify=y, 
        random_state=random_state
    )

    # Second split: validation vs test
    relative_test_size = test_size / (test_size + val_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test_size,
        stratify=y_temp, 
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_features(X_train, X_val, X_test, numerical_cols=None):
    """
    1. Drops IDs safely (using copies).
    2. One-hot encodes categorical variables.
    3. Scales numerical variables (Crucial for Logistic Regression).
    """
    # 1. Create copies to avoid Side Effects
    X_train_proc = X_train.copy()
    X_val_proc = X_val.copy()
    X_test_proc = X_test.copy()

    # Drop ID if present
    for df in [X_train_proc, X_val_proc, X_test_proc]:
        if "customerID" in df.columns:
            df.drop(columns=["customerID"], inplace=True)

    # 2. One-hot encode
    X_train_enc = pd.get_dummies(X_train_proc, drop_first=True)
    X_val_enc = pd.get_dummies(X_val_proc, drop_first=True)
    X_test_enc = pd.get_dummies(X_test_proc, drop_first=True)

    # Align columns
    X_val_enc = X_val_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    # 3. Scale Numerical Columns
    if numerical_cols:
        scaler = StandardScaler()
        X_train_enc[numerical_cols] = scaler.fit_transform(X_train_enc[numerical_cols])
        X_val_enc[numerical_cols] = scaler.transform(X_val_enc[numerical_cols])
        X_test_enc[numerical_cols] = scaler.transform(X_test_enc[numerical_cols])

    return X_train_enc, X_val_enc, X_test_enc