import pandas as pd

def load_telco_data(file_path):
    """Load the Telco dataset from CSV"""
    df = pd.read_csv(file_path)
    return df

def convert_numeric(df, numeric_cols):
    """Convert specified columns to numeric, drop rows where the conversion fails."""
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_cols)
    return df

def check_outliers(df, numeric_cols):
    """Return number of outliers per numeric column using IQR"""
    outlier_summary = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5*IQR)) | (df[col] > (Q3 + 1.5*IQR))]
        outlier_summary[col] = len(outliers)
    return outlier_summary

def check_categorical(df, categorical_cols):
    """Return unique values for categorical columns"""
    return {col: df[col].unique() for col in categorical_cols}
