# importing libraries 
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fintech_roi_ai.data_cleaning import load_telco_data, convert_numeric

#seaborn theme for graphs
sns.set_theme(style="whitegrid", palette="pastel", font_scale=1.2)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def univariate_analysis(df, numeric_cols, categorical_cols, out_dir="reports/figures"):
    """Univariate analysis for numeric and categorical columns"""
    ensure_dir(out_dir)

    # Numeric variables
    for col in numeric_cols:
        print(f"\n=== {col} Summary ===")
        print(df[col].describe())

        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True, bins=25, color="skyblue", edgecolor='black', alpha=0.7)
        sns.rugplot(df[col], color="darkblue")
        plt.title(f"Distribution of {col}", fontsize=16)
        plt.xlabel(f"{col} ({'months' if col=='tenure' else '$' if 'Charges' in col else ''})")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/univariate_{col}.png", dpi=150)
        plt.close()

    # Categorical variables
    for col in categorical_cols:
        plt.figure(figsize=(8, 6))
        ax = df[col].value_counts().sort_values(ascending=False).plot(
            kind='bar', color=sns.color_palette("Set2"), edgecolor='black', alpha=0.8
        )
        plt.title(f"Distribution of {col}", fontsize=16)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # Annotate bar counts
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        plt.savefig(f"{out_dir}/univariate_{col}.png", dpi=150)
        plt.close()


def bivariate_analysis(df, numeric_cols, target_col, out_dir="reports/figures"):
    """Bivariate analysis with correlation matrix and detailed scatterplots with reg lines"""
    ensure_dir(out_dir)

    # Handle categorical target
    if df[target_col].dtype == object:
        df[target_col + "_num"] = df[target_col].map({"No": 0, "Yes": 1})
        numeric_target = target_col + "_num"
    else:
        numeric_target = target_col

    # Correlation matrix
    plt.figure(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))  
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        mask=mask,
        cbar_kws={'shrink': 0.8},
        linewidths=0.5
    )
    plt.title("Correlation Matrix", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/correlation_matrix.png", dpi=150)
    plt.close()


    # Scatterplots with regression lines
    for col in numeric_cols:
        if col != numeric_target:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=df[col], y=df[numeric_target], alpha=0.6, s=60, color="steelblue")
            sns.regplot(x=df[col], y=df[numeric_target], scatter=False, color='red', line_kws={'linewidth':2})
            plt.title(f"{col} vs {target_col}", fontsize=16)
            plt.xlabel(f"{col} ({'months' if col=='tenure' else '$' if 'Charges' in col else ''})")
            plt.ylabel(f"{target_col} ({'0=No, 1=Yes' if 'Churn' in target_col else ''})")
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f"{out_dir}/bivariate_{col}_vs_{target_col}.png", dpi=150)
            plt.close()


def explore_class_imbalance(df, target_col, out_dir="reports/figures"):
    """Check class balance of target variable with annotated bar plot."""
    ensure_dir(out_dir)

    plt.figure(figsize=(6, 5))
    ax = df[target_col].value_counts().plot(
        kind='bar', color=sns.color_palette("coolwarm"), edgecolor='black', alpha=0.8
    )
    plt.title(f"Class Balance of {target_col}", fontsize=16)
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/class_balance.png", dpi=150)
    plt.close()

    print("\nClass Distribution:\n", df[target_col].value_counts(normalize=True))


def eda_pipeline(file_path, numeric_cols, categorical_cols, target_col,
                 out_dir="reports/figures", tables_out_dir="reports/tables",
                 processed_out_path="data/processed/telco_cleaned.csv"):
    """Full EDA pipeline"""
    ensure_dir(out_dir)
    ensure_dir(tables_out_dir)

    # Load & clean data
    df = load_telco_data(file_path)
    df = convert_numeric(df, numeric_cols)
    df.drop_duplicates(inplace=True)

    # Saving the cleaned dataset
    df.to_csv(processed_out_path, index=False)

    # Univariate Analysis
    univariate_analysis(df, numeric_cols, categorical_cols, out_dir)

    # Bivariate Analysis
    bivariate_analysis(df, numeric_cols, target_col, out_dir)

    # Class imbalance
    explore_class_imbalance(df, target_col, out_dir)

    # Saving summary statistics
    df.describe().to_csv(f"{tables_out_dir}/summary_stats.csv")

    print("\nEDA complete. Figures saved in:", out_dir)
    print("Summary stats saved in:", tables_out_dir)
