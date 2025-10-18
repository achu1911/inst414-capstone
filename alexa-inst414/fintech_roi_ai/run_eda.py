from fintech_roi_ai.eda_analysis import eda_pipeline

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_cols = ["gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines",
                    "InternetService","OnlineSecurity","DeviceProtection","TechSupport","StreamingTV",
                    "StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
target_col = "Churn"
file_path = "data/raw/telco_churn.csv"

eda_pipeline(file_path, numeric_cols, categorical_cols, target_col)

