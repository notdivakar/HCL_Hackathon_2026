import pandas as pd
import numpy as np
import os

def clean_telecom_data(input_path="data/Telco-Customer-Churn-merged.csv", output_path="data/cleaned_telecom_data.csv"):
    # 1. Load raw data
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source file not found at {input_path}")
    
    df = pd.read_csv(input_path)
    initial_count = len(df)
    
    # 2. Drop duplicate customer IDs if any exist
    df = df.drop_duplicates(subset=['customerID'])
    
    # 3. Clean numeric fields (Fixing the Telco TotalCharges whitespace bug)
    numeric_cols = ['MonthlyCharges', 'TotalCharges', 'tenure', 'amount_due', 'days_overdue']
    for col in numeric_cols:
        if col in df.columns:
            # Replace empty strings or whitespace with NaN
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
            # Fill missing monetary or numeric values with 0
            df[col] = df[col].fillna(0.0)

    # 4. Standardize text / categorical columns
    text_cols = ['gender', 'Contract', 'PaymentMethod', 'status', 'loyalty_tier']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # 5. Ensure boolean columns are properly formatted
    bool_cols = ['autopay', 'dispute_open', 'email_opt_in', 'marketing_opt_in']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # 6. Save cleaned data to the Silver layer folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Data Cleaning Complete! Processed {initial_count} rows. Cleaned file saved to {output_path}")
    return df

if __name__ == "__main__":
    clean_telecom_data()