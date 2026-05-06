"""
Complete pipeline fitting and testing - SIMPLIFIED VERSION
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# SIMPLE FUNCTIONS (no scikit-learn pipeline)
# ============================================

def clean_data(df):
    """Clean the data"""
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df

def engineer_features(df):
    """Create new features"""
    df = df.copy()
    
    # Average monthly charge
    df['AvgMonthlyCharge'] = df.apply(
        lambda row: row['MonthlyCharges'] if row['tenure'] == 0 
        else row['TotalCharges'] / row['tenure'], axis=1
    )
    df['AvgMonthlyCharge'] = df['AvgMonthlyCharge'].round(2)
    
    # Tenure buckets
    df['TenureBucket'] = pd.cut(
        df['tenure'], bins=[-1, 6, 12, 24, 100],
        labels=['0-6', '6-12', '12-24', '24+']
    )
    
    # Charge categories
    df['ChargeCategory'] = pd.cut(
        df['MonthlyCharges'], bins=[0, 30, 60, 90, 200],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    return df

def encode_categorical(df, encoders=None):
    """Encode categorical variables"""
    df = df.copy()
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove columns that shouldn't be encoded
    exclude = ['customerID', 'Churn']
    categorical_cols = [c for c in categorical_cols if c not in exclude]
    
    if encoders is None:
        # Fitting mode - create new encoders
        encoders = {}
        for col in categorical_cols:
            encoders[col] = LabelEncoder()
            df[col + '_encoded'] = encoders[col].fit_transform(df[col].astype(str))
    else:
        # Transform mode - use existing encoders
        for col in categorical_cols:
            if col in encoders:
                # Handle unseen categories
                df[col] = df[col].astype(str)
                known_classes = set(encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in known_classes else known_classes[0])
                df[col + '_encoded'] = encoders[col].transform(df[col])
            else:
                print(f"Warning: No encoder for {col}")
    
    # Drop original categorical columns
    df = df.drop(columns=categorical_cols, errors='ignore')
    
    return df, encoders

def select_features(df):
    """Select final features for modeling"""
    exclude_patterns = ['customerID', 'Churn']
    features = [col for col in df.columns 
               if not any(p in col for p in exclude_patterns)
               and df[col].dtype in ['int64', 'float64']]
    return df[features]

def fit_full_pipeline(df):
    """Fit the entire pipeline on training data"""
    print("🔄 Fitting pipeline...")
    
    # Step 1: Clean
    df = clean_data(df)
    print("  ✅ Cleaned")
    
    # Step 2: Feature engineering
    df = engineer_features(df)
    print("  ✅ Features engineered")
    
    # Step 3: Encode (fit mode)
    df, encoders = encode_categorical(df, encoders=None)
    print("  ✅ Categories encoded")
    
    # Step 4: Select features
    df = select_features(df)
    print(f"  ✅ Selected {df.shape[1]} features")
    
    # Save encoders for later use
    return df, encoders

def transform_new_data(df, encoders):
    """Transform new data using fitted encoders"""
    df = clean_data(df)
    df = engineer_features(df)
    df, _ = encode_categorical(df, encoders=encoders)
    df = select_features(df)
    return df

# ============================================
# MAIN EXECUTION
# ============================================

print("=" * 70)
print("FITTING AND TESTING PIPELINE")
print("=" * 70)

# Step 1: Find and load data
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
if not csv_files:
    print("❌ No CSV found!")
    exit(1)

print(f"Found CSV: {csv_files[0]}")
df_raw = pd.read_csv(csv_files[0])
print(f"✅ Loaded {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# Step 2: Fit pipeline on full dataset
print("\n" + "=" * 70)
df_processed, encoders = fit_full_pipeline(df_raw)
print(f"\n✅ Pipeline fitted successfully!")
print(f"   Output shape: {df_processed.shape[0]} rows, {df_processed.shape[1]} columns")

# Step 3: Save encoders for later use
joblib.dump(encoders, 'churn_encoders.pkl')
print("✅ Encoders saved to 'churn_encoders.pkl'")

# Step 4: Save the list of expected features
feature_list = list(df_processed.columns)
joblib.dump(feature_list, 'churn_features.pkl')
print("✅ Feature list saved to 'churn_features.pkl'")

# Step 5: Test on new customers
print("\n" + "=" * 70)
print("TESTING ON NEW CUSTOMERS")
print("=" * 70)

test_customers = pd.DataFrame([
    {
        'customerID': 'TEST1',
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 3,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.50,
        'TotalCharges': 250.00,
        'Churn': 'No'
    },
    {
        'customerID': 'TEST2',
        'gender': 'Female',
        'SeniorCitizen': 1,
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'tenure': 24,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Two year',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Bank transfer (automatic)',
        'MonthlyCharges': 45.20,
        'TotalCharges': 1080.00,
        'Churn': 'No'
    }
])

print(f"📊 Testing with {len(test_customers)} new customers")

try:
    # Transform using the saved encoders
    result = transform_new_data(test_customers, encoders)
    print(f"✅ Test successful!")
    print(f"   Result shape: {result.shape[0]} rows, {result.shape[1]} columns")
    print(f"   Features: {list(result.columns[:5])}...")
    
    print("\n" + "=" * 70)
    print("✅ DAY 11 COMPLETE - PIPELINE IS READY FOR DAY 12!")
    print("=" * 70)
    print("\nSaved files:")
    print("  📁 churn_encoders.pkl - Label encoders for categorical variables")
    print("  📁 churn_features.pkl - List of expected features")
    print("\nThe pipeline can now be used for:")
    print("  ✅ Transforming new customer data")
    print("  ✅ Training machine learning models (Day 12)")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()