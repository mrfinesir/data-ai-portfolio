"""
Test the preprocessing pipeline on new customer data
"""

import pandas as pd
import joblib
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the pipeline classes (required for loading)
from preprocessing_pipeline import DataCleaner, FeatureEngineer, EncodeCategorical, FeatureSelector

print("=" * 60)
print("TESTING PREPROCESSING PIPELINE")
print("=" * 60)

# Check if pipeline file exists
pipeline_file = 'churn_preprocessing_pipeline.pkl'
if not os.path.exists(pipeline_file):
    print(f"❌ Pipeline file '{pipeline_file}' not found.")
    print("Please run 'python preprocessing_pipeline.py' first.")
    exit(1)

# Load the saved pipeline
try:
    pipeline = joblib.load(pipeline_file)
    print("✅ Pipeline loaded successfully")
except Exception as e:
    print(f"❌ Error loading pipeline: {e}")
    print("\nTrying alternative load method...")
    try:
        import preprocessing_pipeline
        pipeline = joblib.load(pipeline_file)
        print("✅ Pipeline loaded with module import")
    except Exception as e2:
        print(f"❌ Still failing: {e2}")
        exit(1)

# Create sample new customer data
new_customers = pd.DataFrame([
    {
        'customerID': 'NEW001',
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
        'customerID': 'NEW002',
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

print(f"\n📊 Testing with {len(new_customers)} new customers")

# Apply pipeline
print("\n🔄 Transforming new customer data...")

try:
    processed = pipeline.transform(new_customers)
    print("✅ Transformation successful!")
    
    print(f"\n📊 Processed data shape: {processed.shape}")
    print(f"   Rows: {processed.shape[0]}, Features: {processed.shape[1]}")
    
    print("\n📋 First 10 processed features:")
    for i, col in enumerate(processed.columns[:10], 1):
        print(f"  {i}. {col}")
    
    print("\n📊 First customer values (first 8 features):")
    for i, col in enumerate(processed.columns[:8]):
        value = processed[col].iloc[0]
        if isinstance(value, (int, float)):
            print(f"  {col}: {value:.4f}")
        else:
            print(f"  {col}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE TEST PASSED")
    print("=" * 60)
    print("\nThe pipeline is ready for Day 12 (Model Training)!")
    
except Exception as e:
    print(f"\n❌ Error during transformation: {e}")
    import traceback
    traceback.print_exc()