"""
Project B: Reusable Preprocessing Pipeline for Customer Churn Data
Author: Daniel McCartney
Date: 06/05/2026
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import warnings
import os

warnings.filterwarnings('ignore')


class DataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
        X['TotalCharges'] = X['TotalCharges'].fillna(0)
        print(f"✅ Cleaned data")
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X['AvgMonthlyCharge'] = X.apply(
            lambda row: row['MonthlyCharges'] if row['tenure'] == 0 
            else row['TotalCharges'] / row['tenure'], axis=1
        )
        X['AvgMonthlyCharge'] = X['AvgMonthlyCharge'].round(2)
        
        X['TenureBucket'] = pd.cut(
            X['tenure'], bins=[-1, 6, 12, 24, 100],
            labels=['0-6 months', '6-12 months', '12-24 months', '24+ months']
        )
        
        X['ChargeCategory'] = pd.cut(
            X['MonthlyCharges'], bins=[0, 30, 60, 90, 200],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        print(f"✅ Feature engineering complete")
        return X


class EncodeCategorical(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
        self.fitted = False
    
    def fit(self, X, y=None):
        self.categorical_cols = X.select_dtypes(include=['object']).columns
        for col in self.categorical_cols:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(X[col])
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("EncodeCategorical must be fitted before transform")
        
        X = X.copy()
        for col in self.categorical_cols:
            # Handle unseen categories
            known_classes = set(self.encoders[col].classes_)
            X[col] = X[col].apply(
                lambda x: x if x in known_classes else self.encoders[col].classes_[0]
            )
            X[col + '_encoded'] = self.encoders[col].transform(X[col])
        
        X = X.drop(columns=self.categorical_cols, errors='ignore')
        print(f"✅ Encoded {len(self.categorical_cols)} categorical features")
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features
        self.fitted = False
    
    def fit(self, X, y=None):
        if self.features is None:
            exclude_patterns = ['Churn', 'customerID']
            self.features = [col for col in X.columns 
                           if not any(p in col.lower() for p in exclude_patterns)
                           and X[col].dtype in ['int64', 'float64']]
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("FeatureSelector must be fitted before transform")
        
        available = [f for f in self.features if f in X.columns]
        print(f"✅ Selected {len(available)} features")
        return X[available]


def create_full_pipeline():
    pipeline = Pipeline([
        ('cleaner', DataCleaner()),
        ('engineer', FeatureEngineer()),
        ('encoder', EncodeCategorical()),
        ('selector', FeatureSelector())
    ])
    print("✅ Pipeline created")
    return pipeline


def save_pipeline(pipeline, filepath='churn_preprocessing_pipeline.pkl'):
    joblib.dump(pipeline, filepath)
    print(f"✅ Pipeline saved to {filepath}")


def load_pipeline(filepath='churn_preprocessing_pipeline.pkl'):
    pipeline = joblib.load(filepath)
    print(f"✅ Pipeline loaded from {filepath}")
    return pipeline


if __name__ == "__main__":
    print("=" * 60)
    print("CREATING PREPROCESSING PIPELINE")
    print("=" * 60)
    
    print(f"\nCurrent directory: {os.getcwd()}")
    
    # Find CSV file
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV files found!")
        exit(1)
    
    filepath = csv_files[0]
    print(f"✅ Found CSV: {filepath}")
    
    # Load data
    print(f"\n📂 Loading {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Create pipeline
    pipeline = create_full_pipeline()
    
    # IMPORTANT: Fit the pipeline on the data
    print("\n🔄 Fitting pipeline on training data...")
    processed = pipeline.fit_transform(df)
    
    print(f"\n✅ Final shape after fitting: {processed.shape[0]} rows, {processed.shape[1]} columns")
    
    # Save the fitted pipeline
    save_pipeline(pipeline)
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE CREATED AND FITTED SUCCESSFULLY")
    print("=" * 60)
    print("\nNext: Run 'python test_pipeline.py'")