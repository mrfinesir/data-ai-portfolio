"""
Day 12: Model Training for Customer Churn Prediction (COMPLETE FIXED VERSION)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# Import pipeline functions from Day 11
from fit_and_test import clean_data, engineer_features, select_features

print("=" * 70)
print("DAY 12: MODEL TRAINING (COMPLETE FIXED VERSION)")
print("=" * 70)

# ============================================
# HELPER FUNCTION: Safe categorical encoding
# ============================================

def safe_first_element(s):
    """Safely get first element from a set"""
    return next(iter(s)) if s else None

def encode_categorical_safe(df, encoders):
    """Encode categorical variables without set indexing issues"""
    df = df.copy()
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove columns that shouldn't be encoded
    exclude = ['customerID', 'Churn']
    categorical_cols = [c for c in categorical_cols if c not in exclude]
    
    for col in categorical_cols:
        if col in encoders:
            # Handle unseen categories safely
            df[col] = df[col].astype(str)
            known_classes = set(encoders[col].classes_)
            default_value = safe_first_element(known_classes)
            df[col] = df[col].apply(lambda x: x if x in known_classes else default_value)
            df[col + '_encoded'] = encoders[col].transform(df[col])
    
    # Drop original categorical columns
    df = df.drop(columns=[c for c in categorical_cols if c in df.columns], errors='ignore')
    
    return df

def predict_churn(customer_df, model, encoders):
    """Predict churn for a single customer"""
    cleaned = clean_data(customer_df)
    engineered = engineer_features(cleaned)
    encoded = encode_categorical_safe(engineered, encoders)
    features = select_features(encoded)
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    return pred, prob

# ============================================
# STEP 1: Load and preprocess data
# ============================================

print("\n📂 STEP 1: Loading raw data...")

csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
if not csv_files:
    print("❌ No CSV found!")
    exit(1)

df_raw = pd.read_csv(csv_files[0])
print(f"✅ Loaded {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# ============================================
# STEP 2: Apply preprocessing pipeline
# ============================================

print("\n🔄 STEP 2: Applying preprocessing pipeline...")

df = clean_data(df_raw)
print("  ✅ Cleaned")

df = engineer_features(df)
print("  ✅ Features engineered")

# Use the original encode function for training (not the safe one)
from fit_and_test import encode_categorical
df, encoders = encode_categorical(df, encoders=None)
print("  ✅ Categories encoded")

X = select_features(df)
print(f"  ✅ Features selected: {X.shape[1]} columns")

y = (df_raw['Churn'] == 'Yes').astype(int)
print(f"  ✅ Target extracted: {y.sum()} churners ({y.mean()*100:.1f}%)")

# ============================================
# STEP 3: Train/Test Split
# ============================================

print("\n📊 STEP 3: Splitting data (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")
print(f"  Training churn rate: {y_train.mean()*100:.1f}%")
print(f"  Test churn rate: {y_test.mean()*100:.1f}%")

# ============================================
# STEP 4: Train Random Forest (OPTIMIZED)
# ============================================

print("\n" + "=" * 70)
print("🌲 STEP 4: Training Random Forest (Optimized)")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("✅ Random Forest trained successfully")

# Predictions
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

# Metrics
rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_test_acc = accuracy_score(y_test, rf_test_pred)
rf_precision = precision_score(y_test, rf_test_pred)
rf_recall = recall_score(y_test, rf_test_pred)
rf_f1 = f1_score(y_test, rf_test_pred)
rf_auc = roc_auc_score(y_test, rf_test_pred)

print(f"\n📊 Random Forest Performance:")
print(f"  Training Accuracy: {rf_train_acc:.3f}")
print(f"  Test Accuracy:     {rf_test_acc:.3f}")
print(f"  Precision:         {rf_precision:.3f}")
print(f"  Recall:            {rf_recall:.3f}")
print(f"  F1-Score:          {rf_f1:.3f}")
print(f"  AUC:               {rf_auc:.3f}")

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"  Cross-val AUC:      {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# ============================================
# STEP 5: Train XGBoost (OPTIMIZED)
# ============================================

print("\n" + "=" * 70)
print("⚡ STEP 5: Training XGBoost (Optimized)")
print("=" * 70)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    use_label_encoder=False
)

xgb_model.fit(X_train, y_train)
print("✅ XGBoost trained successfully")

# Predictions
xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)

# Metrics
xgb_train_acc = accuracy_score(y_train, xgb_train_pred)
xgb_test_acc = accuracy_score(y_test, xgb_test_pred)
xgb_precision = precision_score(y_test, xgb_test_pred)
xgb_recall = recall_score(y_test, xgb_test_pred)
xgb_f1 = f1_score(y_test, xgb_test_pred)
xgb_auc = roc_auc_score(y_test, xgb_test_pred)

print(f"\n📊 XGBoost Performance:")
print(f"  Training Accuracy: {xgb_train_acc:.3f}")
print(f"  Test Accuracy:     {xgb_test_acc:.3f}")
print(f"  Precision:         {xgb_precision:.3f}")
print(f"  Recall:            {xgb_recall:.3f}")
print(f"  F1-Score:          {xgb_f1:.3f}")
print(f"  AUC:               {xgb_auc:.3f}")

# Cross-validation
cv_scores_xgb = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"  Cross-val AUC:      {cv_scores_xgb.mean():.3f} (+/- {cv_scores_xgb.std()*2:.3f})")

# ============================================
# STEP 6: Model Comparison
# ============================================

print("\n" + "=" * 70)
print("📊 STEP 6: Model Comparison")
print("=" * 70)

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'Accuracy': [rf_test_acc, xgb_test_acc],
    'Precision': [rf_precision, xgb_precision],
    'Recall': [rf_recall, xgb_recall],
    'F1-Score': [rf_f1, xgb_f1],
    'AUC': [rf_auc, xgb_auc]
})

print(comparison.to_string(index=False))

# Determine best model
if xgb_test_acc > rf_test_acc:
    best_model = 'XGBoost'
    best_score = xgb_test_acc
    best_recall = xgb_recall
else:
    best_model = 'Random Forest'
    best_score = rf_test_acc
    best_recall = rf_recall

print(f"\n🏆 Best Model: {best_model} (Accuracy: {best_score:.3f})")
print(f"   Recall (churn captured): {best_recall:.3f}")

# ============================================
# STEP 7: Feature Importance (XGBoost)
# ============================================

print("\n" + "=" * 70)
print("🔍 STEP 7: Feature Importance (XGBoost)")
print("=" * 70)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.3f}")

# ============================================
# STEP 8: Confusion Matrix Analysis (Random Forest - Best Model)
# ============================================

print("\n" + "=" * 70)
print("📊 STEP 8: Confusion Matrix (Best Model - Random Forest)")
print("=" * 70)

cm = confusion_matrix(y_test, rf_test_pred)
tn, fp, fn, tp = cm.ravel()

print(f"  True Negatives (correctly predicted no churn):  {tn}")
print(f"  False Positives (incorrectly predicted churn): {fp}")
print(f"  False Negatives (missed churners):             {fn}")
print(f"  True Positives (correctly predicted churn):    {tp}")
print(f"\n  Hit Rate (caught churners): {tp/(tp+fn):.1%}")
print(f"  False Alarm Rate: {fp/(tn+fp):.1%}")

# ============================================
# STEP 9: Save Models
# ============================================

print("\n" + "=" * 70)
print("💾 STEP 9: Saving Models")
print("=" * 70)

joblib.dump(rf_model, 'random_forest_model.pkl')
print("✅ Saved: random_forest_model.pkl")

joblib.dump(xgb_model, 'xgboost_model.pkl')
print("✅ Saved: xgboost_model.pkl")

joblib.dump(encoders, 'churn_encoders.pkl')
print("✅ Saved: churn_encoders.pkl")

joblib.dump(list(X.columns), 'churn_features.pkl')
print("✅ Saved: churn_features.pkl")

# ============================================
# STEP 10: Testing on Sample Customers (FIXED)
# ============================================

print("\n" + "=" * 70)
print("🧪 STEP 10: Testing on Sample Customers")
print("=" * 70)

# High-risk customer
high_risk = pd.DataFrame([{
    'customerID': 'RISK001', 'gender': 'Male', 'SeniorCitizen': 0,
    'Partner': 'No', 'Dependents': 'No', 'tenure': 1,
    'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No',
    'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
    'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check', 'MonthlyCharges': 95.00,
    'TotalCharges': 95.00, 'Churn': 'Yes'
}])

# Low-risk customer
low_risk = pd.DataFrame([{
    'customerID': 'SAFE001', 'gender': 'Female', 'SeniorCitizen': 0,
    'Partner': 'Yes', 'Dependents': 'Yes', 'tenure': 36,
    'PhoneService': 'Yes', 'MultipleLines': 'Yes', 'InternetService': 'DSL',
    'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes',
    'TechSupport': 'Yes', 'StreamingTV': 'No', 'StreamingMovies': 'No',
    'Contract': 'Two year', 'PaperlessBilling': 'No',
    'PaymentMethod': 'Credit card', 'MonthlyCharges': 55.00,
    'TotalCharges': 1980.00, 'Churn': 'No'
}])

# Test high-risk (using Random Forest as best model)
try:
    pred, prob = predict_churn(high_risk, rf_model, encoders)
    print(f"High-Risk Customer (1 month tenure, month-to-month):")
    print(f"  → Prediction: {'CHURN' if pred == 1 else 'NO CHURN'}")
    print(f"  → Probability: {prob:.1%}\n")
except Exception as e:
    print(f"Error testing high-risk: {e}")

# Test low-risk
try:
    pred, prob = predict_churn(low_risk, rf_model, encoders)
    print(f"Low-Risk Customer (36 months tenure, 2-year contract):")
    print(f"  → Prediction: {'CHURN' if pred == 1 else 'NO CHURN'}")
    print(f"  → Probability: {prob:.1%}")
except Exception as e:
    print(f"Error testing low-risk: {e}")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 70)
print("✅ DAY 12 COMPLETE!")
print("=" * 70)
print(f"\n📊 Final Model Performance:")
print(f"  Best Model: {best_model}")
print(f"  Accuracy: {best_score:.3f} ({best_score*100:.1f}%)")
print(f"  Recall: {best_recall:.3f} (catches {best_recall*100:.1f}% of churners)")
print(f"\n📁 Saved Files:")
print("  • random_forest_model.pkl")
print("  • xgboost_model.pkl")
print("  • churn_encoders.pkl")
print("  • churn_features.pkl")
print("\n🚀 Ready for Deployment or Next Steps!")