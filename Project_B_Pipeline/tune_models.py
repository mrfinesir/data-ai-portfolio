"""
Day 13: Hyperparameter Tuning for Churn Prediction Models
Optimizes Random Forest and XGBoost for better performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import warnings
import os
import time

warnings.filterwarnings('ignore')

# Import pipeline functions from Day 11
from fit_and_test import clean_data, engineer_features, encode_categorical, select_features

print("=" * 70)
print("DAY 13: HYPERPARAMETER TUNING")
print("=" * 70)

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
df = engineer_features(df)
df, encoders = encode_categorical(df, encoders=None)
X = select_features(df)
y = (df_raw['Churn'] == 'Yes').astype(int)

print(f"  ✅ Features: {X.shape[1]} columns")
print(f"  ✅ Target: {y.sum()} churners ({y.mean()*100:.1f}%)")

# ============================================
# STEP 3: Train/Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# ============================================
# STEP 4: Random Forest Hyperparameter Tuning
# ============================================

print("\n" + "=" * 70)
print("🌲 STEP 4: Random Forest Hyperparameter Tuning")
print("=" * 70)

# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 8],
    'class_weight': ['balanced', 'balanced_subsample']
}

print("Parameter grid:")
for param, values in rf_param_grid.items():
    print(f"  {param}: {values}")

print("\n🔄 Running RandomizedSearchCV (this may take 3-5 minutes)...")
start_time = time.time()

# Use RandomizedSearchCV for faster results (tests fewer combinations)
rf_random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    n_iter=20,  # Test 20 random combinations
    cv=3,       # 3-fold cross-validation
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_random_search.fit(X_train, y_train)

elapsed = time.time() - start_time
print(f"\n✅ Tuning completed in {elapsed:.1f} seconds")

# Best parameters and score
print(f"\n🏆 Best Random Forest Parameters:")
for param, value in rf_random_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\n📊 Best Cross-Validation AUC: {rf_random_search.best_score_:.4f}")

# Evaluate on test set
rf_best = rf_random_search.best_estimator_
rf_test_pred = rf_best.predict(X_test)
rf_test_proba = rf_best.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_test_pred)
rf_precision = precision_score(y_test, rf_test_pred)
rf_recall = recall_score(y_test, rf_test_pred)
rf_f1 = f1_score(y_test, rf_test_pred)
rf_auc = roc_auc_score(y_test, rf_test_proba)

print(f"\n📊 Random Forest Test Performance:")
print(f"  Accuracy:  {rf_accuracy:.4f}")
print(f"  Precision: {rf_precision:.4f}")
print(f"  Recall:    {rf_recall:.4f}")
print(f"  F1-Score:  {rf_f1:.4f}")
print(f"  AUC:       {rf_auc:.4f}")

# ============================================
# STEP 5: XGBoost Hyperparameter Tuning
# ============================================

print("\n" + "=" * 70)
print("⚡ STEP 5: XGBoost Hyperparameter Tuning")
print("=" * 70)

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Define parameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

print("Parameter grid:")
for param, values in xgb_param_grid.items():
    print(f"  {param}: {values}")

print("\n🔄 Running RandomizedSearchCV (this may take 3-5 minutes)...")
start_time = time.time()

xgb_random_search = RandomizedSearchCV(
    XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        use_label_encoder=False
    ),
    xgb_param_grid,
    n_iter=20,
    cv=3,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_random_search.fit(X_train, y_train)

elapsed = time.time() - start_time
print(f"\n✅ Tuning completed in {elapsed:.1f} seconds")

# Best parameters and score
print(f"\n🏆 Best XGBoost Parameters:")
for param, value in xgb_random_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\n📊 Best Cross-Validation AUC: {xgb_random_search.best_score_:.4f}")

# Evaluate on test set
xgb_best = xgb_random_search.best_estimator_
xgb_test_pred = xgb_best.predict(X_test)
xgb_test_proba = xgb_best.predict_proba(X_test)[:, 1]

xgb_accuracy = accuracy_score(y_test, xgb_test_pred)
xgb_precision = precision_score(y_test, xgb_test_pred)
xgb_recall = recall_score(y_test, xgb_test_pred)
xgb_f1 = f1_score(y_test, xgb_test_pred)
xgb_auc = roc_auc_score(y_test, xgb_test_proba)

print(f"\n📊 XGBoost Test Performance:")
print(f"  Accuracy:  {xgb_accuracy:.4f}")
print(f"  Precision: {xgb_precision:.4f}")
print(f"  Recall:    {xgb_recall:.4f}")
print(f"  F1-Score:  {xgb_f1:.4f}")
print(f"  AUC:       {xgb_auc:.4f}")

# ============================================
# STEP 6: Compare Before vs After Tuning
# ============================================

print("\n" + "=" * 70)
print("📊 STEP 6: Performance Improvement Summary")
print("=" * 70)

# Baseline metrics (from Day 12)
rf_baseline = {'Accuracy': 0.778, 'Recall': 0.706, 'AUC': 0.755}
xgb_baseline = {'Accuracy': 0.764, 'Recall': 0.676, 'AUC': 0.736}

print("\nRandom Forest Improvement:")
print(f"  Accuracy:  {rf_baseline['Accuracy']:.3f} → {rf_accuracy:.3f} ({rf_accuracy - rf_baseline['Accuracy']:+.3f})")
print(f"  Recall:    {rf_baseline['Recall']:.3f} → {rf_recall:.3f} ({rf_recall - rf_baseline['Recall']:+.3f})")
print(f"  AUC:       {rf_baseline['AUC']:.3f} → {rf_auc:.3f} ({rf_auc - rf_baseline['AUC']:+.3f})")

print("\nXGBoost Improvement:")
print(f"  Accuracy:  {xgb_baseline['Accuracy']:.3f} → {xgb_accuracy:.3f} ({xgb_accuracy - xgb_baseline['Accuracy']:+.3f})")
print(f"  Recall:    {xgb_baseline['Recall']:.3f} → {xgb_recall:.3f} ({xgb_recall - xgb_baseline['Recall']:+.3f})")
print(f"  AUC:       {xgb_baseline['AUC']:.3f} → {xgb_auc:.3f} ({xgb_auc - xgb_baseline['AUC']:+.3f})")

# ============================================
# STEP 7: Determine Best Tuned Model
# ============================================

print("\n" + "=" * 70)
print("🏆 STEP 7: Best Tuned Model Selection")
print("=" * 70)

comparison = pd.DataFrame({
    'Model': ['Random Forest (Tuned)', 'XGBoost (Tuned)'],
    'Accuracy': [rf_accuracy, xgb_accuracy],
    'Recall': [rf_recall, xgb_recall],
    'AUC': [rf_auc, xgb_auc]
})

print("\n", comparison.to_string(index=False))

if xgb_auc > rf_auc:
    best_model = xgb_best
    best_name = "XGBoost"
    best_score = xgb_accuracy
    best_recall = xgb_recall
    print(f"\n🏆 Best Tuned Model: {best_name} (AUC: {xgb_auc:.3f}, Recall: {xgb_recall:.3f})")
else:
    best_model = rf_best
    best_name = "Random Forest"
    best_score = rf_accuracy
    best_recall = rf_recall
    print(f"\n🏆 Best Tuned Model: {best_name} (AUC: {rf_auc:.3f}, Recall: {rf_recall:.3f})")

# ============================================
# STEP 8: Confusion Matrix for Best Model
# ============================================

print("\n" + "=" * 70)
print(f"📊 STEP 8: Confusion Matrix ({best_name} - Tuned)")
print("=" * 70)

best_test_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_test_pred)
tn, fp, fn, tp = cm.ravel()

print(f"  True Negatives (correctly predicted no churn):  {tn}")
print(f"  False Positives (incorrectly predicted churn): {fp}")
print(f"  False Negatives (missed churners):             {fn}")
print(f"  True Positives (correctly predicted churn):    {tp}")
print(f"\n  Hit Rate (caught churners): {tp/(tp+fn):.1%}")
print(f"  False Alarm Rate: {fp/(tn+fp):.1%}")

# ============================================
# STEP 9: Feature Importance (Best Model)
# ============================================

print("\n" + "=" * 70)
print(f"🔍 STEP 9: Feature Importance ({best_name})")
print("=" * 70)

if best_name == "Random Forest":
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
else:
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
for i, row in importance.head(10).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.3f}")

# ============================================
# STEP 10: Save Tuned Models
# ============================================

print("\n" + "=" * 70)
print("💾 STEP 10: Saving Tuned Models")
print("=" * 70)

joblib.dump(rf_best, 'random_forest_tuned.pkl')
print("✅ Saved: random_forest_tuned.pkl")

joblib.dump(xgb_best, 'xgboost_tuned.pkl')
print("✅ Saved: xgboost_tuned.pkl")

joblib.dump(best_model, 'best_churn_model.pkl')
print(f"✅ Saved: best_churn_model.pkl ({best_name})")

joblib.dump(encoders, 'churn_encoders.pkl')
print("✅ Saved: churn_encoders.pkl")

joblib.dump(list(X.columns), 'churn_features.pkl')
print("✅ Saved: churn_features.pkl")

# ============================================
# STEP 11: Quick Test on Sample Customer
# ============================================

print("\n" + "=" * 70)
print("🧪 STEP 11: Testing Best Model on Sample Customers")
print("=" * 70)

# Helper for safe encoding
def safe_first_element(s):
    return next(iter(s)) if s else None

def encode_categorical_safe(df, encoders):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    exclude = ['customerID', 'Churn']
    categorical_cols = [c for c in categorical_cols if c not in exclude]
    
    for col in categorical_cols:
        if col in encoders:
            df[col] = df[col].astype(str)
            known_classes = set(encoders[col].classes_)
            default_value = safe_first_element(known_classes)
            df[col] = df[col].apply(lambda x: x if x in known_classes else default_value)
            df[col + '_encoded'] = encoders[col].transform(df[col])
    
    df = df.drop(columns=[c for c in categorical_cols if c in df.columns], errors='ignore')
    return df

def predict_churn(customer_df, model, encoders):
    cleaned = clean_data(customer_df)
    engineered = engineer_features(cleaned)
    encoded = encode_categorical_safe(engineered, encoders)
    features = select_features(encoded)
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    return pred, prob

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

# Test
try:
    pred, prob = predict_churn(high_risk, best_model, encoders)
    print(f"High-Risk Customer (1 month tenure, month-to-month):")
    print(f"  → Prediction: {'CHURN' if pred == 1 else 'NO CHURN'}")
    print(f"  → Probability: {prob:.1%}\n")
except Exception as e:
    print(f"Error testing high-risk: {e}")

try:
    pred, prob = predict_churn(low_risk, best_model, encoders)
    print(f"Low-Risk Customer (36 months tenure, 2-year contract):")
    print(f"  → Prediction: {'CHURN' if pred == 1 else 'NO CHURN'}")
    print(f"  → Probability: {prob:.1%}")
except Exception as e:
    print(f"Error testing low-risk: {e}")

# ============================================
# SUMMARY
# ============================================

print("\n" + "=" * 70)
print("✅ DAY 13 COMPLETE!")
print("=" * 70)
print(f"\n📊 Final Tuned Model Performance ({best_name}):")
print(f"  Accuracy: {best_score:.3f} ({best_score*100:.1f}%)")
print(f"  Recall:   {best_recall:.3f} (catches {best_recall*100:.1f}% of churners)")
print(f"  AUC:      {roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]):.3f}")
print(f"\n📁 Saved Files:")
print("  • random_forest_tuned.pkl")
print("  • xgboost_tuned.pkl")
print("  • best_churn_model.pkl ← Use this for predictions")
print("\n🚀 Day 13 Complete! Your model is now optimized!")