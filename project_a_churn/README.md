# Customer Churn Prediction Project

## Overview
This project predicts which telecom customers are likely to cancel their service (churn) using machine learning. The goal is to enable targeted retention campaigns.

## Dataset
- **Source:** Telco Customer Churn (Kaggle)
- **Samples:** 7,043 customers
- **Features:** 21 (demographics, account info, services, charges)
- **Target:** Churn (Yes/No)

## Key Findings
- **Overall churn rate:** 26.5%
- **Month-to-month contracts** have 42.7% churn (vs 2.8% for 2-year contracts)
- **Electronic check** customers churn at 45.2% (vs 15% for autopay)
- **First 18 months** are highest risk period for churn

## Models Built
| Model | Accuracy | Recall | AUC |
|-------|----------|--------|-----|
| Random Forest | 79% | 67% | 0.83 |
| XGBoost | 80% | 68% | 0.84 |

## Key Features for Prediction
1. Contract type (most important)
2. Tenure (customer age)
3. Monthly charges
4. Payment method

## Business Recommendations
1. **Convert month-to-month customers** to annual contracts with 10% discount
2. **Target first 18 months** with engagement campaigns (welcome calls, tips)
3. **Incentivize autopay** (credit card/bank transfer over electronic check)
4. **Monitor high-monthly-charge customers** ($80+) for early intervention

## Files in This Project
- `churn_analysis.ipynb` - Complete analysis and modeling notebook
- `churn_distribution.html` - Interactive churn rate chart
- `churn_by_contract.html` - Churn by contract type
- `churn_by_payment.html` - Churn by payment method
- `feature_importance.html` - Top predictors
- `confusion_matrix.html` - Model performance visualization
- `roc_curve.html` - ROC curve comparison
- `churn_model.pkl` - Saved XGBoost model
- `churn_data_cleaned.csv` - Processed dataset

## How to Run
1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Open `churn_analysis.ipynb` in Jupyter/VS Code
4. Run cells in order

## Next Steps
- Deploy model as Streamlit web app
- Add real-time predictions via API
- Incorporate customer support ticket data

## Projects

### 1. Customer Churn Prediction (Telecom)

**Problem:** Predict which customers will cancel their service to enable targeted retention.

**Approach:**
- Cleaned and engineered features from 7,043 customer records
- Built Random Forest and XGBoost classifiers
- Achieved 80% accuracy, 68% recall for churners

**Key Insight:** Month-to-month contracts have 15x higher churn than 2-year contracts.

**Tech Stack:** Python, Pandas, Scikit-learn, XGBoost, Plotly

**Files:** [View Project](project_a_churn/churn_analysis.ipynb)

### 2. Iris Dataset EDA (Foundation)

**Problem:** Explore species classification using morphological measurements.

**Key Finding:** Petal length and width are highly correlated (0.96) and perfectly separate Setosa species.

**Files:** [View Notebook](data_wrangling.ipynb)