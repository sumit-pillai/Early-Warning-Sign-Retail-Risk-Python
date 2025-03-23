import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 🚀 Step 1: Generate Dummy Data

# Corporate Risk (Market Intelligence & Early Warning System)
np.random.seed(42)
n_corp = 5000  # Number of corporate clients

corporate_df = pd.DataFrame({
    'company_id': np.arange(1, n_corp+1),
    'industry': np.random.choice(['Finance', 'Retail', 'Manufacturing', 'Technology'], n_corp),
    'region': np.random.choice(['North America', 'Europe', 'Asia'], n_corp),
    'credit_score': np.random.uniform(300, 850, n_corp),
    'debt_to_equity_ratio': np.random.uniform(0.1, 5.0, n_corp),
    'revenue_growth_rate': np.random.uniform(-0.5, 0.5, n_corp),
    'cash_flow_stability': np.random.uniform(0, 1, n_corp),
    'economic_index': np.random.uniform(0.5, 1.5, n_corp),
    'default_risk': np.random.choice([0, 1], n_corp, p=[0.9, 0.1])  # 1 means high default risk
})

# Retail Risk (Fraud Detection)
n_trans = 10000  # Number of transactions

retail_df = pd.DataFrame({
    'transaction_id': np.arange(1, n_trans+1),
    'customer_id': np.random.randint(1000, 5000, n_trans),
    'transaction_amount': np.random.uniform(10, 10000, n_trans),
    'transaction_type': np.random.choice(['Online', 'In-Store', 'ATM', 'Wire Transfer'], n_trans),
    'account_age': np.random.randint(1, 20, n_trans),
    'num_previous_frauds': np.random.randint(0, 5, n_trans),
    'merchant_risk_score': np.random.uniform(0.1, 1.0, n_trans),
    'fraud_flag': np.random.choice([0, 1], n_trans, p=[0.97, 0.03])  # 1 means fraudulent transaction
})

# 🚀 Step 2: Early Warning System (Corporate Risk)
X_corp = corporate_df[['credit_score', 'debt_to_equity_ratio', 'revenue_growth_rate', 'cash_flow_stability', 'economic_index']]
y_corp = corporate_df['default_risk']

X_corp_train, X_corp_test, y_corp_train, y_corp_test = train_test_split(X_corp, y_corp, test_size=0.3, random_state=42)

ews_model = RandomForestClassifier(n_estimators=100, random_state=42)
ews_model.fit(X_corp_train, y_corp_train)

y_corp_pred = ews_model.predict(X_corp_test)
print("Early Warning System Accuracy:", accuracy_score(y_corp_test, y_corp_pred))

# 🚀 Step 3: Fraud Detection Model (Retail Risk)
X_retail = retail_df[['transaction_amount', 'account_age', 'num_previous_frauds', 'merchant_risk_score']]
y_retail = retail_df['fraud_flag']

X_retail_train, X_retail_test, y_retail_train, y_retail_test = train_test_split(X_retail, y_retail, test_size=0.3, random_state=42)

fraud_model = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
fraud_model.fit(X_retail_train)

fraud_predictions = fraud_model.predict(X_retail_test)
fraud_predictions = [1 if p == -1 else 0 for p in fraud_predictions]  # Convert -1 to 1 (fraudulent)

print("Fraud Detection Model Accuracy:", accuracy_score(y_retail_test, fraud_predictions))
print(classification_report(y_retail_test, fraud_predictions))

# 🚀 Step 4: Save Data and Models
corporate_df.to_csv("corporate_risk_data.csv", index=False)
retail_df.to_csv("fraud_risk_data.csv", index=False)

joblib.dump(ews_model, "early_warning_system.pkl")
joblib.dump(fraud_model, "fraud_detection_model.pkl")

print("Data and Models Saved Successfully!")
