import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. SETUP PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'final_model_data.csv')

# 2. LOAD DATA
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data not found at {DATA_PATH}. Run data_processing.py first!")

df = pd.read_csv(DATA_PATH)

# 3. FEATURE SELECTION (Avoid Data Leakage)
# We exclude the 'Recency', 'Frequency', and 'Monetary' columns because 
# they were used to create the 'is_high_risk' label (Task 4).
# Training on them would be "cheating".
cols_to_exclude = [
    'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 
    'CustomerId', 'CurrencyCode', 'TransactionStartTime', 
    'is_high_risk', 'Recency', 'Frequency', 'Monetary', 'Cluster'
]

# Get only the numeric features (like encoded categories and hours)
X = df.drop(columns=[c for c in cols_to_exclude if c in df.columns]).select_dtypes(include=[np.number])
y = df['is_high_risk']

# 4. SPLIT DATA (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. MLFLOW EXPERIMENT SETUP
mlflow.set_experiment("Credit_Scoring_Bati_Bank")

def train_and_log_model(model, name):
    with mlflow.start_run(run_name=name):
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        preds = model.predict(X_test)
        
        # Calculate scores
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0)
        }
        
        # Calculate ROC AUC if the model supports probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, probs)
        
        # Log Metrics and Model to MLflow
        mlflow.log_metrics(metrics)
        
        # We use mlflow.sklearn for XGBoost too to avoid the _estimator_type error
        mlflow.sklearn.log_model(model, "model")
            
        print(f"\n[{name}] Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

# 6. RUN THE TRAINING
print(f"Starting Task 5 with {X.shape[1]} features...")

# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000, solver='lbfgs')
train_and_log_model(lr, "Logistic_Regression")

# Model 2: XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
train_and_log_model(xgb, "XGBoost")

print("\nTask 5 Complete. Run 'mlflow ui' to view the dashboard.")