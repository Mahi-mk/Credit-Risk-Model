import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. SETUP PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'final_model_data.csv')

# 2. LOAD DATA
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data not found at {DATA_PATH}. Run data_processing.py first!")

df = pd.read_csv(DATA_PATH)

# 3. FEATURE SELECTION (Matching API exactly)
# We select ONLY the 3 features that the API's TransactionData model uses.
# This prevents the "feature_names mismatch" error in the browser.
features_for_api = ['TransactionHour', 'TransactionDay', 'Amount_scaled']
X = df[features_for_api]
y = df['is_high_risk']

# 4. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. MLFLOW EXPERIMENT SETUP
mlflow.set_experiment("Credit_Scoring_Bati_Bank")

def train_and_log_model(model, name):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0)
        }
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, probs)
        
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
            
        print(f"\n[{name}] Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

# 6. RUN THE TRAINING
print(f"Starting Task 5 with {X.shape[1]} features: {features_for_api}")

# Training XGBoost as our 'best_model'
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
train_and_log_model(xgb, "XGBoost_API_Version")

# 7. SAVE FOR API
# This overwrites the old 29-feature model with the new 3-feature model.
save_path = os.path.join(os.getcwd(), 'src', 'api', 'best_model.pkl')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
joblib.dump(xgb, save_path)

print(f"\nTask 5 Complete. Model saved at: {save_path}")
