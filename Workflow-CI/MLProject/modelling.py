import os
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Izinkan file store lokal jika fallback digunakan
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

DAGSHUB_USERNAME = "Blckjackk"
DAGSHUB_REPO_NAME = "Eksperimen_SML_ahmad_izzuddin_azzam"

def setup_mlflow():
    """
    Menyiapkan MLflow tracking URI & experiment secara aman tanpa konflik MLFLOW_RUN_ID.
    """
    # Bersihkan env vars yang di-set oleh 'mlflow run' CLI wrapper agar tidak conflict experiment
    os.environ.pop("MLFLOW_RUN_ID", None)
    os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
    os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)

    token = os.getenv("MLFLOW_TRACKING_PASSWORD", os.getenv("DAGSHUB_TOKEN", ""))
    if token:
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = token
        remote_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
        mlflow.set_tracking_uri(remote_uri)
        print(f"Tracking MLflow terhubung ke DagsHub Remote: {remote_uri}")
    else:
        # Fallback ke SQLite database lokal jika tidak ada token
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        print("Tracking MLflow menggunakan penyimpanan lokal (sqlite:///mlflow.db).")

    try:
        mlflow.set_experiment("CI_Retraining_Obesity")
    except Exception as e:
        print(f"Warning: Exception set_experiment ({e}). Falling back to sqlite:///mlflow.db")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("CI_Retraining_Obesity")

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "obesity_preprocessing")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(base_dir, "..", "..", "preprocessing", "obesity_preprocessing")
        
    print(f"Memuat dataset dari: {data_dir}")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    return X_train, y_train, X_test, y_test

def main():
    setup_mlflow()
    X_train, y_train, X_test, y_test = load_data()
    
    with mlflow.start_run(run_name="CI_Automated_Retraining") as run:
        run_id = run.info.run_id
        
        # Simpan Run ID ke file run_id.txt untuk CI step
        with open("run_id.txt", "w") as f:
            f.write(run_id)
            
        print(f"Run ID berhasil dibuat: {run_id}")
        
        n_estimators = 100
        max_depth = 10
        random_state = 42
        
        print("Melatih model Random Forest (CI Retraining)...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # Log model ke MLflow
        try:
            mlflow.sklearn.log_model(model, name="model")
        except Exception:
            mlflow.sklearn.log_model(model, artifact_path="model")
        
        # Simpan juga model ke folder local_model
        local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_model")
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
        mlflow.sklearn.save_model(model, local_model_path)
        
        print(f"✅ Retraining CI Selesai! Run ID: {run_id} | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
