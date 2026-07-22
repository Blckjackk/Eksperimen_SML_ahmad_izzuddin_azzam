import os
import json
import pandas as pd
import numpy as np
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

# --- KONFIGURASI DAGSHUB / MLFLOW ---
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "Blckjackk")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "Eksperimen_SML_ahmad_izzuddin_azzam")
DAGSHUB_TOKEN = os.getenv("MLFLOW_TRACKING_PASSWORD", os.getenv("DAGSHUB_TOKEN", ""))

def setup_mlflow():
    """
    Mengatur MLflow Tracking URI: DagsHub Remote jika token tersedia, atau Local Fallback jika offline/unauthenticated.
    """
    if DAGSHUB_TOKEN:
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
        remote_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
        mlflow.set_tracking_uri(remote_uri)
        print(f"Tracking MLflow terhubung ke DagsHub Remote: {remote_uri}")
    else:
        # Fallback ke database SQLite lokal
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        print("Tracking MLflow menggunakan penyimpanan lokal (sqlite:///mlflow.db).")
        print("Petunjuk: Set environment variable MLFLOW_TRACKING_PASSWORD='TOKEN_DAGSHUB' untuk push ke DagsHub Remote.")
        
    try:
        mlflow.set_experiment("Baseline_Model_Obesity")
    except Exception as e:
        print(f"Warning: Gagal terhubung ke remote ({e}). Mengalihkan ke SQLite MLflow lokal.")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Baseline_Model_Obesity")

def load_data():
    """
    Memuat dataset yang sudah dipreprocess dari folder obesity_preprocessing.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "obesity_preprocessing")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(base_dir, "..", "preprocessing", "obesity_preprocessing")
        
    print(f"Memuat dataset preprocessing dari: {data_dir}")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    return X_train, y_train, X_test, y_test

def create_confusion_matrix_plot(y_test, y_pred, output_path="confusion_matrix.png"):
    """
    Membuat dan menyimpan plot confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Baseline Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path

def create_feature_importance_plot(model, feature_names, output_path="feature_importance.png"):
    """
    Membuat dan menyimpan plot 10 fitur terpenting.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    plt.figure(figsize=(10, 6))
    plt.title("Top 10 Feature Importances - Baseline Model")
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path

def main():
    setup_mlflow()
    X_train, y_train, X_test, y_test = load_data()
    
    with mlflow.start_run(run_name="Baseline_RandomForest"):
        # Hyperparameter baseline
        n_estimators = 100
        max_depth = 10
        random_state = 42
        
        print("Melatih Baseline Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Predict & Evaluate
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 1. MANUAL LOGGING Hyperparameters ke MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        # 2. MANUAL LOGGING Metrics ke MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # 3. MANUAL LOGGING Artefak Gambar (Tambahan Syarat Advance: minimal 2 artefak gambar)
        cm_path = create_confusion_matrix_plot(y_test, y_pred, "confusion_matrix.png")
        fi_path = create_feature_importance_plot(model, X_train.columns, "feature_importance.png")
        
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)
        
        # Simpan Laporan Klasifikasi JSON sebagai artefak tambahan
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_path = "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=4)
        mlflow.log_artifact(report_path)
        
        # 4. MANUAL LOGGING Model Artifact
        mlflow.sklearn.log_model(model, "model")
        
        # Hapus file temporary lokal
        for tmp_file in [cm_path, fi_path, report_path]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
                
        print(f"✅ Baseline Run Selesai dengan Sukses!")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1-Score:  {f1:.4f}")

if __name__ == "__main__":
    main()
