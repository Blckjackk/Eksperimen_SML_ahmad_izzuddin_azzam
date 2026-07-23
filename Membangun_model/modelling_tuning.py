import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "Blckjackk")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "Eksperimen_SML_ahmad_izzuddin_azzam")
DAGSHUB_TOKEN = os.getenv("MLFLOW_TRACKING_PASSWORD", os.getenv("DAGSHUB_TOKEN", ""))

def setup_mlflow():
    if DAGSHUB_TOKEN:
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
        remote_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
        mlflow.set_tracking_uri(remote_uri)
        print(f"Tracking MLflow connected to DagsHub remote: {remote_uri}")
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        print("Tracking MLflow using local sqlite storage.")
        
    try:
        mlflow.set_experiment("Hyperparameter_Tuning_Obesity")
    except Exception as e:
        print(f"Warning: Failed to connect to remote tracking server ({e}). Falling back to local storage.")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("Hyperparameter_Tuning_Obesity")

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "obesity_preprocessing")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(base_dir, "..", "preprocessing", "obesity_preprocessing")
        
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    return X_train, y_train, X_test, y_test

def create_confusion_matrix_plot(y_test, y_pred, output_path="confusion_matrix_tuning.png"):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix - Hyperparameter Tuning Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path

def create_feature_importance_plot(model, feature_names, output_path="feature_importance_tuning.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    plt.figure(figsize=(10, 6))
    plt.title("Top 10 Feature Importances - Hyperparameter Tuning Model")
    plt.bar(range(len(indices)), importances[indices], align="center", color='forestgreen')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path

def main():
    setup_mlflow()
    X_train, y_train, X_test, y_test = load_data()
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'criterion': ['gini', 'entropy']
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    
    print("Running GridSearchCV hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    with mlflow.start_run(run_name="GridSearch_Tuning_Best"):
        for key, val in best_params.items():
            mlflow.log_param(key, val)
        mlflow.log_param("best_cv_score", float(grid_search.best_score_))
        mlflow.log_param("tuning_method", "GridSearchCV")
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        cm_path = create_confusion_matrix_plot(y_test, y_pred, "confusion_matrix_tuning.png")
        fi_path = create_feature_importance_plot(best_model, X_train.columns, "feature_importance_tuning.png")
        
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)
        
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_path = "classification_report_tuning.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=4)
        mlflow.log_artifact(report_path)
        
        mlflow.sklearn.log_model(best_model, "model")
        
        for tmp_file in [cm_path, fi_path, report_path]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
                
        print("Tuning run completed successfully.")
        print(f"Best Params: {best_params}")
        print(f"Test Accuracy:  {acc:.4f}")
        print(f"Test Precision: {prec:.4f}")
        print(f"Test Recall:    {rec:.4f}")
        print(f"Test F1-Score:  {f1:.4f}")

if __name__ == "__main__":
    main()
