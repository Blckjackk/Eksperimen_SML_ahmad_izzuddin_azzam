import os
import time
import random
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    ProcessCollector,
    GCCollector
)

app = Flask(__name__)

# --- INISIALISASI PROMETHEUS METRICS (10 METRIKS DENGAN METRIKS APLIKASI + METRIKS SISTEM) ---
registry = CollectorRegistry()
ProcessCollector(registry=registry)
GCCollector(registry=registry)

# 1. Custom Metric 1: Total HTTP Requests
HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total',
    'Total number of HTTP requests received',
    ['method', 'endpoint', 'status'],
    registry=registry
)

# 2. Custom Metric 2: Total Predictions Processed
PREDICTION_COUNT_TOTAL = Counter(
    'prediction_count_total',
    'Total predictions processed per predicted class',
    ['predicted_class'],
    registry=registry
)

# 3. Custom Metric 3: Prediction Latency
PREDICTION_LATENCY_SECONDS = Histogram(
    'prediction_latency_seconds',
    'Latency of model prediction responses in seconds',
    ['endpoint'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    registry=registry
)

# 4. Custom Metric 4: Prediction Error Count
PREDICTION_ERROR_COUNT = Counter(
    'prediction_error_count',
    'Total prediction errors encountered',
    ['error_type'],
    registry=registry
)

# 5. Custom Metric 5: Active Concurrent Requests
ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Number of active concurrent requests being processed',
    registry=registry
)

# --- LOAD ATAU TRAIN MODEL MACHINE LEARNING DUMMY/SAVED ---
MODEL = None
CLASS_NAMES = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

def load_or_train_model():
    global MODEL
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.join(base_dir, "..", "Workflow-CI", "MLProject", "local_model")
    
    if os.path.exists(local_model_path):
        try:
            MODEL = mlflow.sklearn.load_model(local_model_path)
            print(f"✅ Model ML berhasil dimuat dari: {local_model_path}")
            return
        except Exception as e:
            print(f"Warning: Gagal memuat model dari {local_model_path}: {e}")
            
    print("Melatih model Random Forest fallback untuk Exporter...")
    X_dummy = np.random.rand(100, 23)
    y_dummy = np.random.randint(0, 7, size=100)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_dummy, y_dummy)
    MODEL = rf
    print("✅ Model Fallback siap digerakkan.")

load_or_train_model()

@app.before_request
def before_request_func():
    ACTIVE_REQUESTS.inc()

@app.after_request
def after_request_func(response):
    ACTIVE_REQUESTS.dec()
    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=request.path,
        status=response.status_code
    ).inc()
    return response

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "Obesity ML Exporter"})

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'features' not in data:
            # Simulasi random feature jika request kosong
            features = np.random.rand(1, 23)
        else:
            features = np.array(data['features']).reshape(1, -1)

        # Opsi simulasi error untuk pengujian alert error rate
        if data and data.get('simulate_error') is True:
            PREDICTION_ERROR_COUNT.labels(error_type="SimulatedException").inc()
            return jsonify({"error": "Simulated error during inference"}), 500

        # Opsi simulasi high latency untuk pengujian alert latency (>2 detik)
        if data and data.get('simulate_delay'):
            delay = float(data.get('simulate_delay'))
            time.sleep(delay)

        pred_idx = int(MODEL.predict(features)[0])
        predicted_class = CLASS_NAMES[pred_idx if pred_idx < len(CLASS_NAMES) else 0]

        PREDICTION_COUNT_TOTAL.labels(predicted_class=predicted_class).inc()
        
        latency = time.time() - start_time
        PREDICTION_LATENCY_SECONDS.labels(endpoint='/predict').observe(latency)

        return jsonify({
            "status": "success",
            "prediction_index": pred_idx,
            "predicted_class": predicted_class,
            "latency_seconds": round(latency, 4)
        }), 200

    except Exception as e:
        PREDICTION_ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        latency = time.time() - start_time
        PREDICTION_LATENCY_SECONDS.labels(endpoint='/predict').observe(latency)
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(registry), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    print("🚀 Running Prometheus Exporter ML Server di http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=False)
