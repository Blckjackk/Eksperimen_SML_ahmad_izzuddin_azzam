import time
import random
import requests
import json

EXPORTER_URL = "http://localhost:8000"

def send_normal_predictions(n=20):
    print(f"🔄 Mengirimkan {n} request prediksi normal...")
    for i in range(n):
        features = [random.uniform(0, 1) for _ in range(23)]
        payload = {"features": features}
        try:
            res = requests.post(f"{EXPORTER_URL}/predict", json=payload, timeout=5)
            if res.status_code == 200:
                data = res.json()
                print(f"  [{i+1}/{n}] Prediksi: {data['predicted_class']} | Latency: {data['latency_seconds']}s")
            else:
                print(f"  [{i+1}/{n}] Failed: {res.status_code}")
        except Exception as e:
            print(f"  [{i+1}/{n}] Error: {e}")
        time.sleep(0.1)

def send_high_latency_predictions(n=5):
    print(f"⚠️ Mengirimkan {n} request dengan high latency (> 2 detik) untuk memicu Alerting Grafana Rule 1...")
    for i in range(n):
        payload = {"features": [0.5]*23, "simulate_delay": 2.5}
        try:
            res = requests.post(f"{EXPORTER_URL}/predict", json=payload, timeout=10)
            print(f"  [Latency Spike {i+1}/{n}] Status: {res.status_code} | Respon disimulasikan 2.5 detik")
        except Exception as e:
            print(f"  [Latency Spike {i+1}/{n}] Error: {e}")

def send_error_predictions(n=6):
    print(f"❌ Mengirimkan {n} request error untuk memicu Alerting Grafana Rule 2 (Error Rate > 5)...")
    for i in range(n):
        payload = {"simulate_error": True}
        try:
            res = requests.post(f"{EXPORTER_URL}/predict", json=payload, timeout=5)
            print(f"  [Error Request {i+1}/{n}] Status: {res.status_code} (Expected 500)")
        except Exception as e:
            print(f"  [Error Request {i+1}/{n}] Exception: {e}")

def check_metrics():
    print("📊 Mengambil snapshot metriks dari /metrics endpoint...")
    try:
        res = requests.get(f"{EXPORTER_URL}/metrics", timeout=5)
        lines = [line for line in res.text.split('\n') if line and not line.startswith('#')]
        print(f"✅ Berhasil mengambil {len(lines)} garis metriks Prometheus!")
        print("Sample Metriks Aktif:")
        for sample in lines[:10]:
            print(f"   {sample}")
    except Exception as e:
        print(f"❌ Gagal mengambil metrics: {e}")

def main():
    print("🚀 Memulai Simulasi Trafik Inference...")
    send_normal_predictions(15)
    send_high_latency_predictions(3)
    send_error_predictions(6)
    send_normal_predictions(10)
    check_metrics()
    print("✨ Simulasi trafik selesai dengan sukses!")

if __name__ == "__main__":
    main()
