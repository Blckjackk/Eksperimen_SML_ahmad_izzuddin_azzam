import time
import random
import requests
import json

EXPORTER_URL = "http://localhost:8000"

def send_normal_predictions(n=20):
    print(f"Sending {n} normal prediction requests...")
    for i in range(n):
        features = [random.uniform(0, 1) for _ in range(23)]
        payload = {"features": features}
        try:
            res = requests.post(f"{EXPORTER_URL}/predict", json=payload, timeout=5)
            if res.status_code == 200:
                data = res.json()
                print(f"  [{i+1}/{n}] Prediction: {data['predicted_class']} | Latency: {data['latency_seconds']}s")
            else:
                print(f"  [{i+1}/{n}] Failed with status code: {res.status_code}")
        except Exception as e:
            print(f"  [{i+1}/{n}] Error: {e}")
        time.sleep(0.1)

def send_high_latency_predictions(n=5):
    print(f"Sending {n} high-latency requests (>2.0s delay)...")
    for i in range(n):
        payload = {"features": [0.5]*23, "simulate_delay": 2.5}
        try:
            res = requests.post(f"{EXPORTER_URL}/predict", json=payload, timeout=10)
            print(f"  [Latency Test {i+1}/{n}] Status: {res.status_code} | Simulated delay 2.5s")
        except Exception as e:
            print(f"  [Latency Test {i+1}/{n}] Error: {e}")

def send_error_predictions(n=6):
    print(f"Sending {n} error requests...")
    for i in range(n):
        payload = {"simulate_error": True}
        try:
            res = requests.post(f"{EXPORTER_URL}/predict", json=payload, timeout=5)
            print(f"  [Error Test {i+1}/{n}] Status: {res.status_code}")
        except Exception as e:
            print(f"  [Error Test {i+1}/{n}] Exception: {e}")

def check_metrics():
    print("Fetching metrics from /metrics endpoint...")
    try:
        res = requests.get(f"{EXPORTER_URL}/metrics", timeout=5)
        lines = [line for line in res.text.split('\n') if line and not line.startswith('#')]
        print(f"Successfully retrieved {len(lines)} Prometheus metric lines.")
        print("Sample active metrics:")
        for sample in lines[:10]:
            print(f"   {sample}")
    except Exception as e:
        print(f"Failed to fetch metrics: {e}")

def main():
    print("Starting inference traffic simulation...")
    send_normal_predictions(15)
    send_high_latency_predictions(3)
    send_error_predictions(6)
    send_normal_predictions(10)
    check_metrics()
    print("Inference simulation completed successfully.")

if __name__ == "__main__":
    main()
