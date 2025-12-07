import time
import random
import statistics
import subprocess

EDGE_URL = "http://localhost:8001"
CLOUD_URL = "http://localhost:9000"

OBJECTS = ["/video1.mp4", "/video2.mp4", "/sensor.json"]

# Curl format template to capture total time
CURL_FMT = """
 time_namelookup:  %{time_namelookup}\\n
 time_connect:  %{time_connect}\\n
 time_appconnect:  %{time_appconnect}\\n
 time_pretransfer:  %{time_pretransfer}\\n
 time_starttransfer:  %{time_starttransfer}\\n
 ----------
 time_total:  %{time_total}\\n
"""


def measure_request(url):
    cmd = [
        "curl", "-o", "/dev/null", "-s", "-w", CURL_FMT, url
    ]
    out = subprocess.check_output(cmd, text=True)
    for line in out.splitlines():
        if "time_total" in line:
            val = float(line.split()[-1])
            return val
    return None


def run_experiment(num_requests=100, target="edge"):
    base = EDGE_URL if target == "edge" else CLOUD_URL
    latencies = []

    for i in range(num_requests):
        obj = random.choice(OBJECTS)
        url = base + obj
        t = measure_request(url)
        latencies.append(t)
        print(f"[{target.upper()}] {i+1}/{num_requests} {obj} -> {t:.4f}s")
        time.sleep(0.1)

    avg = statistics.mean(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18]  # approximate 95th
    print(f"\n[{target.upper()}] avg={avg:.4f}s, p95={p95:.4f}s")


if __name__ == "__main__":
    print("=== Baseline (cloud-only) ===")
    run_experiment(num_requests=50, target="cloud")

    print("\n=== Edge-assisted (edge1) ===")
    run_experiment(num_requests=50, target="edge")
