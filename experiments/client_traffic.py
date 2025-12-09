import time
import random
import statistics
import subprocess

CLOUD_URL = "http://localhost:9000"
EDGE_NODES = {
    "edge1": "http://localhost:8001",
    "edge2": "http://localhost:8002",
    "edge3": "http://localhost:8003",
}

MP4_VIDEOS = [f"/video{i}.mp4" for i in range(1, 6)]
OBJECTS = MP4_VIDEOS + ["/sensor.json"]

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


def run_experiment(base_url, label, num_requests=100, delay_s=0.1):
    latencies = []

    for i in range(num_requests):
        obj = random.choice(OBJECTS)
        url = base_url + obj
        t = measure_request(url)
        latencies.append(t)
        print(f"[{label.upper()}] {i+1}/{num_requests} {obj} -> {t:.4f}s")
        time.sleep(delay_s)

    avg = statistics.mean(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18]  # approximate 95th
    print(f"\n[{label.upper()}] avg={avg:.4f}s, p95={p95:.4f}s\n")


if __name__ == "__main__":
    print("=== Baseline (cloud-only) ===")
    run_experiment(base_url=CLOUD_URL, label="cloud", num_requests=50)

    print("\n=== Edge-assisted (per node) ===")
    for label, base_url in EDGE_NODES.items():
        run_experiment(base_url=base_url, label=label, num_requests=50)
