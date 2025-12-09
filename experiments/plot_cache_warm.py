import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

# URLs to compare (make sure docker-compose is up)
EDGE_URL = "http://localhost:8001"
CLOUD_URL = "http://localhost:9000"

# Choose objects the orchestrator is unlikely to prefetch to keep the first hit cold
OBJECTS = [
    "/video10.mp4",
    "/video20.mp4",
    "/sensor10.json",
    "/sensor20.json",
]


def fetch_latency(url, pause_s=0.2):
    """
    Fetch a URL, return (latency_seconds, cache_status_header).
    pause_s allows the cache to settle between back-to-back requests.
    """
    start = time.perf_counter()
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    _ = resp.content  # read body to include transfer time
    elapsed = time.perf_counter() - start
    cache_hdr = resp.headers.get("X-Cache-Status", "n/a")
    time.sleep(pause_s)
    return elapsed, cache_hdr


def measure_first_vs_cached(base_url, objects):
    """Measure first (cold) vs second (cached) fetch times for each object."""
    results = {}
    for obj in objects:
        url = base_url + obj
        cold, cold_hdr = fetch_latency(url)
        warm, warm_hdr = fetch_latency(url)
        results[obj] = {
            "cold": cold,
            "warm": warm,
            "cold_status": cold_hdr,
            "warm_status": warm_hdr,
        }
        print(f"{base_url} {obj}: cold={cold:.3f}s ({cold_hdr}) warm={warm:.3f}s ({warm_hdr})")
    return results


def warmup_series(base_url, obj, repeats=6):
    """Repeatedly fetch the same object to show warm-up over a short series."""
    url = base_url + obj
    times = []
    statuses = []
    for i in range(repeats):
        t, hdr = fetch_latency(url)
        times.append(t)
        statuses.append(hdr)
        print(f"{base_url} {obj} hit {i+1}: {t:.3f}s ({hdr})")
    return times, statuses


def plot_bar(results, out_path):
    objects = list(results.keys())
    cold = [results[o]["cold"] for o in objects]
    warm = [results[o]["warm"] for o in objects]

    x = range(len(objects))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], cold, width, label="First fetch (cold miss)")
    ax.bar([i + width / 2 for i in x], warm, width, label="Second fetch (cached)")

    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Cold vs Cached Fetch Latency per Object (edge)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(objects, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved bar plot to {out_path}")


def plot_hist(cold_vals, warm_vals, out_path):
    bins = np.linspace(0, max(cold_vals + warm_vals) * 1.1, 15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cold_vals, bins=bins, alpha=0.6, label="Cold")
    ax.hist(warm_vals, bins=bins, alpha=0.6, label="Cached")
    ax.set_xlabel("Latency (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Latency Distribution: Cold vs Cached (edge)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved histogram to {out_path}")


def plot_edge_vs_cloud(edge_results, cloud_results, out_path):
    labels = ["Edge cold", "Edge cached", "Cloud first", "Cloud second"]
    vals = [
        statistics.mean([edge_results[o]["cold"] for o in edge_results]),
        statistics.mean([edge_results[o]["warm"] for o in edge_results]),
        statistics.mean([cloud_results[o]["cold"] for o in cloud_results]),
        statistics.mean([cloud_results[o]["warm"] for o in cloud_results]),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, vals, color=["#cc5500", "#2a9d8f", "#3a86ff", "#9d4edd"])
    ax.set_ylabel("Avg latency (seconds)")
    ax.set_title("Edge vs Cloud Average Latency")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.autofmt_xdate(rotation=15)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved edge-vs-cloud plot to {out_path}")


def plot_warmup(times, statuses, out_path):
    hits = list(range(1, len(times) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hits, times, marker="o")
    for x, y, status in zip(hits, times, statuses):
        ax.text(x, y, status, ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Hit number")
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Warm-up for repeated hits on one object (edge)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved warm-up series plot to {out_path}")


if __name__ == "__main__":
    # Measure edge cold vs cached per object
    edge_results = measure_first_vs_cached(EDGE_URL, OBJECTS)

    # Measure cloud twice for comparison (no cache, but provides baseline)
    cloud_results = measure_first_vs_cached(CLOUD_URL, OBJECTS)

    # Show how latency drops as cache warms on one object
    warm_times, warm_statuses = warmup_series(EDGE_URL, OBJECTS[0], repeats=6)

    out_dir = Path(__file__).parent
    plot_bar(edge_results, out_dir / "cache_bar.png")

    all_cold = [edge_results[o]["cold"] for o in OBJECTS]
    all_warm = [edge_results[o]["warm"] for o in OBJECTS]
    plot_hist(all_cold, all_warm, out_dir / "cache_hist.png")

    plot_edge_vs_cloud(edge_results, cloud_results, out_dir / "edge_vs_cloud.png")

    plot_warmup(warm_times, warm_statuses, out_dir / "cache_warmup_series.png")
