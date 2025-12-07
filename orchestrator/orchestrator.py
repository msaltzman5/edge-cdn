import time
import requests
from rl_agent import PrefetchQLearning
from lstm_predictor import predict_popularity

EDGES = ["http://edge1:80", "http://edge2:80", "http://edge3:80"]

# Objects we care about
OBJECTS = ["/video1.mp4", "/video2.mp4", "/sensor.json"]

# Keep a simple popularity history per object (sliding window length 10)
WINDOW = 10
pop_history = {obj: [0] * WINDOW for obj in OBJECTS}

agent = PrefetchQLearning()


def simulate_request_counts():
    """
    In a real system, you'd parse logs for counts per interval.
    Here we just simulate some pattern (video1 more popular, etc.).
    """
    import random
    counts = {}
    for obj in OBJECTS:
        if "video1" in obj:
            base = 10
        elif "video2" in obj:
            base = 5
        else:
            base = 2
        counts[obj] = max(0, int(random.gauss(base, base * 0.3)))
    return counts


def prefetch_object(path):
    """Issue a GET to all edges to warm their caches."""
    for edge in EDGES:
        url = edge + path
        try:
            r = requests.get(url, timeout=3)
            print(f"[PREFETCH] {url} -> {r.status_code}")
        except Exception as e:
            print(f"[PREFETCH-ERROR] {url}: {e}")


def main_loop(interval=10.0):
    print("Starting orchestrator main loop...")
    step = 0
    while True:
        step += 1
        print(f"\n=== Interval {step} ===")

        # Update synthetic per-interval request counts
        new_counts = simulate_request_counts()
        print("Simulated counts:", new_counts)

        for obj in OBJECTS:
            hist = pop_history[obj]
            hist.append(new_counts[obj])
            if len(hist) > WINDOW:
                hist.pop(0)
            pop_history[obj] = hist

            pred = predict_popularity(hist)  # scalar
            # Discretize prediction to small integer state
            state = int(round(pred))

            action = agent.choose_action(state)
            print(f"[DECISION] obj={obj}, pred={pred:.2f}, state={state}, action={action}")

            # Simple reward: if popularity is high, we benefit from prefetching
            if action == "prefetch":
                prefetch_object(obj)
                reward = 1.0 if pred > 5 else -0.2
            else:
                reward = -0.5 if pred > 5 else 0.1

            # For now, next_state == state (no explicit transition model)
            agent.update(state, action, reward, state)

        time.sleep(interval)


if __name__ == "__main__":
    main_loop()
