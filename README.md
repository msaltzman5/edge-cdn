# Edge-Assisted CDN for Video & Sensor Streams

This repo contains a small-scale **edge-assisted CDN prototype** for CSCE 990 (Multi-Access Edge Computing).
It demonstrates how **edge caching, simple popularity prediction, and reinforcement-learning-based prefetch**
can reduce latency and cloud load compared to a cloud-only CDN.

## Architecture

- **Cloud origin**: Nginx serving full content catalog.
- **Edge nodes (3x)**: Nginx caches that fetch from the cloud origin.
- **Orchestrator**: Python container that:
  - Tracks synthetic popularity signals for a few objects.
  - Predicts future popularity using a small LSTM model or EMA fallback.
  - Uses tabular Q-learning to decide whether to prefetch objects to edges.

All services run in a single Docker Compose network on your Ubuntu VM.

## Requirements

- Ubuntu (tested on 22.04)
- Docker
- Docker Compose
- Python 3.10+ (for experiments on the host)

## Setup

```bash
# Clone / init repo
git clone <your-repo-url> edge-cdn
cd edge-cdn

# Create some test content
mkdir -p cloud/content
echo "hello from cloud" > cloud/content/sensor.json
truncate -s 5M cloud/content/video1.mp4
truncate -s 5M cloud/content/video2.mp4

# Create edge cache directory
mkdir -p edge/cache
