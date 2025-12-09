import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PopularityLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def synth_trace(seq_len=20):
    """Generate a single synthetic popularity trace (non-stationary + noise)."""
    level = np.random.uniform(2, 40)
    trend = np.random.uniform(-0.4, 0.6)
    season_amp = np.random.uniform(0, 8)
    noise = np.random.normal(0, 2, size=seq_len + 1)
    ticks = np.arange(seq_len + 1)
    season = season_amp * np.sin(ticks / np.random.uniform(4, 10))
    raw = level + trend * ticks + season + noise
    raw = np.maximum(raw, 0.0)
    return raw.astype(np.float32)


def make_dataset(num_series=1000, seq_len=20):
    X, y = [], []
    for _ in range(num_series):
        trace = synth_trace(seq_len)
        X.append(trace[:seq_len])
        y.append(trace[seq_len])
    X = torch.tensor(np.array(X)).unsqueeze(-1)
    y = torch.tensor(np.array(y)).unsqueeze(-1)
    return TensorDataset(X, y)


def train_model(epochs=5, batch_size=64, lr=1e-3, seq_len=20):
    ds = make_dataset(num_series=2000, seq_len=seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = PopularityLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total = 0.0
        count = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            count += len(xb)
        print(f"epoch={epoch+1} mse={total / count:.4f}")
    return model


model = train_model()

# Resolve to repo root/orchestrator/popularity.pt regardless of cwd
repo_root = Path(__file__).resolve().parents[1]
out_dir = repo_root / "orchestrator"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "popularity.pt"

torch.save(model.state_dict(), out_path)
print(f"Saved model to {out_path}")
