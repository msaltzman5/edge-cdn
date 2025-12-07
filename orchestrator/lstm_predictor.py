import os
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = "popularity.pt"

class PopularityLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def _load_model():
    model = PopularityLSTM()
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            print("Loaded LSTM popularity model.")
            return model
        except Exception as e:
            print("Failed to load LSTM model, falling back to EMA:", e)
    print("No LSTM model found; using EMA fallback.")
    return None


MODEL = _load_model()


def predict_popularity(history):
    """
    history: list of recent request counts for an object.
    returns: scalar predicted next-step popularity.
    """
    if not history:
        return 0.0

    arr = np.array(history, dtype=float)

    if MODEL is None:
        # Exponential moving average fallback
        alpha = 0.5
        ema = arr[0]
        for v in arr[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return float(ema)

    # LSTM path
    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        y = MODEL(x)
    return float(y.item())
