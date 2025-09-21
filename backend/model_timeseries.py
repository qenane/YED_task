import torch
import torch.nn as nn
import numpy as np

# -----------------------------
# Model tanımı
# -----------------------------
class CNN_BiLSTM_Reg(nn.Module):
    def __init__(self, in_dim=3, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden,
            num_layers=layers,
            dropout=(dropout if layers > 1 else 0.0),
            bidirectional=True,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        x = x.transpose(1,2)         # (B, C, L)
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.transpose(1,2)         # (B, L', F)
        out, _ = self.lstm(x)
        yhat = self.head(out)        # (B, L', 1)
        return yhat.squeeze(-1)

# -----------------------------
# Yardımcı fonksiyonlar
# -----------------------------
def downsample_to_length(arr, L=256):
    if len(arr) == 0:
        return np.zeros(L, dtype=np.float32)
    idx = np.linspace(0, len(arr)-1, L)
    return np.interp(idx, np.arange(len(arr)), arr).astype(np.float32)

# -----------------------------
# Model yükleme
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None

def load_timeseries_model(model_path="models/cnn_bilstm_best.pt", in_dim=3, hidden=128, layers=2, dropout=0.3):
    global _model
    model = CNN_BiLSTM_Reg(in_dim=in_dim, hidden=hidden, layers=layers, dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    _model = model
    return _model

# -----------------------------
# Prediction
# -----------------------------
def predict_timeseries(voltage, current, temperature, seq_len=256):
    global _model
    if _model is None:
        raise RuntimeError("Timeseries model yüklenmedi. Önce load_timeseries_model() çağırın.")

    Vd = downsample_to_length(voltage, seq_len)
    Id = downsample_to_length(current, seq_len)
    Td = downsample_to_length(temperature, seq_len)

    X = np.stack([Vd, Id, Td], axis=1)  # (L, 3)
    X = torch.from_numpy(X).unsqueeze(0).to(device)  # (1, L, 3)

    with torch.no_grad():
        yhat = _model(X)                # (1, L’)
        soc_pred = yhat.squeeze(0).cpu().numpy().tolist()  # array döndür

    return soc_pred   # artık her timestep için SOC array

