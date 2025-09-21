# train_lstm.py
import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def chronological_indices(n, test_size=0.3):
    cut = int(n * (1 - test_size))
    idx_train = np.arange(0, cut)
    idx_test  = np.arange(cut, n)
    return idx_train, idx_test

def downsample_to_length(arr, L=256):
    # lineer örnekleme (basit)
    if len(arr) == 0:
        return np.zeros(L)
    idx = np.linspace(0, len(arr)-1, L)
    return np.interp(idx, np.arange(len(arr)), arr)

class SocDataset(Dataset):
    def __init__(self, df_timeseries, L=256, use_load=False):
        self.L = L
        self.use_load = use_load
        self.rows = []
        for _, r in df_timeseries.iterrows():
            V = downsample_to_length(r["voltage_V"], L).astype(np.float32)
            I = downsample_to_length(r["current_A"], L).astype(np.float32)
            T = downsample_to_length(r["temperature_C"], L).astype(np.float32)
            if use_load:
                Vload = downsample_to_length(r["voltage_load_V"], L).astype(np.float32)
                Iload = downsample_to_length(r["current_load_A"], L).astype(np.float32)
                X = np.stack([V, I, T, Vload, Iload], axis=1)  # (L, C)
            else:
                X = np.stack([V, I, T], axis=1)
            y = np.array([r["soc_cycle"]], dtype=np.float32)
            self.rows.append((X, y))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        X, y = self.rows[idx]
        return torch.from_numpy(X), torch.from_numpy(y)

class LSTMReg(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden,
                            num_layers=layers, batch_first=True, dropout=(dropout if layers>1 else 0.0))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)      # (B, L, H)
        h_last = out[:, -1, :]     # son zaman adımı
        return self.head(h_last)

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0.0
    for X, y in loader:
        X = X.to(device); y = y.to(device)
        opt.zero_grad()
        yhat = model(X)
        loss = loss_fn(yhat, y)
        loss.backward()
        opt.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    ys, yps = [], []
    for X, y in loader:
        X = X.to(device); y = y.to(device)
        yp = model(X)
        ys.append(y.cpu().numpy())
        yps.append(yp.cpu().numpy())
    y = np.concatenate(ys).ravel()
    yp = np.concatenate(yps).ravel()
    mae = mean_absolute_error(y, yp)
    mse = mean_squared_error(y, yp); rmse = np.sqrt(mse)
    r2  = r2_score(y, yp)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}, y, yp

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df_ts = pd.read_pickle(args.timeseries_path)
    # kronolojik split
    df_ts = df_ts.sort_values("datetime").reset_index(drop=True)
    idx_tr, idx_te = chronological_indices(len(df_ts), test_size=args.test_size)
    ds_tr = SocDataset(df_ts.iloc[idx_tr], L=args.seq_len, use_load=args.use_load)
    ds_te = SocDataset(df_ts.iloc[idx_te], L=args.seq_len, use_load=args.use_load)

    in_dim = (5 if args.use_load else 3)
    model = LSTMReg(in_dim, hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(device)

    train_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best = {"rmse": 1e9}
    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_loader, opt, loss_fn, device)
        metrics, y_true, y_pred = eval_epoch(model, test_loader, device)
        if metrics["rmse"] < best["rmse"]:
            best = {**metrics, "epoch": epoch}
            torch.save(model.state_dict(), outdir / "lstm_best.pt")
        if epoch % max(1, args.log_every) == 0:
            print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | "
                  f"val: MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f}")

    # son metrikleri kaydet
    (outdir / "last_metrics.json").write_text(json.dumps(best, indent=2))
    print("Best:", best, "saved to", outdir)
    
    
    metrics, y_true, y_pred = eval_epoch(model, test_loader, device)

    # Tahminleri DataFrame'e koy
    results = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })
    print(results.head())

    # Çizim
    plt.figure(figsize=(10,6))
    plt.plot(y_true, label="True SOC_cycle")
    plt.plot(y_pred, label="Predicted SOC_cycle")
    plt.xlabel("Test Sample Index")
    plt.ylabel("SOC_cycle")
    plt.title("LSTM Predictions vs True")
    plt.legend()
    plt.show()

    # Scatter
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("True SOC_cycle")
    plt.ylabel("Predicted SOC_cycle")
    plt.title("LSTM: True vs Predicted")
    plt.plot([0,1],[0,1], color="red", linestyle="--")
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries_path", default="dataset_timeseries.pkl")
    ap.add_argument("--outdir", default="artifacts_lstm")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--use_load", action="store_true") 
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--log_every", type=int, default=5)
    args = ap.parse_args()
    main(args)
