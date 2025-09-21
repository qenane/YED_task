# train_seq_soc.py
import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

# -----------------------------
# Dataset
# -----------------------------
class SocWholeDataset(Dataset):
    def __init__(self, df_timeseries, L=256, use_load=False):
        self.L = L
        self.use_load = use_load
        self.rows = []
        for _, r in df_timeseries.iterrows():
            # giriş sinyalleri
            V = self.downsample_to_length(r["voltage_V"], L).astype(np.float32)
            T = self.downsample_to_length(r["temperature_C"], L).astype(np.float32)

            if use_load:
                Vload = self.downsample_to_length(r["voltage_load_V"], L).astype(np.float32)
                Iload = self.downsample_to_length(r["current_load_A"], L).astype(np.float32)
                X = np.stack([V, T, Vload, Iload], axis=1)
            else:
                X = np.stack([V, T], axis=1)

            # hedef = zaman serisi SOC
            y = self.downsample_to_length(r["capacity_Ah"], L).astype(np.float32)
            I = self.downsample_to_length(r["current_A"], L).astype(np.float32)
            y = np.stack([y, I], axis=1)  # (L, 2)  SOC ve Current birlikte

            self.rows.append((X, y))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        X, y = self.rows[idx]
        return torch.from_numpy(X), torch.from_numpy(y)

    def downsample_to_length(self, arr, L=256):
        if len(arr) == 0:
            return np.zeros(L)
        idx = np.linspace(0, len(arr)-1, L)
        return np.interp(idx, np.arange(len(arr)), arr)

# -----------------------------
# CNN + BiLSTM model
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
            nn.Linear(hidden, 1)  # her timestep için çıkış
        )

    def forward(self, x):
        # x: (B, L, C)
        x = x.transpose(1,2)         # (B, C, L)
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.transpose(1,2)         # (B, L', F)

        out, _ = self.lstm(x)        # (B, L', 2H)
        yhat = self.head(out)        # (B, L', 1)
        return yhat.squeeze(-1)

# -----------------------------
# Train / Eval
# -----------------------------
def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        yhat = model(X)
        # yhat ve y uzunlukları eşleşmeli
        min_len = min(yhat.size(1), y.size(1))
        loss = loss_fn(yhat[:, :min_len], y[:, :min_len])
        loss.backward()
        opt.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    ys, yps = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        min_len = min(yp.size(1), y.size(1))
        ys.append(y[:, :min_len].cpu().numpy())
        yps.append(yp[:, :min_len].cpu().numpy())
    y = np.concatenate(ys, axis=0)
    yp = np.concatenate(yps, axis=0)
    rmse = np.sqrt(mean_squared_error(y.ravel(), yp.ravel()))
    return rmse, y, yp

# -----------------------------
# Main
# -----------------------------
def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df_ts = pd.read_pickle(args.timeseries_path).sort_values("datetime").reset_index(drop=True)

    n = len(df_ts)
    cut = int(n * (1 - args.test_size))
    train_df, test_df = df_ts.iloc[:cut], df_ts.iloc[cut:]
    ds = SocWholeDataset(train_df, L=256)



    ds_tr = SocWholeDataset(train_df, L=args.seq_len, use_load=args.use_load)
    ds_te = SocWholeDataset(test_df, L=args.seq_len, use_load=args.use_load)

    train_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False)

    model = CNN_BiLSTM_Reg(in_dim=(5 if args.use_load else 3),
                           hidden=args.hidden,
                           layers=args.layers,
                           dropout=args.dropout).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_rmse = 1e9
    X, y = ds[0]

    # import matplotlib.pyplot as plt
    # plt.plot(train_df.iloc[0]["soc_whole_t"], label="Orijinal SOC")
    # plt.plot(np.linspace(0, len(train_df.iloc[0]["soc_whole_t"]), len(y)), y, label="Downsample SOC")
    # plt.legend()
    # plt.show()
    
    # print("Train set ilk satır SOC başı:", train_df.iloc[0]["soc_whole_t"][0])
    # print("Test set ilk satır SOC başı:", test_df.iloc[0]["soc_whole_t"][0])

        
    
    
    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_loader, opt, loss_fn, device)
        rmse, y_true, y_pred = eval_epoch(model, test_loader, device)
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), outdir / "cnn_bilstm_best.pt")
        if epoch % args.log_every == 0:
            print(f"Epoch {epoch} | TrainLoss={tr_loss:.4f} | Test RMSE={rmse:.4f}")
    import json


    print("Best RMSE:", best_rmse)
    # Son epoch sonunda:
    rmse, y_true, y_pred = eval_epoch(model, test_loader, device)

    results = {
        "best_rmse": float(best_rmse),
        "final_rmse": float(rmse),
        "n_test_cycles": len(test_df),
        "seq_len": args.seq_len,
        "hidden": args.hidden,
        "layers": args.layers
    }

    with open(outdir / "results_timeseries.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Time-series results saved to", outdir / "results_timeseries.json")

    # Örnek çizim
    plt.figure(figsize=(10,5))
    plt.plot(y_true[0], label="True soc_cyclecap_t")
    plt.plot(y_pred[0], label="Predicted soc_cyclecap_t")
    plt.xlabel("Time step")
    plt.ylabel("SOC")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries_path", default="dataset18_timeseries.pkl")
    ap.add_argument("--outdir", default="artifacts_seqsoc")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--use_load", action="store_true")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--log_every", type=int, default=5)
    args = ap.parse_args()
    main(args)
