import argparse
import json
import re
import io
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from pymongo import MongoClient
import gridfs

# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Dataset
# ============================================================
class CurrCapDataset(Dataset):
    def __init__(self, df, L=256, use_load=False, use_time=True, use_cap=False):
        self.L = L
        self.use_load = use_load
        self.use_time = use_time
        self.use_cap = use_cap
        self.rows = []

        first_dt = df["datetime"].min()
        df = df.copy()
        df["days_since_start"] = (df["datetime"] - first_dt).dt.days

        has_vload = "voltage_load_V" in df.columns
        has_iload = "current_load_A" in df.columns
        warned_missing_load = False

        for _, r in df.iterrows():
            V = self.downsample(np.asarray(r["voltage_V"], dtype=np.float32), L)
            T = self.downsample(np.asarray(r["temperature_C"], dtype=np.float32), L)

            feats = [V, T]

            if use_load:
                if has_vload and has_iload:
                    Vload = self.downsample(np.asarray(r["voltage_load_V"], dtype=np.float32), L)
                    Iload = self.downsample(np.asarray(r["current_load_A"], dtype=np.float32), L)
                else:
                    if not warned_missing_load:
                        print("⚠️  use_load=True ama voltage_load_V/current_load_A sütunları yok; sıfırlarla doldurulacak.")
                        warned_missing_load = True
                    Vload = np.zeros(L, dtype=np.float32)
                    Iload = np.zeros(L, dtype=np.float32)
                feats += [Vload, Iload]

            if self.use_time:
                t = self.downsample(np.asarray(r["time_s"], dtype=np.float32), L)
                feats.append(t / (t.max() + 1e-6))  # normalize

            if self.use_cap:
                cap_val = float(r["capacity_Ah"])
                cap_feat = np.full(L, cap_val, dtype=np.float32)
                feats.append(cap_feat)

            X = np.stack(feats, axis=1).astype(np.float32)

            I = self.downsample(np.asarray(r["current_A"], dtype=np.float32), L).astype(np.float32)
            cap = np.float32(r["capacity_Ah"])  # scalar target

            t_ds = self.downsample(np.asarray(r["time_s"], dtype=np.float32), L)
            self.rows.append((X, I, cap, t_ds))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        X, I, cap, time_s = self.rows[idx]
        return torch.from_numpy(X), torch.from_numpy(I), torch.tensor(cap), time_s

    @staticmethod
    def downsample(arr, L):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            return np.zeros(L, dtype=np.float32)
        idx = np.linspace(0, max(len(arr) - 1, 0), L)
        return np.interp(idx, np.arange(len(arr)), arr).astype(np.float32)


# ============================================================
# Model
# ============================================================
class CNN_BiLSTM_CurrCap(nn.Module):
    def __init__(self, in_dim=3, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        # self.pool = nn.MaxPool1d(2)  # İstersen ekleyebilirsin; şu an kullanılmıyor

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden,
            num_layers=layers,
            dropout=(dropout if layers > 1 else 0.0),
            bidirectional=True,
            batch_first=True,
        )

        self.head_I = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),  # I per timestep
        )
        self.head_cap = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),  # cycle capacity
        )

    def forward(self, x):
        # x: (B, L, C)
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (B, L, F)

        out, _ = self.lstm(x)  # (B, L, 2H)

        I_hat = self.head_I(out).squeeze(-1)   # (B, L)
        cap_hat = self.head_cap(out[:, -1])    # (B, 1)
        return I_hat, cap_hat.squeeze(-1)


# ============================================================
# Train / Eval
# ============================================================
def train_epoch(model, loader, opt, loss_fn, dev, alpha=10.0):
    model.train()
    total = 0.0
    for X, I, cap, _ in loader:
        X, I, cap = X.to(dev), I.to(dev), cap.to(dev)
        opt.zero_grad()
        I_hat, cap_hat = model(X)
        loss_I = loss_fn(I_hat, I)
        loss_cap = loss_fn(cap_hat, cap)
        loss = loss_I + alpha * loss_cap
        loss.backward()
        opt.step()
        total += loss.item() * X.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, dev, alpha=10.0):
    model.eval()
    Is, I_preds, caps, cap_preds = [], [], [], []
    for X, I, cap, _ in loader:
        X, I, cap = X.to(dev), I.to(dev), cap.to(dev)
        I_hat, cap_hat = model(X)
        Is.append(I.cpu().numpy())
        I_preds.append(I_hat.cpu().numpy())
        caps.append(cap.cpu().numpy())
        cap_preds.append(cap_hat.cpu().numpy())
    I = np.concatenate(Is, axis=0)
    I_pred = np.concatenate(I_preds, axis=0)
    caps = np.concatenate(caps, axis=0)
    cap_preds = np.concatenate(cap_preds, axis=0)
    rmse_I = np.sqrt(mean_squared_error(I.ravel(), I_pred.ravel()))
    rmse_cap = np.sqrt(mean_squared_error(caps, cap_preds))
    return rmse_I, rmse_cap, I, I_pred, caps, cap_preds


def collate_fn(batch):
    X, I, cap, time_s = zip(*batch)
    return torch.stack(X), torch.stack(I), torch.stack(cap), list(time_s)


# ============================================================
# SOC hesaplama
# ============================================================
def compute_soc(I_pred, cap_pred, time_s, first_cap):
    dt_arr = np.diff(time_s, prepend=time_s[0])
    sign = -1.0 if np.nanmedian(I_pred) < 0 else 1.0
    q_drawn_ah = np.cumsum(sign * I_pred * dt_arr) / 3600.0

    soc_cycle = cap_pred / first_cap
    soc_whole_t = np.clip(soc_cycle - q_drawn_ah / first_cap, 0, 1.05)
    soc_cyclecap_t = np.clip(1 - q_drawn_ah / cap_pred, 0, 1.05)
    return soc_cycle, soc_whole_t, soc_cyclecap_t


# ============================================================
# Mongo helpers
# ============================================================
def load_from_mongo(client: MongoClient, db_name: str, coll_name: str) -> pd.DataFrame:
    coll = client[db_name][coll_name]
    docs = list(coll.find({}))
    if not docs:
        raise ValueError(f"{coll_name} koleksiyonunda veri yok!")
    df = pd.DataFrame(docs)
    if not np.issubdtype(df["datetime"].dtype, np.datetime64):
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def _infer_arch_from_state(state_dict):
    """Infer in_dim, hidden, layers from a PyTorch state_dict."""
    in_dim = int(state_dict["conv1.weight"].shape[1])  # (32, in_dim, k)
    hidden = int(state_dict["lstm.weight_ih_l0"].shape[0] // 4)  # (4H, input)
    layer_idxs = {0}
    pat = re.compile(r"^lstm\.weight_ih_l(\d+)")
    for k in state_dict.keys():
        m = pat.match(k)
        if m:
            layer_idxs.add(int(m.group(1)))
    layers = max(layer_idxs) + 1
    return in_dim, hidden, layers


def load_model_from_mongo(db, filename, device_override=None):
    dev = device_override or device
    _fs = gridfs.GridFS(db)
    file = _fs.find_one({"filename": filename})
    if not file:
        raise FileNotFoundError(f"{filename} MongoDB'de bulunamadı")

    raw = file.read()
    buffer = io.BytesIO(raw); buffer.seek(0)

    state_dict = torch.load(buffer, map_location=dev)

    meta = getattr(file, "metadata", None) or {}
    meta_in_dim = meta.get("in_dim")
    meta_hidden = meta.get("hidden")
    meta_layers = meta.get("layers")
    dropout = float(meta.get("dropout", 0.3))

    inf_in_dim, inf_hidden, inf_layers = _infer_arch_from_state(state_dict)

    use_meta = (
        meta_in_dim is not None and meta_hidden is not None and meta_layers is not None
        and int(meta_in_dim) == inf_in_dim
        and int(meta_hidden) == inf_hidden
        and int(meta_layers) == inf_layers
    )

    if use_meta:
        in_dim, hidden, layers = int(meta_in_dim), int(meta_hidden), int(meta_layers)
    else:
        if meta and (meta_in_dim is not None or meta_hidden is not None or meta_layers is not None):
            print(
                f"⚠️  Metadata/state_dict uyumsuz (meta: in_dim={meta_in_dim}, hidden={meta_hidden}, layers={meta_layers} | "
                f"inferred: in_dim={inf_in_dim}, hidden={inf_hidden}, layers={inf_layers}). Inferred kullanılacak."
            )
        in_dim, hidden, layers = inf_in_dim, inf_hidden, inf_layers

    print(f"📦 Loading weights from GridFS: {filename} | in_dim={in_dim}, hidden={hidden}, layers={layers}")

    model = CNN_BiLSTM_CurrCap(in_dim=in_dim, hidden=hidden, layers=layers, dropout=dropout).to(dev)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def save_model_to_mongo(
    db,
    model,
    in_dim,
    hidden,
    layers,
    dropout,
    name="cnn_bilstm_currcap_best",
    with_cap=False,
    use_load=False,
    train_sets=None,
    test_set=None,
    metrics=None,
):
    """Modeli GridFS'e metadata ile kaydet (hedef DB'ye)."""
    _fs = gridfs.GridFS(db)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer); buffer.seek(0)

    filename = f"{name}{'_w_cap' if with_cap else ''}{'_ul' if use_load else ''}.pt"

    # aynı isimde varsa temizle
    existing = _fs.find({"filename": filename})
    for f in existing:
        _fs.delete(f._id)

    meta = {
        "model": name,
        "with_cap": with_cap,
        "in_dim": int(in_dim),
        "hidden": int(hidden),
        "layers": int(layers),
        "dropout": float(dropout),
        "use_load": bool(use_load),
        "train_sets": train_sets or [],
        "test_set": test_set,
        "metrics": metrics or {},
    }
    file_id = _fs.put(buffer.read(), filename=filename, metadata=meta)
    print(f"Model MongoDB'ye kaydedildi: {filename}, id={file_id}")
    return file_id


# ============================================================
# Utils
# ============================================================
def ensure_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, list):
        return x
    else:
        return [x]


# ============================================================
# Main training
# ============================================================
def main(args):
    dev = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mongo_client = MongoClient(args.mongo_uri)
    mongo_db = mongo_client[args.db_name]

    # Load train sets
    train_dfs = []
    for coll in args.train_colls:
        df_tmp = load_from_mongo(mongo_client, args.db_name, coll)
        train_dfs.append(df_tmp)
    train_df = pd.concat(train_dfs, ignore_index=True)

    # Load test set
    test_df = load_from_mongo(mongo_client, args.db_name, args.test_coll)

    # Datasets & Loaders
    ds_tr = CurrCapDataset(train_df, L=args.seq_len, use_load=args.use_load, use_time=True, use_cap=args.load_cap)
    ds_te = CurrCapDataset(test_df, L=args.seq_len, use_load=args.use_load, use_time=True, use_cap=args.load_cap)

    train_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Input feature dim
    in_dim = 3  # V, T, time
    if args.use_load:
        in_dim += 2
    if args.load_cap:
        in_dim += 1

    model = CNN_BiLSTM_CurrCap(in_dim=in_dim, hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_score = float("inf")
    best_rmse_I = None
    best_rmse_cap = None
    best_state = None
    best_model_name = None

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, opt, loss_fn, dev, alpha=args.alpha)
        rmse_I, rmse_cap, *_ = eval_epoch(model, test_loader, dev)
        score = float(rmse_I) + args.alpha * float(rmse_cap)

        if score < best_score:
            best_score = score
            best_rmse_I = float(rmse_I)
            best_rmse_cap = float(rmse_cap)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            best_model_name = "cnn_bilstm_currcap_best"
            if args.load_cap:
                best_model_name += "_w_cap"
            if args.use_load:
                best_model_name += "_ul"

            torch.save(model.state_dict(), outdir / f"{best_model_name}.pt")

        if epoch % args.log_every == 0:
            print(f"Epoch {epoch} | TrainLoss={tr_loss:.4f} | RMSE_I={rmse_I:.4f} | RMSE_cap={rmse_cap:.4f}")

    # Eğitim bitti: en iyi ağırlıkları geri yükle
    if best_state is None:
        # hiçbir iyileşme olmadıysa mevcut haliyle devam
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_rmse_I, best_rmse_cap, *_ = eval_epoch(model, test_loader, dev)

    model.load_state_dict(best_state, strict=True)
    print("Best combined score:", best_score)

    # Metrikler (best)
    metrics_doc = {
        "rmse_I": float(best_rmse_I),
        "rmse_cap": float(best_rmse_cap),
        "best_score": float(best_score),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

    # Split metadata
    def _norm_name(x):
        return x.replace("_timeseries", "").replace("_processed", "")

    train_sets = sorted([_norm_name(c) for c in args.train_colls])
    test_set = _norm_name(args.test_coll)

    # Model’i GridFS’e metadata ile kaydet
    save_model_to_mongo(
        db=mongo_db,
        model=model,
        in_dim=in_dim,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        name="cnn_bilstm_currcap_best",
        with_cap=args.load_cap,
        use_load=args.use_load,
        train_sets=train_sets,
        test_set=test_set,
        metrics=metrics_doc,
    )

    # İsteğe bağlı: ayrıca registry koleksiyonuna yaz
    mongo_db["model_metrics"].insert_one({
        "model_name": (best_model_name or "cnn_bilstm_currcap_best"
                       + ("_w_cap" if args.load_cap else "")
                       + ("_ul" if args.use_load else "")),
        "train_sets": train_sets,
        "test_set": test_set,
        "metrics": metrics_doc,
        "saved_at": datetime.datetime.utcnow()
    })

    # ========================================================
    # Test sonuçlarını BEST model ile JSON/CSV yaz
    # ========================================================
    results = []
    with torch.no_grad():
        for X, I, cap, time_s in test_loader:
            X = X.to(dev)
            I_hat, cap_hat = model(X)

            for b in range(len(X)):
                I_pred = I_hat[b].cpu().numpy().tolist()
                cap_pred = float(cap_hat[b].cpu().numpy())
                time_arr = time_s[b].tolist()
                first_cap = float(train_df.iloc[0]["capacity_Ah"])  # reference

                soc_cycle, soc_whole_t_pred, soc_cyclecap_t_pred = compute_soc(
                    np.array(I_pred), cap_pred, np.array(time_arr), first_cap
                )

                gt_row = test_df.iloc[b]

                def _safe_get(row, key, default):
                    try:
                        val = row[key]
                        return ensure_list(val)
                    except Exception:
                        return ensure_list(default)

                cap_true = float(gt_row.get("capacity_Ah", cap_pred))

                results.append({
                    "cycle_id": int(gt_row.get("cycle_id", b)),
                    "cap_true": float(cap_true),
                    "cap_pred": float(cap_pred),
                    "soc_whole_t_true": _safe_get(gt_row, "soc_whole_t", soc_whole_t_pred),
                    "soc_whole_t_pred": ensure_list(soc_whole_t_pred),
                    "soc_cyclecap_t_true": _safe_get(gt_row, "soc_cyclecap_t", soc_cyclecap_t_pred),
                    "soc_cyclecap_t_pred": ensure_list(soc_cyclecap_t_pred),
                    "I_true": _safe_get(gt_row, "current_A", I_pred),
                    "I_pred": ensure_list(I_pred),
                    "time_s": ensure_list(time_arr),
                })

    json_path = outdir / "test_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Tüm test sonuçları JSON olarak kaydedildi: {json_path}")

    csv_path = outdir / "test_results.csv"
    flat_records = []
    for r in results:
        flat_records.append({
            "cycle_id": r["cycle_id"],
            "cap_true": r["cap_true"],
            "cap_pred": r["cap_pred"],
            "soc_whole_t_true": json.dumps(r["soc_whole_t_true"]),
            "soc_whole_t_pred": json.dumps(r["soc_whole_t_pred"]),
            "soc_cyclecap_t_true": json.dumps(r["soc_cyclecap_t_true"]),
            "soc_cyclecap_t_pred": json.dumps(r["soc_cyclecap_t_pred"]),
        })
    pd.DataFrame(flat_records).to_csv(csv_path, index=False)
    print(f"Tüm test sonuçları CSV olarak kaydedildi: {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mongo_uri", default="mongodb://localhost:27017", help="MongoDB URI")
    ap.add_argument("--db_name", required=True, help="Database ismi")
    ap.add_argument("--train_colls", nargs="+", required=True, help="Birden fazla train koleksiyon ismi")
    ap.add_argument("--test_coll", required=True, help="Test koleksiyon ismi")
    ap.add_argument("--outdir", default="artifacts_seqsoc")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--use_load", action="store_true")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--load_cap", action="store_true", help="Capacity değerini input feature olarak ekle")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=10.0)
    args = ap.parse_args()
    main(args)
