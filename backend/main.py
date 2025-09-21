from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import os, sys
from functools import lru_cache

# Proje yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_curr_cap import CNN_BiLSTM_CurrCap, compute_soc, load_model_from_mongo

app = FastAPI(title="Battery SOC Prediction API")

# --- MQTT --- #
@app.on_event("startup")
def _start_mqtt():
    try:
        from mqtt_integration import start_mqtt
        start_mqtt()
    except Exception as e:
        # MQTT opsiyonel; hata verirse API yine ayağa kalksın
        print("[MQTT] startup skipped/error:", repr(e))

# --- CORS --- #
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# --- Mongo --- #
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
MONGO_DB  = os.getenv("MONGO_DB", "battery_db")  # 🔴 ÖNEMLİ: eğitimdeki DB ile aynı olsun
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model cache --- #
@lru_cache(maxsize=8)
def _get_model_by_filename(filename: str):
    # load_model_from_mongo, dosya metadata'sından in_dim/hidden/layers/… okur
    model = load_model_from_mongo(db, filename)
    return model

def get_model(mode: str, use_load: bool):
    filename = "cnn_bilstm_currcap_best"
    if mode == "currcap_w_cap":
        filename += "_w_cap"
    if use_load:
        filename += "_ul"
    filename += ".pt"
    return _get_model_by_filename(filename)

# --- Schemas --- #
class PredictRequest(BaseModel):
    dataset: str
    cycle_id: int
    mode: str      # "currcap" | "currcap_w_cap"
    use_load: bool

# --- Helpers --- #
def _as_f32(arr, like_len=None):
    """None/eksik gelirse güvenli sıfır dizisi üret."""
    if arr is None:
        if like_len is None:
            return np.zeros(0, dtype=np.float32)
        return np.zeros(like_len, dtype=np.float32)
    a = np.array(arr, dtype=np.float32)
    if like_len is not None and a.size == 0:
        return np.zeros(like_len, dtype=np.float32)
    return a

def _model_filename(mode: str, use_load: bool):
    base = "cnn_bilstm_currcap_best"
    if mode == "currcap_w_cap":
        base += "_w_cap"
    if use_load:
        base += "_ul"
    return base + ".pt"

def _get_gridfs_file_by_name(db, filename):
    fs = gridfs.GridFS(db)
    return fs.find_one({"filename": filename})

def _list_all_datasets(db):
    cols = db.list_collection_names()
    ds = set()
    for c in cols:
        if c.endswith("_processed"): ds.add(c[:-10])
        elif c.endswith("_timeseries"): ds.add(c[:-11])
    return sorted(ds)


from fastapi import Query

@app.get("/metrics")
def get_metrics(
    mode: str = Query("currcap"),
    use_load: bool = Query(False)
):
    fname = _model_filename(mode, use_load)
    gf = _get_gridfs_file_by_name(db, fname)
    if not gf:
        return {"error": f"{fname} bulunamadı"}
    meta = gf.metadata or {}
    out = {
        "model_file": fname,
        "train_sets": meta.get("train_sets", []),
        "test_set": meta.get("test_set"),
        "metrics": meta.get("metrics", {})
    }
    return out


# --- Endpoints --- #
@app.post("/predict")
def predict_soc(req: PredictRequest):
    try:
        print("📌 Step 1: Request alındı")

        collection_name = f"{req.dataset}_timeseries"
        doc = db[collection_name].find_one({"cycle_id": req.cycle_id})
        if not doc:
            raise HTTPException(status_code=404, detail=f"{collection_name} içinde cycle {req.cycle_id} yok")

        # features (seçili cycle)
        V = _as_f32(doc.get("voltage_V"))
        T = _as_f32(doc.get("temperature_C"), like_len=len(V))
        time_s = _as_f32(doc.get("time_s"), like_len=len(V))
        cap_true = float(doc["capacity_Ah"])

        feats = [V, T, time_s / (time_s.max() + 1e-6)]
        if req.mode == "currcap_w_cap":
            feats.append(np.full_like(V, cap_true, dtype=np.float32))
        if req.use_load:
            Vload = _as_f32(doc.get("voltage_load_V"), like_len=len(V))
            Iload = _as_f32(doc.get("current_load_A"), like_len=len(V))
            feats += [Vload, Iload]

        X = np.stack(feats, axis=1).astype(np.float32)  # (L, C)
        X = torch.from_numpy(X).unsqueeze(0).to(device) # (1, L, C)

        model = get_model(req.mode, req.use_load)

        print("📌 Step 2: Mongo'dan veri çekildi")
        print("📌 Step 3: Features hazırlandı, shape:", X.shape)

        with torch.no_grad():
            I_hat, cap_hat = model(X)

        I_pred = I_hat.squeeze(0).detach().cpu().numpy()
        cap_pred = float(cap_hat.detach().cpu().numpy().item())

        # --- Tarih boyunca TRUE & PRED soc_cycle çizgileri --- #
        history_docs = list(
            db[collection_name].find(
                {},
                {
                    "cycle_id": 1, "datetime": 1, "capacity_Ah": 1,
                    "voltage_V": 1, "temperature_C": 1, "time_s": 1,
                    "voltage_load_V": 1, "current_load_A": 1,
                }
            )
        )
        if not history_docs:
            raise HTTPException(status_code=404, detail=f"{collection_name} boş")

        history_docs.sort(key=lambda d: d["datetime"])
        first_cap_global = float(history_docs[0]["capacity_Ah"])

        history = []
        for d in history_docs:
            cap_true_i = float(d["capacity_Ah"])
            history.append({
                "cycle_id": int(d["cycle_id"]),
                "datetime": str(d["datetime"]),
                "soc_cycle_true": cap_true_i / first_cap_global
            })

        # aynı modelle her cycle için cap_pred → soc_cycle_pred
        for i, d in enumerate(history_docs):
            V_l = _as_f32(d.get("voltage_V"))
            T_l = _as_f32(d.get("temperature_C"), like_len=len(V_l))
            t_l = _as_f32(d.get("time_s"), like_len=len(V_l))
            f_l = [V_l, T_l, t_l / (t_l.max() + 1e-6)]
            if req.mode == "currcap_w_cap":
                f_l.append(np.full_like(V_l, float(d["capacity_Ah"]), dtype=np.float32))
            if req.use_load:
                Vload_l = _as_f32(d.get("voltage_load_V"), like_len=len(V_l))
                Iload_l = _as_f32(d.get("current_load_A"), like_len=len(V_l))
                f_l += [Vload_l, Iload_l]
            X_local = np.stack(f_l, axis=1).astype(np.float32)
            X_t = torch.from_numpy(X_local).unsqueeze(0).to(device)
            with torch.no_grad():
                _, cap_hat_local = model(X_t)
            cap_pred_local = float(cap_hat_local.detach().cpu().numpy().item())
            history[i]["soc_cycle_pred"] = cap_pred_local / first_cap_global

        # Seçilen cycle için zaman serisi SOC'ler
        soc_cycle, soc_whole_t_pred, soc_cyclecap_t_pred = compute_soc(
            I_pred, cap_pred, time_s, first_cap=first_cap_global
        )

        print("📌 Step 4: Model çağrılıyor:", req.mode, req.use_load)

        return {
            "dataset": req.dataset,
            "cycle_id": req.cycle_id,
            "mode": req.mode,
            "use_load": req.use_load,
            "cap_true": float(doc["capacity_Ah"]),
            "cap_pred": cap_pred,
            "soc_cycle": float(soc_cycle),
            "soc_whole_t_pred": soc_whole_t_pred.tolist(),
            "soc_cyclecap_t_pred": soc_cyclecap_t_pred.tolist(),
            "soc_whole_t_true": doc.get("soc_whole_t", []),
            "soc_cyclecap_t_true": doc.get("soc_cyclecap_t", []),
            "datetime": str(doc["datetime"]),
            "I_pred": I_pred.tolist(),
            "history": history
        }
    except HTTPException:
        raise
    except FileNotFoundError as e:
        # GridFS dosyası yoksa
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets")
def list_datasets(scope: str = Query("all"), mode: str = Query("currcap"), use_load: bool = Query(False)):
    all_ds = _list_all_datasets(db)
    if scope == "all":
        return {"datasets": all_ds}

    if scope == "test":
        fname = _model_filename(mode, use_load)
        gf = _get_gridfs_file_by_name(db, fname)
        if not gf:
            return {"datasets": all_ds, "warning": f"{fname} bulunamadı; tüm liste döndürüldü"}
        train_sets = set((gf.metadata or {}).get("train_sets", []))
        testables = [d for d in all_ds if d not in train_sets]
        return {"datasets": testables, "train_sets": sorted(list(train_sets))}
    
    return {"datasets": all_ds}


@app.get("/health")
def health():
    return {"ok": True}
