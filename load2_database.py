import scipy.io
import numpy as np
import pandas as pd
import datetime
import os
import glob
from pymongo import MongoClient

def load_discharge_timeseries(mat_files):
    if isinstance(mat_files, str):
        mat_files = [mat_files]

    results = {}

    for file in mat_files:
        data = scipy.io.loadmat(file)
        root_key = [k for k in data.keys() if not k.startswith("__")][0]
        cycles = data[root_key]['cycle'][0,0][0]

        discharge_cycles = []
        for idx, c in enumerate(cycles, start=1): 
            ctype = c['type'][0]
            if ctype == 'discharge':
                discharge_cycles.append((idx, c))

        if len(discharge_cycles) == 0:
            print(f"{file} içinde discharge cycle bulunamadı!")
            continue

        first_cap = discharge_cycles[0][1]['data'][0,0]['Capacity'][0,0]

        ts_rows = []
        for idx, c in discharge_cycles:
            t = c['time'][0]
            start_dt = datetime.datetime(int(t[0]), int(t[1]), int(t[2]),
                                        int(t[3]), int(t[4]), int(t[5]))
            d = c['data'][0,0]

            cap = float(d['Capacity'][0,0])
            V   = d['Voltage_measured'][0].astype(float)
            I   = d['Current_measured'][0].astype(float)
            T   = d['Temperature_measured'][0].astype(float)
            tm  = d['Time'][0].astype(float)

            I_load = d['Current_load'][0].astype(float) if 'Current_load' in d.dtype.names else np.full_like(V, np.nan)
            V_load = d['Voltage_load'][0].astype(float) if 'Voltage_load' in d.dtype.names else np.full_like(V, np.nan)

            soc_cycle = cap / first_cap
            dt_arr = np.diff(tm, prepend=tm[0])
            sign = -1.0 if np.nanmedian(I) < 0 else 1.0
            q_drawn_ah = np.cumsum(sign * I * dt_arr) / 3600.0

            soc_whole_t   = np.clip(soc_cycle - q_drawn_ah / first_cap, 0, 1.05)
            soc_cyclecap_t  = np.clip(1 - q_drawn_ah / cap, 0, 1.05)

            ts_rows.append({
                "cycle_id": idx,
                "datetime": start_dt,
                "time_s": tm.tolist(),
                "voltage_V": V.tolist(),
                "current_A": I.tolist(),
                "temperature_C": T.tolist(),
                "current_load_A": I_load.tolist(),
                "voltage_load_V": V_load.tolist(),
                "soc_whole_t": soc_whole_t.tolist(),
                "soc_cyclecap_t": soc_cyclecap_t.tolist(),
                "capacity_Ah": cap,
                "soc_cycle": soc_cycle
            })

        df_timeseries = pd.DataFrame(ts_rows).sort_values("datetime").reset_index(drop=True)
        results[os.path.basename(file).replace(".mat", "")] = df_timeseries

    return results


# --- MongoDB'ye yazma kısmı ---
client = MongoClient("mongodb://localhost:27017/")
db = client["battery_db"]

dfs = load_discharge_timeseries(["./dataset/B0018.mat", "./dataset/B0006.mat", "./dataset/B0007.mat", "./dataset/B0005.mat"])

for name, df in dfs.items():
    collection = db[f"{name}_timeseries"]
    # ✅ önce veri var mı kontrol et
    if collection.estimated_document_count() > 0:
        print(f"⏩ {name}_timeseries koleksiyonu zaten dolu, atlanıyor.")
        continue
    collection.insert_many(df.to_dict(orient="records"))
    print(f"✅ {name}_timeseries koleksiyonuna {len(df)} cycle eklendi.")


def insert_pkl_to_mongo(pkl_path, mongo_uri="mongodb://localhost:27017/",
                        db_name="battery_db", collection_name="processed_data"):
    """
    Bir .pkl dosyasındaki DataFrame'i okuyup MongoDB'ye kaydeder (eğer boşsa).
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # ✅ veri var mı kontrol
    if collection.estimated_document_count() > 0:
        print(f"⏩ {collection_name} koleksiyonu zaten dolu, atlanıyor.")
        return 0

    df = pd.read_pickle(pkl_path)

    records = []
    for _, row in df.iterrows():
        doc = {}
        for col, val in row.items():
            if isinstance(val, np.ndarray):
                doc[col] = val.tolist()
            elif isinstance(val, (np.generic,)):
                doc[col] = val.item()
            elif hasattr(val, "to_pydatetime"):
                doc[col] = val.to_pydatetime()
            else:
                doc[col] = val
        records.append(doc)

    if records:
        collection.insert_many(records)
    print(f"✅ {len(records)} kayıt eklendi -> {db_name}.{collection_name}")
    return len(records)


def insert_all_processed_pkls_to_mongo(directory="./", mongo_uri="mongodb://localhost:27017/",
                                      db_name="battery_db"):
    pkl_files = glob.glob(os.path.join(directory, "*_processed.pkl"))
    if not pkl_files:
        print("⚠️ Dizinde *_processed.pkl dosyası bulunamadı!")
        return

    for pkl_path in pkl_files:
        file_name = os.path.basename(pkl_path).replace(".pkl", "")
        collection_name = file_name
        insert_pkl_to_mongo(pkl_path, mongo_uri, db_name, collection_name)


# ---- Çalıştır ----
# insert_all_processed_pkls_to_mongo(directory="./")
