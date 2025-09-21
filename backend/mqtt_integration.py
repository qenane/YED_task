import os, json, threading
import numpy as np
import torch
from pymongo import MongoClient
import paho.mqtt.client as mqtt
from train_curr_cap import compute_soc, load_model_from_mongo

MQTT_HOST = os.getenv("MQTT_HOST", "mqtt")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
DB_NAME   = os.getenv("DB_NAME", "battery_db")

TOPIC_IN       = os.getenv("MQTT_TOPIC_IN", "yed/battery/+/telemetry")
TOPIC_OUT_FMT  = os.getenv("MQTT_TOPIC_OUT_FMT", "yed/battery/{device}/predictions")
SEQ_LEN        = int(os.getenv("SEQ_LEN", "256"))

_client = None
_model  = None
_db     = None

def _downsample(arr, L):
    arr = np.array(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(L, dtype=np.float32)
    idx = np.linspace(0, len(arr) - 1, L)
    return np.interp(idx, np.arange(len(arr)), arr).astype(np.float32)

def _ensure_db():
    global _db
    if _db is None:
        _db = MongoClient(MONGO_URI)[DB_NAME]
    return _db

def _ensure_model():
    global _model
    if _model is not None:
        return _model
    db = _ensure_db()
    # GridFS'te olabilecek isimleri sırayla dene:
    candidates = [
        "cnn_bilstm_currcap_best_w_cap_ul.pt",
        "cnn_bilstm_currcap_best_w_cap.pt",
        "cnn_bilstm_currcap_best_ul.pt",
        "cnn_bilstm_currcap_best.pt",
    ]
    last_err = None
    for name in candidates:
        try:
            _model = load_model_from_mongo(db, name)
            print(f"[MQTT] Model yüklendi: {name}")
            return _model
        except Exception as e:
            last_err = e
    raise last_err

def _infer_from_payload(payload: dict):
    """
    payload: {
      "cycle_id": 101,
      "voltage_V": [...],
      "temperature_C": [...],
      "current_A": [...],
      "time_s": [...],
      "capacity_Ah": 1.85
    }
    """
    L = SEQ_LEN
    V = _downsample(payload.get("voltage_V", []), L)
    T = _downsample(payload.get("temperature_C", []), L)
    t = _downsample(payload.get("time_s", []), L)
    # Eğitimde V,T,time kullanıyorduk:
    feats = [V, T, t / (t.max() + 1e-6)]
    X = np.stack(feats, axis=1)[None, ...]  # (1, L, C)
    X_t = torch.from_numpy(X).float()

    m = _ensure_model()
    with torch.no_grad():
        I_hat, cap_hat = m(X_t)
        I_pred   = I_hat[0].cpu().numpy()
        cap_pred = float(cap_hat[0].cpu().numpy())

    first_cap = float(payload.get("capacity_Ah") or cap_pred)
    _, soc_whole_t, soc_cyclecap = compute_soc(I_pred, cap_pred, t, first_cap)

    return {
        "cycle_id": payload.get("cycle_id"),
        "cap_pred": cap_pred,
        "soc_whole_t_pred": soc_whole_t.tolist(),
        "soc_cyclecap_t_pred": soc_cyclecap.tolist(),
    }

def _on_connect(c, userdata, flags, rc, properties=None):
    print(f"[MQTT] Connected rc={rc}")
    c.subscribe(TOPIC_IN)
    print(f"[MQTT] Subscribed: {TOPIC_IN}")

def _on_message(c, userdata, msg):
    try:
        device = msg.topic.split("/")[2] if len(msg.topic.split("/")) >= 3 else "unknown"
        payload = json.loads(msg.payload.decode("utf-8"))
        out = _infer_from_payload(payload)
        out_topic = TOPIC_OUT_FMT.format(device=device)
        c.publish(out_topic, json.dumps(out), qos=0, retain=False)
        print(f"[MQTT] Published -> {out_topic}")
    except Exception as e:
        print("[MQTT] on_message error:", repr(e))

def start_mqtt():
    global _client
    if _client:
        return
    _client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    _client.on_connect = _on_connect
    _client.on_message = _on_message
    _client.connect(MQTT_HOST, MQTT_PORT, 60)
    th = threading.Thread(target=_client.loop_forever, daemon=True)
    th.start()
    print(f"[MQTT] Loop started on {MQTT_HOST}:{MQTT_PORT}")
