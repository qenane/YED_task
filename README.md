Proje bileşenleri

backend (FastAPI + PyTorch + Paho MQTT + PyMongo/GridFS)

frontend (React, Nginx ile servis)

mongodb (veri ve model ağırlıkları GridFS)

mqtt (Eclipse Mosquitto broker)

ngrok (opsiyonel, dış dünyaya açmak için)

Servis Portları

Backend: 8000

Frontend (Nginx): 5173

MongoDB: 27017

MQTT: 1883

ngrok UI: 4040

Hızlı Başlangıç
# kök dizinde
docker compose up -d --build


Kontrol:

# backend
curl http://localhost:8000/datasets
# frontend tarayıcıda
http://localhost:5173


Windows PowerShell’de curl yerine Invoke-WebRequest kullanabilirsiniz:
iwr http://localhost:8000/datasets

Çevresel Değişkenler

Backend:

MONGO_URI=mongodb://mongodb:27017

MQTT_HOST=mqtt

MQTT_PORT=1883

Frontend:

VITE_API_BASE=http://localhost:8000
Değiştirdiyseniz frontend imajını yeniden build edin.

API’ler
1) Dataset listesi
GET /datasets


Yanıt:

{"datasets":["B0005","B0006","B0007","B0018"]}

2) Tahmin
POST /predict
Content-Type: application/json
{
  "dataset": "B0007",
  "cycle_id": 42,
  "mode": "currcap_w_cap",   // veya "currcap"
  "use_load": false          // true ise voltage_load_V/current_load_A beklenir
}


Yanıt (özet):

{
  "dataset":"B0007",
  "cycle_id":42,
  "cap_true": 1.84,
  "cap_pred": 1.80,
  "soc_cycle": 0.97,
  "soc_whole_t_pred": [...],
  "soc_cyclecap_t_pred": [...],
  "soc_whole_t_true": [...],
  "soc_cyclecap_t_true": [...],
  "datetime": "2020-01-02T03:04:05Z",
  "I_pred": [...],
  "history": [
    {"cycle_id":1,"datetime":"...","soc_cycle_true":0.99,"soc_cycle_pred":0.98},
    ...
  ]
}

3) Model bilgisi / metrikler (YENİ)

Aşağıdaki kod ile ekleyeceksin (backend kodu bölümüne bak).

GET /model-info


Yanıt (örnek):

{
  "filename":"cnn_bilstm_currcap_best_w_cap.pt",
  "train_sets":["B0005","B0006","B0018"],
  "test_set":"B0007",
  "metrics":{"rmse_I":0.032,"rmse_cap":0.045,"best_score":0.47,"timestamp":"..."}
}
