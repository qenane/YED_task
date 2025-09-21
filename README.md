YED Battery SOC — End-to-End Demo

Bu repo; FastAPI backend + MQTT entegrasyonu + React/NGINX frontend + MongoDB + Mosquitto + Ngrok ile, batarya deşarj verilerinden akım dalgası (I[t]) ve cycle kapasitesi (Ah) tahmini yapan ve SOC (state of charge) eğrilerini hesaplayıp görselleştiren uçtan uca bir demo içerir.
Amaç: Bir batarya setiyle eğitilen modelin başka bataryaların MQTT üzerinden yayınlanan verileriyle test edilebildiğini göstermek (case gereksinimi).

İçindekiler

Mimari

Özellikler

Veri Kümeleri

Model ve Eğitim

Docker ile Çalıştırma

MQTT Test Senaryosu

REST API

Frontend

Ngrok (Opsiyonel)

Metrikler ve Raporlama

Sınırlamalar & Notlar


#Mimari
[React+NGINX]  <--->  [FastAPI Backend]  <--->  [MongoDB(GridFS)]
       ^                        ^
       |                        |-- subscribes: yed/battery/+/telemetry (MQTT)
       |                        |-- publishes: yed/battery/+/predictions (MQTT)
       |                        |
     Browser               Mosquitto Broker
                              (TCP 1883)


MongoDB: Zaman serisi/cycle dokümanları ve model ağırlıkları (GridFS)

FastAPI: Tahmin endpoint’i, dataset listesi, MQTT işleyicileri

MQTT: Telemetry alımı (ham ölçümler) ve prediction yayını

Frontend (React/NGINX): Form + grafikler (tarih bazlı soc_cycle_true & soc_cycle_pred çizgileri, seçili cycle için SOC panelleri)

#Özellikler

Çoklu girdi: voltage_V, temperature_C, time_s (+ opsiyonel voltage_load_V, current_load_A ve/veya capacity_Ah)

Model: CNN + BiLSTM, iki “head”:

Zaman adımı bazında I[t]

Cycle başına kapasite (Ah)

SOC Hesapları:

soc_cycle: cap_pred / first_cap_global

soc_whole_t_pred: cap_pred referanslı integral SOC

soc_cyclecap_t_pred: cap_pred normalize integral SOC

Tarih çizgisi: Tüm cycle’lar için true & predicted SOC_cycle zaman çizgisi (frontend grafiğinde)

Eğitim/Test Ayrımı Koruması: Model GridFS meta verisiyle hangi dataset’lerle eğitildiği ve hangi dataset’in test olduğu kayıtlıdır; backend UI’da test set seçilemez.

#Veri Kümeleri

Case’de geçen 4 dataset: B0005, B0006, B0018, B0007

Örnek iş akışı:

Eğitim: B0005, B0006, B0018

Test: B0007

Mongo koleksiyon adlandırma:

Ham zaman serisi: {DATASET}_timeseries (ör. B0005_timeseries)

(Varsa) ara işlemler: {DATASET}_processed

Not: Sistem, eğitim/test split bilgisini model meta verisine yazar.

#Model ve Eğitim
Eğitim script’i

train_curr_cap.py (güncellenmiş sürüm) – başlıklar:

CurrCapDataset: downsample + feature birleştirme

CNN_BiLSTM_CurrCap: 1D CNN (2 katman) + BiLSTM + 2 head

compute_soc: entegral SOC hesapları

GridFS kayıt: Ağırlık + metadata (train_sets, test_set, metrics, use_load, with_cap vs.)

Önemli Argümanlar
python train_curr_cap.py ^
  --mongo_uri "mongodb://localhost:27017" ^
  --db_name battery_db ^
  --train_colls B0005_timeseries B0006_timeseries B0018_timeseries ^
  --test_coll B0007_timeseries ^
  --seq_len 256 ^
  --use_load            # load ölçüleri varsa ekler
  --load_cap            # cap'i feature olarak ekler
  --hidden 128 --layers 2 --dropout 0.3 ^
  --batch_size 16 --epochs 50 --lr 1e-3 --alpha 10.0


--alpha: kapasite kaybının ağırlığı (loss = MSE(I) + α·MSE(cap))

#Eğitim/Değerlendirme Çıktıları

artifacts_seqsoc/test_results.json & test_results.csv

Model GridFS metadata:

{
  "model": "cnn_bilstm_currcap_best",
  "with_cap": true,
  "use_load": true,
  "in_dim": 5,
  "hidden": 128,
  "layers": 2,
  "dropout": 0.3,
  "train_sets": ["B0005","B0006","B0018"],
  "test_set": "B0007",
  "metrics": {"rmse_I": ..., "rmse_cap": ..., "best_score": ...}
}

#Docker ile Çalıştırma
#1) Ortam Değişkenleri

Backend CORS için frontend URL’si: http://localhost:5173

Compose içinde:

MONGO_URI=mongodb://mongodb:27017

MQTT_HOST=mqtt

MQTT_PORT=1883

(Opsiyonel) NGROK_AUTHTOKEN=...

#2) Compose
docker compose up -d --build
 veya ilk kurulum sonrası:
docker compose up -d


Servisler:

mongodb → 27017

mqtt (Mosquitto) → 1883

backend (FastAPI) → 8000

frontend (NGINX) → 5173

ngrok (opsiyonel UI: 4040)

3) Sağlık Kontrolü
# Backend dataset listesi
curl http://localhost:8000/datasets
# Frontend
http://localhost:5173

#MQTT Test Senaryosu

Amaç: Eğitim B0005/6/18 ile yapılmış modeli, B0007 telemetri yayını gelince test etmek.

Publish (örnek)

Aşağıdaki JSON’u tek satır olacak şekilde gönderin (kaçış karakterlerine dikkat):

docker exec -it mqtt sh -lc 'apk add --no-cache mosquitto-clients >/dev/null && \
mosquitto_pub -h localhost -p 1883 -t "yed/battery/B0007/telemetry" -m "{\"dataset\":\"B0007\",\"cycle_id\":42,\"voltage_V\":[3.9,3.85,3.8],\"temperature_C\":[24.0,24.1,24.2],\"time_s\":[0,5,10]}"'


Backend, mesajı alır; modele sokar; sonuçları yed/battery/B0007/predictions konusuna yayınlar.

Subscribe (gözlem)
docker exec -it mqtt sh -lc 'apk add --no-cache mosquitto-clients >/dev/null && \
mosquitto_sub -h localhost -p 1883 -t "yed/battery/B0007/predictions" -v'

#REST API
GET /datasets

Mevcut dataset isimlerini döner (backend, model meta verisini kontrol ederek test set’in eğitime seçilmesini engeller).

Örnek:

{"datasets": ["B0005","B0006","B0007","B0018"]}

POST /predict

Seçili dataset & cycle için tahmin ve SOC serileri döner.

Body:

{
  "dataset": "B0007",
  "cycle_id": 42,
  "mode": "currcap_w_cap",
  "use_load": true
}


Yanıt (kısaltılmış):

{
  "dataset": "B0007",
  "cycle_id": 42,
  "cap_true": 1.92,
  "cap_pred": 1.88,
  "soc_cycle": 0.96,
  "soc_whole_t_pred": [...],
  "soc_cyclecap_t_pred": [...],
  "datetime": "2020-01-10T02:34:00",
  "I_pred": [...],
  "history": [
    {"cycle_id":0,"datetime":"...","soc_cycle_true":0.99,"soc_cycle_pred":0.98},
    ...
  ]
}

#Frontend

PredictForm.jsx: VITE_API_BASE üzerinden POST /predict çağırır.
NGINX imajında build çıktısı otomatik kopyalanır.

Grafikler:

Üstte tarih bazlı soc_cycle_true & soc_cycle_pred

Altta seçili cycle için SOC(t) panelleri

CORS: Backend CORSMiddleware ile http://localhost:5173 izinlidir.

#Ngrok (Opsiyonel)

ngrok.yml:

version: "2"
authtoken: ${NGROK_AUTHTOKEN}
web_addr: 0.0.0.0:4040
tunnels:
  api-http:
    proto: http
    addr: backend:8000
  mqtt-tcp:
    proto: tcp
    addr: mqtt:1883


Çalışan tünelleri görmek için: http://localhost:4040

#Metrikler ve Raporlama

Eğitim sonunda metrics.json ve GridFS metadata doldurulur:

rmse_I, rmse_cap, best_score

Backend, /predict sonuçlarıyla birlikte history alanında tüm cycle’ların predicted vs. true SOC_cycle verisini frontend’e gönderir.

İsteğe bağlı CSV/JSON çıktı:

artifacts_seqsoc/test_results.(csv|json)

#Sınırlamalar & Notlar

MQTT payload formatı JSON ve tek satır olmalı. \ kaçışlarına dikkat.

use_load=True iken load ölçüleri yoksa, dataset tarafında sıfırlarla doldurma yapılır (uyarı basılır).

Model sequence length (L=256) için downsample uygular; farklı örnekleme hızlarında da çalışır.

Test set (ör. B0007) eğitim listesine düşmemeli; backend ve metadata kontrolü bu durumu önler.
