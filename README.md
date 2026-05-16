# NILM Energy Monitoring Dashboard

Dashboard modern untuk sistem Non-Intrusive Load Monitoring (NILM) berbasis Deep Learning dengan sumber data dari `ESP32 + PZEM-004T` melalui `ThingsBoard`, lalu diteruskan ke service inferensi model.

## Stack

- Next.js App Router
- TypeScript
- Tailwind CSS
- Recharts
- Lucide React

## Fitur Utama

- Monitoring parameter listrik real-time
- API route backend `GET /api/blynk/latest` sebagai gateway data telemetry ke frontend
- Integrasi inferensi model NILM melalui Flask `ml_service`
- Fallback mock data untuk simulasi saat mode dummy diaktifkan
- Summary cards, power chart, device detection panel, energy cost estimation
- System status page dan settings page
- Responsive layout untuk desktop dan smartphone

## Install Dependencies

```bash
npm install
```

## Menjalankan Project

```bash
npm run dev
```

Lalu buka `http://localhost:3000`.

## Build Production

```bash
npm run build
npm run start
```

## Konfigurasi `.env.local`

Buat file `.env.local` di root project:

```env
NILM_MODEL_DIR=src/nilm_models_v9
# atau jika ingin gunakan format file alias yang kadang dipakai VS Code:
# NILM_MODEL_DIR=@file:src/nilm_models_v9
NILM_DATA_SOURCE=thingsboard
THINGSBOARD_ACCESS_TOKEN=token_device_esp32
THINGSBOARD_BASE_URL=https://your-thingsboard-host
THINGSBOARD_JWT_TOKEN=jwt_token_thingsboard
THINGSBOARD_DEVICE_ID=opsional_device_id
THINGSBOARD_USERNAME=opsional_fallback_login
THINGSBOARD_PASSWORD=opsional_fallback_login
ML_SERVICE_URL=http://127.0.0.1:5001
USE_DUMMY_BLYNK=false
NEXT_PUBLIC_REFRESH_INTERVAL=3000
```

## Menjalankan ML Service

```bash
cd ml_service
python app.py
```

Service Flask berjalan default di `http://127.0.0.1:5001`.

## Alur Live ThingsBoard

- ESP32 mengirim telemetry ke ThingsBoard memakai `THINGSBOARD_ACCESS_TOKEN`.
- Backend Next memprioritaskan `THINGSBOARD_JWT_TOKEN` untuk akses REST API ThingsBoard.
- Jika JWT tidak diisi atau expired, backend akan fallback login memakai `THINGSBOARD_USERNAME` dan `THINGSBOARD_PASSWORD`.
- Backend mengambil latest telemetry device secara live dari ThingsBoard, lalu mengirim sample ke `ml_service` untuk inferensi label.
- `THINGSBOARD_DEVICE_ID` opsional. Jika tidak diisi, backend akan mencoba mencari device berdasarkan `THINGSBOARD_ACCESS_TOKEN`.

## Key Telemetry ThingsBoard

- `tegangan -> voltage`
- `arus -> current`
- `daya -> power`
- `kwh -> energy`
- `frekuensi -> frequency`
- `power_factor -> power_factor`

Output inferensi yang dikembalikan backend:

- `device_detected`
- `confidence`
- `model_version`
- `timestamp`

## Cara Ganti Tarif Listrik

- Buka menu `Settings`
- Ubah field `Tarif listrik per kWh`
- Nilai otomatis disimpan di browser melalui `localStorage`

## Catatan API

- Endpoint backend: `GET /api/blynk/latest`
- Jika `NILM_DATA_SOURCE=thingsboard`, route akan login ke ThingsBoard dan membaca latest telemetry live dari device.
- `THINGSBOARD_ACCESS_TOKEN` dipakai ESP32 untuk publish telemetry ke ThingsBoard.
- Backend bisa langsung memakai `THINGSBOARD_JWT_TOKEN` manual jika sudah tersedia.
