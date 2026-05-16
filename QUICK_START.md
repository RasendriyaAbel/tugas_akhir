# 🚀 NILM v9 Model - Deploy & Run

Model v9 multi-label sudah siap. Berikut cara menjalankan:

## ✅ Verifikasi Files

Model directory: `src/nilm_models_v9/`

| File | Status | Size |
|------|--------|------|
| `best_nilm_model.keras` | ✓ | 2.58 MB |
| `meta_nilm.json` | ✓ | Metadata lengkap + scaler stats |
| `scaler_nilm.pkl` | ✓ | StandardScaler |
| `nilm_inference.py` | ✓ | Inference utilities |
| Training logs & EDA charts | ✓ | Dokumentasi |

## 🎯 Quick Start (3 Steps)

### Step 1️⃣ - ML Service (Backend Flask)

**Terminal 1:**
```bash
cd ml_service
.\.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac

python app.py
```

Output:
```
 * Running on http://127.0.0.1:5001
```

### Step 2️⃣ - Frontend (Next.js)

**Terminal 2:**
```bash
npm install    # hanya first time
npm run dev
```

Output:
```
◇ Ready in 1.2s
○ http://localhost:3000
```

### Step 3️⃣ - Open Dashboard

Buka browser: **http://localhost:3000**

## 📊 Apa yang Akan Terjadi

1. **Dashboard Load** ✓
   - Model version: `v9_multilabel`
   - Display 4 devices: charger_hp, hair_dryer, kipas, laptop

2. **Sensor Data** ✓
   - ThingsBoard → Backend → ML Service
   - Real-time voltage, current, power, frequency
   - Power Factor untuk load signature

3. **Model Inference** ✓
   - 30 timestep rolling window
   - Multi-label detection (1-4 devices dapat aktif sekaligus)
   - Threshold 0.5 per device
   - Majority voting smoothing

4. **Results Display** ✓
   - Detected devices: `hair_dryer + kipas + laptop`
   - Confidence: 97.5%
   - Model version: v9_multilabel

## 🔧 Environment Variables

File `.env` sudah benar:

```env
NILM_MODEL_DIR=src/nilm_models_v9
NILM_DATA_SOURCE=thingsboard
ML_SERVICE_URL=http://127.0.0.1:5001
THINGSBOARD_BASE_URL=https://eu.thingsboard.cloud/
THINGSBOARD_JWT_TOKEN=eyJhbGci...
```

Alternative path format (keduanya valid):
```env
NILM_MODEL_DIR=@file:src/nilm_models_v9
NILM_MODEL_DIR=file://src/nilm_models_v9
```

## 📝 Model Specification

| Aspek | Detail |
|-------|--------|
| **Versi** | v9_multilabel |
| **Type** | Multi-label classification |
| **Devices** | 4 (charger_hp, hair_dryer, kipas, laptop) |
| **Input** | 30 timestep × 8 fitur |
| **Output** | 4 sigmoid neurons |
| **Val Accuracy** | 94.49% exact match |
| **Val AUC** | 0.9997 |
| **Features** | voltage, current, power, power_factor, frequency, apparent_power, reactive_power, power_ratio |

## 🐛 Troubleshooting

### ❌ ML Service tidak start

```bash
# Cek model files
python -c "from pathlib import Path; print(list(Path('src/nilm_models_v9').glob('*')))"

# Cek Python packages
pip list | grep tensorflow keras numpy
```

### ❌ Frontend tidak connect ke ML Service

```bash
# Verify ML Service running
curl http://127.0.0.1:5001/health

# Should return: {"success": true, "model_dir": "..."}
```

### ❌ Model loading error

```bash
# Run test
python test_model_integration.py

# Should pass: 4/4 tests
```

## 📚 Files Reference

| File | Purpose |
|------|---------|
| `quick_start.ps1` | Windows start script |
| `quick_start.sh` | Linux/Mac start script |
| `test_model_integration.py` | Verify model integration |
| `MODEL_INTEGRATION_GUIDE.md` | Detailed documentation |
| `MODEL_V9_CHECKLIST.md` | Technical checklist |

## ✨ Features

✓ Multi-device detection  
✓ Real-time streaming  
✓ 30-sample rolling window  
✓ Confidence scoring  
✓ Device combination support  
✓ Power-based validation  
✓ Majority voting smoothing  

## 🎉 Ready!

Model v9 integration complete. Siap untuk production!

**Next**: Jalankan Terminal 1 & 2 sesuai Quick Start 3 steps di atas.
