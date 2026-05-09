import json
import os
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
import numpy as np
import requests
from flask import Flask, jsonify, request

# Load environment variables from the repository root .env file if available.
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

_model_dir = Path(os.environ.get("NILM_MODEL_DIR", "src/nilm_models_v9"))
MODEL_DIR = _model_dir if _model_dir.is_absolute() else (ROOT_DIR / _model_dir)
MODEL_DIR = MODEL_DIR.resolve()
_DUMMY_FILE = Path(__file__).resolve().parent / "dummy_blynk_samples.json"
_MODEL_TEXT_FILES = ("config.json", "metadata.json", "labels.json", "meta_nilm.json")
_MODEL_BINARY_FILES = ("model.weights.h5",)
_NOTEBOOK_GLOB = "*.ipynb"
_MODEL_ARCHIVE_GLOB = "*.keras"

app = Flask(__name__)

_MODEL = None
_LABELS_CACHE = None
_LABEL_SOURCE_CACHE = None
_LABELS_CACHE_KEY = None
_MODEL_META_CACHE = None
_EMA_PROBS = None
_PREV_POWER = None
_LATEST_RESULT = None
_REQUEST_COUNT = 0
_SEQ_BUFFER = deque(maxlen=99)
_LAST_RAW_SAMPLE = None
_LOCK = Lock()


def _read_json(path: Path):
  return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path):
  return path.read_text(encoding="utf-8")


def _sanitize_keras_config(value):
  if isinstance(value, dict):
    return {
      key: _sanitize_keras_config(item)
      for key, item in value.items()
      if key not in {"quantization_config"}
    }
  if isinstance(value, list):
    return [_sanitize_keras_config(item) for item in value]
  return value


def _model_root() -> Path:
  return MODEL_DIR if MODEL_DIR.is_dir() else MODEL_DIR.parent


def _find_keras_file() -> Path | None:
  if MODEL_DIR.is_file() and MODEL_DIR.suffix == ".keras":
    return MODEL_DIR

  if MODEL_DIR.is_dir():
    candidates = sorted(MODEL_DIR.glob(_MODEL_ARCHIVE_GLOB))
    if candidates:
      return candidates[0]

  return None


def _get_model_files():
  root = _model_root()
  files = []
  for name in (*_MODEL_TEXT_FILES, *_MODEL_BINARY_FILES):
    path = root / name
    files.append(
      {
        "name": name,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "type": "text" if name in _MODEL_TEXT_FILES else "binary",
      }
    )

  keras_file = _find_keras_file()
  if keras_file is not None:
    files.append(
      {
        "name": keras_file.name,
        "exists": True,
        "size_bytes": keras_file.stat().st_size,
        "type": "binary",
      }
    )
  return files


def _resolve_model_file(name: str):
  normalized = Path(name).name
  allowed_files = set(_MODEL_TEXT_FILES) | set(_MODEL_BINARY_FILES)
  keras_file = _find_keras_file()
  if keras_file is not None:
    allowed_files.add(keras_file.name)

  if normalized not in allowed_files:
    raise ValueError("File tidak diizinkan. Gunakan config.json, metadata.json, labels.json, meta_nilm.json, model.weights.h5, atau file .keras model")

  root = _model_root()
  if normalized == keras_file.name if keras_file is not None else False:
    return keras_file

  return root / normalized


def _extract_notebook_classes():
  candidates = sorted(MODEL_DIR.glob(_NOTEBOOK_GLOB))
  for notebook_path in candidates:
    try:
      notebook = _read_json(notebook_path)
    except Exception:
      continue

    for cell in notebook.get("cells", []):
      for line in cell.get("source", []):
        if "\"classes\":" in line.lower():
          try:
            snippet = line[line.index("{"):]
            payload = json.loads(snippet)
            classes = payload.get("classes")
            if isinstance(classes, list) and all(isinstance(item, str) for item in classes):
              return [item.strip() for item in classes if item.strip()], notebook_path.name
          except Exception:
            continue
    for cell in notebook.get("cells", []):
      source = "".join(cell.get("source", []))
      marker = "CLASSES = ["
      if marker not in source:
        continue
      try:
        start = source.index(marker) + len(marker)
        end = source.index("]", start)
        raw_items = source[start:end].splitlines()
        classes = [item.strip().strip(",").strip("'\"") for item in raw_items]
        classes = [item for item in classes if item]
        if classes:
          return classes, notebook_path.name
      except Exception:
        continue
  return None, None


def _read_model_meta():
  global _MODEL_META_CACHE
  if _MODEL_META_CACHE is not None:
    return _MODEL_META_CACHE

  root = _model_root()
  meta_path = root / "meta_nilm.json"
  if not meta_path.exists():
    meta_path = root / "metadata.json"

  if meta_path.exists():
    meta = _read_json(meta_path)
    model_name = meta.get("model_version") or "unknown_model"
    input_shape = []
    if isinstance(meta.get("window_size"), int) and isinstance(meta.get("n_features"), int):
      input_shape = [meta["window_size"], meta["n_features"]]

    output_units = meta.get("n_classes")
    if not isinstance(output_units, int) or output_units <= 0:
      session_to_label = meta.get("session_to_label")
      if isinstance(session_to_label, dict):
        output_units = len(session_to_label)

    _MODEL_META_CACHE = {
      "model_name": model_name,
      "input_shape": input_shape,
      "output_units": output_units,
      "scaler_mean": meta.get("scaler_mean"),
      "scaler_scale": meta.get("scaler_scale"),
      "noise_floor_w": meta.get("noise_floor_w"),
      "transition_delta": meta.get("transition_delta"),
      "conf_thresh": meta.get("conf_thresh"),
      "power_range": meta.get("power_range"),
    }
    return _MODEL_META_CACHE

  config = _read_json(root / "config.json")
  layers = config.get("config", {}).get("layers", [])
  input_layer = next((layer for layer in layers if layer.get("class_name") == "InputLayer"), None)
  output_layer = next(
    (
      layer
      for layer in reversed(layers)
      if layer.get("class_name") == "Dense" and layer.get("config", {}).get("activation") == "softmax"
    ),
    None,
  )

  input_shape = (input_layer or {}).get("config", {}).get("batch_shape") or []
  input_shape = [value for value in input_shape if isinstance(value, int)]
  output_units = (output_layer or {}).get("config", {}).get("units")
  model_name = config.get("config", {}).get("name") or "unknown_model"

  _MODEL_META_CACHE = {
    "model_name": model_name,
    "input_shape": input_shape,
    "output_units": output_units,
  }
  return _MODEL_META_CACHE


def _load_labels():
  global _LABELS_CACHE, _LABEL_SOURCE_CACHE, _LABELS_CACHE_KEY
  root = _model_root()
  labels_path = root / "labels.json"
  meta_path = root / "meta_nilm.json"
  cache_key = None

  if meta_path.exists():
    stat = meta_path.stat()
    cache_key = (str(meta_path), stat.st_mtime_ns, stat.st_size)
  elif labels_path.exists():
    stat = labels_path.stat()
    cache_key = (str(labels_path), stat.st_mtime_ns, stat.st_size)

  if _LABELS_CACHE is not None and _LABELS_CACHE_KEY == cache_key:
    return _LABELS_CACHE

  _LABELS_CACHE = None
  _LABEL_SOURCE_CACHE = None
  _LABELS_CACHE_KEY = cache_key
  meta = _read_model_meta()
  output_units = meta.get("output_units")

  if meta_path.exists():
    meta = _read_json(meta_path)
    classes = meta.get("classes")
    labels = []

    if classes is None:
      session_to_label = meta.get("session_to_label")
      if isinstance(session_to_label, dict):
        seen = set()
        for label in session_to_label.values():
          if isinstance(label, str):
            label = label.strip()
            if label and label not in seen:
              seen.add(label)
              labels.append(label)
      else:
        raise ValueError("meta_nilm.json invalid: field 'classes' harus array string atau field 'session_to_label' harus object string")
    else:
      if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
        raise ValueError("meta_nilm.json invalid: field 'classes' harus array string")
      labels = [item.strip() for item in classes if item.strip()]

    _LABEL_SOURCE_CACHE = "meta_nilm.json"
  elif labels_path.exists():
    configured_labels = _read_json(labels_path).get("labels", [])
    if not isinstance(configured_labels, list) or not all(isinstance(item, str) for item in configured_labels):
      raise ValueError("labels.json invalid: field 'labels' harus array string")

    labels = [item.strip() for item in configured_labels if isinstance(item, str) and item.strip()]

    if isinstance(output_units, int) and output_units > 0:
      if len(labels) > output_units:
        raise ValueError(f"labels.json tidak boleh melebihi {output_units} label, sekarang {len(labels)}")
      if len(labels) < output_units:
        labels.extend(f"unknown_{index}" for index in range(len(labels), output_units))
    _LABEL_SOURCE_CACHE = "labels.json"
  else:
    notebook_labels, notebook_name = _extract_notebook_classes()
    if (
      isinstance(output_units, int)
      and output_units > 0
      and isinstance(notebook_labels, list)
      and len(notebook_labels) == output_units
    ):
      labels = notebook_labels
      _LABEL_SOURCE_CACHE = f"notebook:{notebook_name}"
    elif isinstance(output_units, int) and output_units > 0:
      labels = [f"unknown_{index}" for index in range(output_units)]
      _LABEL_SOURCE_CACHE = "generated"
    else:
      raise ValueError("labels.json tidak ditemukan dan output_units model tidak dapat dibaca")

  labels = [item.strip() for item in labels if isinstance(item, str) and item.strip()]
  _LABELS_CACHE = labels
  return _LABELS_CACHE



def _get_label_source():
  if _LABEL_SOURCE_CACHE is None:
    _load_labels()
  return _LABEL_SOURCE_CACHE


def _get_model():
  global _MODEL
  if _MODEL is not None:
    return _MODEL

  try:
    from keras.models import model_from_json
    from keras.saving import load_model
  except Exception as exc:
    raise RuntimeError(f"Keras tidak tersedia: {exc}") from exc

  model_source = MODEL_DIR
  keras_file = _find_keras_file()
  if keras_file is not None:
    model_source = keras_file

  try:
    _MODEL = load_model(str(model_source), compile=False)
  except Exception as exc:
    root = _model_root()
    config_path = root / "config.json"
    weights_path = root / "model.weights.h5"

    if not config_path.exists() or not weights_path.exists():
      raise RuntimeError(f"Gagal load model dari {model_source}: {exc}") from exc

    try:
      sanitized_config = _sanitize_keras_config(_read_json(config_path))
      _MODEL = model_from_json(json.dumps(sanitized_config))
      _MODEL.load_weights(str(weights_path))
    except Exception as rebuild_exc:
      raise RuntimeError(f"Gagal load model dari {model_source}: {rebuild_exc}") from rebuild_exc

  return _MODEL


def _now_iso():
  return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _blynk_update(pin: str, value: str | float | int):
  token = (os.environ.get("BLYNK_AUTH_TOKEN") or "").strip()
  if not token:
    raise RuntimeError("BLYNK_AUTH_TOKEN belum di-set di environment")

  base = (os.environ.get("BLYNK_BASE_URL") or "https://blynk.cloud/external/api").rstrip("/")
  url = f"{base}/update"
  response = requests.get(url, params={"token": token, pin: value}, timeout=10)
  if response.status_code != 200:
    raise RuntimeError(f"Blynk update {pin} gagal ({response.status_code}): {response.text}")


def _parse_sequence(payload):
  if not isinstance(payload, dict):
    raise ValueError("Body JSON harus object")

  sequence = payload.get("sequence")
  if sequence is None:
    raise ValueError("Field 'sequence' wajib ada")

  arr = np.array(sequence, dtype=np.float32)
  received_len = int(arr.shape[0]) if arr.ndim >= 2 else 1
  meta = _read_model_meta()
  input_shape = meta.get("input_shape") or []
  seq_len = int(input_shape[0]) if len(input_shape) >= 2 and isinstance(input_shape[0], int) else 99

  if arr.ndim == 1:
    if arr.size == 8:
      arr = arr.reshape((1, 8))
    elif arr.size == seq_len * 8:
      arr = arr.reshape((seq_len, 8))
    else:
      raise ValueError(f"Shape sequence tidak valid: {arr.shape}")

  if arr.ndim != 2 or arr.shape[1] != 8:
    raise ValueError(f"Shape sequence harus (*, 8), dapat {arr.shape}")

  if arr.shape[0] > seq_len:
    arr = arr[-seq_len:, :]

  if arr.shape[0] < seq_len:
    last_row = arr[-1:, :]
    repeat = seq_len - arr.shape[0]
    pad = np.repeat(last_row, repeat, axis=0)
    arr = np.concatenate([arr, pad], axis=0)

  return arr, received_len


def _apply_smoothing(probs: np.ndarray, alpha: float):
  global _EMA_PROBS
  if _EMA_PROBS is None:
    _EMA_PROBS = probs.astype(np.float32)
    return probs

  alpha = float(alpha)
  alpha = 0.0 if alpha < 0 else 1.0 if alpha > 1 else alpha
  _EMA_PROBS = (alpha * probs) + ((1.0 - alpha) * _EMA_PROBS)
  return _EMA_PROBS


def _format_buffer_bar(current: int, total: int, width: int = 20):
  total = max(1, int(total))
  current = max(0, min(int(current), total))
  width = max(5, int(width))
  filled = int(round((current / total) * width))
  return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _as_float(value, default=0.0):
  try:
    return float(value)
  except Exception:
    return float(default)


def _make_json_safe(value):
  if isinstance(value, dict):
    return {str(k): _make_json_safe(v) for k, v in value.items()}
  if isinstance(value, list):
    return [_make_json_safe(v) for v in value]
  if isinstance(value, tuple):
    return tuple(_make_json_safe(v) for v in value)
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, (np.ndarray,)):
    return _make_json_safe(value.tolist())
  return value


def _power_from_sample(sample):
  if not isinstance(sample, dict):
    return 0.0
  return _as_float(sample.get("power", sample.get("P")), 0.0)


def _build_feature_vector(sample: dict):
  v = _as_float(sample.get("voltage", sample.get("V")))
  i = _as_float(sample.get("current", sample.get("I")))
  p = _as_float(sample.get("power", sample.get("P")))
  pf = _as_float(sample.get("power_factor", sample.get("PF")))
  hz = _as_float(sample.get("frequency", sample.get("Hz")))

  pf = max(0.0, min(pf, 1.0))
  apparent_power = v * i
  reactive_power = apparent_power * np.sqrt(max(0.0, 1.0 - (pf ** 2)))
  power_ratio = p / (apparent_power + 1e-6)

  return np.array(
    [v, i, p, pf, hz, apparent_power, reactive_power, power_ratio],
    dtype=np.float32,
  )


def _scale_sequence(sequence: np.ndarray, meta: dict):
  scaler_mean = meta.get("scaler_mean")
  scaler_scale = meta.get("scaler_scale")
  if (
    isinstance(scaler_mean, list)
    and isinstance(scaler_scale, list)
    and len(scaler_mean) == sequence.shape[1]
    and len(scaler_scale) == sequence.shape[1]
  ):
    mean_arr = np.array(scaler_mean, dtype=np.float32)
    scale_arr = np.array([float(s) if float(s) != 0.0 else 1.0 for s in scaler_scale], dtype=np.float32)
    return (sequence - mean_arr) / scale_arr

  return sequence


def _should_reset_buffer_for_device_change(sample):
  global _PREV_POWER
  if not isinstance(sample, dict):
    return False

  current_power = _power_from_sample(sample)
  if _PREV_POWER is None:
    _PREV_POWER = current_power
    return False

  meta = _read_model_meta()
  transition_delta = float(meta.get("transition_delta") or 30.0)
  noise_floor_w = float(meta.get("noise_floor_w") or 3.0)

  should_reset = (
    abs(current_power - _PREV_POWER) > transition_delta
    or (_PREV_POWER <= noise_floor_w and current_power > noise_floor_w)
    or (_PREV_POWER > noise_floor_w and current_power <= noise_floor_w)
  )
  _PREV_POWER = current_power
  return should_reset


def _normalize_sample(payload):
  if not isinstance(payload, dict):
    raise ValueError("Body JSON harus object")

  for key in ("sample", "telemetry", "data"):
    candidate = payload.get(key)
    if isinstance(candidate, dict):
      return candidate

  if all(key in payload for key in ("voltage", "current", "power")):
    return payload

  return payload


def _sample_to_dict(sample):
  return {
    "voltage": _as_float(sample.get("voltage", sample.get("V"))),
    "current": _as_float(sample.get("current", sample.get("I"))),
    "power": _as_float(sample.get("power", sample.get("P"))),
    "energy": _as_float(sample.get("energy", sample.get("E"))),
    "power_factor": _as_float(sample.get("power_factor", sample.get("PF"))),
    "frequency": _as_float(sample.get("frequency", sample.get("Hz"))),
  }


def _build_latest_result(sample, response_payload):
  sample_data = _sample_to_dict(sample)
  return {
    "success": True,
    "data": {
      **sample_data,
      "device_detected": response_payload["label"],
      "confidence": response_payload["confidence"],
      "model_version": response_payload["model_version"],
      "timestamp": response_payload["timestamp"],
    },
    "meta": {
      "label_source": response_payload.get("label_source"),
      "buffer": response_payload.get("buffer"),
      "raw_top": response_payload.get("raw_top"),
      "raw_second": response_payload.get("raw_second"),
    },
  }


def _extract_features_from_sample(sample: dict):
  global _LAST_RAW_SAMPLE
  if not isinstance(sample, dict):
    raise ValueError("sample harus object")

  v = _as_float(sample.get("voltage", sample.get("V")))
  i = _as_float(sample.get("current", sample.get("I")))
  p = _as_float(sample.get("power", sample.get("P")))
  pf = _as_float(sample.get("power_factor", sample.get("PF")))
  hz = _as_float(sample.get("frequency", sample.get("Hz")))

  # Sinkron dengan final_pipeline (8).ipynb v6:
  # ['voltage', 'current', 'power', 'power_factor', 'frequency',
  #  'apparent_power', 'reactive_power', 'power_ratio']
  pf = max(0.0, min(pf, 1.0))
  apparent_power = v * i
  reactive_power = apparent_power * np.sqrt(max(0.0, 1.0 - (pf ** 2)))
  power_ratio = p / (apparent_power + 1e-6)

  _LAST_RAW_SAMPLE = sample
  return np.array(
    [v, i, p, pf, hz, apparent_power, reactive_power, power_ratio],
    dtype=np.float32,
  )


def _load_dummy_samples():
  if not _DUMMY_FILE.exists():
    raise FileNotFoundError(f"Dummy file tidak ditemukan: {_DUMMY_FILE}")

  raw = _DUMMY_FILE.read_text(encoding="utf-8").strip()
  if not raw:
    raise ValueError("Dummy file kosong")

  if raw.startswith("["):
    samples = json.loads(raw)
  else:
    samples = [json.loads(line) for line in raw.splitlines() if line.strip()]

  if not isinstance(samples, list) or not samples:
    raise ValueError("Format dummy harus array JSON atau JSONL (per baris)")

  for item in samples:
    if not isinstance(item, dict):
      raise ValueError("Setiap item dummy harus object")

  return samples


def _run_samples(samples: list[dict], payload: dict | None):
  stride = int((payload or {}).get("stride", 1))
  stride = 1 if stride < 1 else stride
  update_blynk = bool((payload or {}).get("update_blynk", False))

  with _LOCK:
    _SEQ_BUFFER.clear()
    global _EMA_PROBS, _LAST_RAW_SAMPLE
    _EMA_PROBS = None
    _LAST_RAW_SAMPLE = None

  timeline = []
  last = None

  for index, sample in enumerate(samples, start=1):
    with _LOCK:
      if _should_reset_buffer_for_device_change(sample):
        _SEQ_BUFFER.clear()
        _EMA_PROBS = None

      features = _extract_features_from_sample(sample)
      _SEQ_BUFFER.append(features.tolist())
      sequence, received_len = _parse_sequence({"sequence": list(_SEQ_BUFFER)})

    result = _predict_from_sequence(sequence, received_len, payload)
    last = result

    if update_blynk:
      meta = _read_model_meta()
      _blynk_update("V6", result["label"])
      _blynk_update("V7", result["confidence"])
      _blynk_update("V8", meta.get("model_name") or "N/A")
      _blynk_update("V9", _now_iso())

    if index == 1 or index == len(samples) or index % stride == 0:
      timeline.append(
        {
          "step": index,
          "buffer": result["buffer"],
          "label": result["label"],
          "confidence": result["confidence"],
        },
      )

  if last is None:
    raise ValueError("Tidak ada sample untuk diproses")

  return last, timeline


def _predict_from_sequence(sequence: np.ndarray, received_len: int, payload: dict | None):
  global _EMA_PROBS, _REQUEST_COUNT
  _REQUEST_COUNT += 1

  labels = _load_labels()
  label_source = _get_label_source()
  model = _get_model()
  meta = _read_model_meta()

  input_shape = meta.get("input_shape") or []
  seq_len = int(input_shape[0]) if len(input_shape) >= 2 and isinstance(input_shape[0], int) else 99
  sequence = _scale_sequence(sequence, meta)
  probs = model.predict(sequence.reshape((1, seq_len, 8)), verbose=0)
  probs = np.array(probs).reshape((-1,))

  if probs.size != len(labels):
    raise RuntimeError(f"Output model {probs.size} tidak cocok dengan labels {len(labels)}")

  smoothing = str((payload or {}).get("smoothing", "ema")).lower()
  if smoothing == "ema":
    alpha = float((payload or {}).get("ema_alpha", 0.6))
    probs = _apply_smoothing(probs, alpha)
  else:
    alpha = None

  top_index = int(np.argmax(probs))
  top_label = labels[top_index]
  top_confidence = float(probs[top_index]) * 100.0

  sorted_indices = list(np.argsort(-probs))
  top3_indices = sorted_indices[:3]
  second_index = int(top3_indices[1]) if len(top3_indices) > 1 else top_index
  second_label = labels[second_index]
  second_confidence = float(probs[second_index]) * 100.0
  third_index = int(top3_indices[2]) if len(top3_indices) > 2 else top_index
  third_label = labels[third_index]
  third_confidence = float(probs[third_index]) * 100.0

  prefer_non_uncertain = bool((payload or {}).get("prefer_non_uncertain", True))
  uncertain_label = str((payload or {}).get("uncertain_label", "uncertain"))
  min_second_confidence = float((payload or {}).get("min_second_confidence", 25.0))

  chosen_index = top_index
  chosen_label = top_label
  chosen_confidence = top_confidence

  if prefer_non_uncertain and top_label == uncertain_label and second_label != uncertain_label and second_confidence >= min_second_confidence:
    chosen_index = second_index
    chosen_label = second_label
    chosen_confidence = second_confidence

  power_w = _power_from_sample(_LAST_RAW_SAMPLE)
  power_range = meta.get("power_range") or {}
  if chosen_label not in ("uncertain", "idle"):
    label_range = power_range.get(chosen_label)
    if label_range and not (label_range[0] <= power_w <= label_range[1] * 1.2):
      for alt_index in top3_indices[1:]:
        alt_label = labels[alt_index]
        alt_range = power_range.get(alt_label)
        alt_confidence = float(probs[alt_index]) * 100.0
        if alt_range and alt_range[0] <= power_w <= alt_range[1] * 1.2:
          chosen_index = alt_index
          chosen_label = alt_label
          chosen_confidence = alt_confidence
          break

  buffer_status = "READY" if received_len >= seq_len else "LOADING"
  buffer_bar = _format_buffer_bar(received_len, seq_len)
  print(
    f"[{_REQUEST_COUNT:05d}] Buffer {received_len}/99 {buffer_bar} {buffer_status} | "
    f"Detected {chosen_label} ({chosen_confidence:.1f}%) | "
    f"Top {top_label} ({top_confidence:.1f}%) | "
    f"smoothing={smoothing}{'' if alpha is None else f' alpha={alpha:.2f}'}"
  )

  return {
    "success": True,
    "label": chosen_label,
    "confidence": round(chosen_confidence, 1),
    "index": chosen_index,
    "model_version": meta.get("model_name") or "unknown_model",
    "label_source": label_source,
    "timestamp": _now_iso(),
    "buffer": {
      "received": received_len,
      "window": 99,
      "status": buffer_status,
      "bar": buffer_bar,
    },
    "raw_top": {
      "label": top_label,
      "confidence": round(top_confidence, 1),
      "index": top_index,
    },
    "raw_second": {
      "label": second_label,
      "confidence": round(second_confidence, 1),
      "index": second_index,
    },
  }



@app.get("/health")
def health():
  return jsonify(
    {
      "success": True,
      "model_dir": str(MODEL_DIR),
      "files": _get_model_files(),
    }
  )


@app.get("/")
def index():
  return jsonify(
    {
      "success": True,
      "message": "NILM ML service aktif",
      "model_dir": str(MODEL_DIR),
      "endpoints": [
        "/health",
        "/labels",
        "/model/files",
        "/model/files/config.json",
        "/model/files/metadata.json",
        "/latest",
        "/predict",
        "/ingest",
        "/thingsboard/ingest",
        "/reset",
        "/demo/dummy",
      ],
    }
  )


@app.get("/model/files")
def model_files():
  meta = _read_model_meta()
  return jsonify(
    {
      "success": True,
      "model_dir": str(MODEL_DIR),
      "model_name": meta.get("model_name"),
      "files": _get_model_files(),
    }
  )


@app.get("/model/files/<path:name>")
def model_file_content(name: str):
  try:
    path = _resolve_model_file(name)
  except ValueError as exc:
    return jsonify({"success": False, "error": str(exc)}), 400

  if not path.exists():
    return jsonify({"success": False, "error": f"File tidak ditemukan: {path.name}"}), 404

  if path.name in _MODEL_BINARY_FILES:
    return jsonify(
      {
        "success": True,
        "name": path.name,
        "path": str(path),
        "type": "binary",
        "size_bytes": path.stat().st_size,
        "content": None,
        "note": "File biner tidak ditampilkan, hanya metadata file.",
      }
    )

  return jsonify(
    {
      "success": True,
      "name": path.name,
      "path": str(path),
      "type": "text",
      "size_bytes": path.stat().st_size,
      "content": _read_text(path),
    }
  )

@app.get("/labels")
def labels():
  meta = _read_model_meta()
  runtime_labels = _load_labels()
  label_source = _get_label_source()
  labels_path = MODEL_DIR / "labels.json"
  configured_labels = None

  if labels_path.exists():
    configured_labels = _read_json(labels_path).get("labels", [])
    if not isinstance(configured_labels, list):
      configured_labels = []

  visible_labels = [item.strip() for item in (configured_labels or runtime_labels) if isinstance(item, str) and item.strip()]
  placeholders = [label for label in runtime_labels if label.startswith("unknown_")]
  return jsonify(
    {
      "success": True,
      "model_dir": str(MODEL_DIR),
      "model_name": meta.get("model_name"),
      "output_units": meta.get("output_units"),
      "labels": visible_labels,
      "label_source": label_source,
      "has_placeholders": len(placeholders) > 0,
      "placeholders": placeholders,
      "configured_label_count": len(visible_labels),
      "runtime_label_count": len(runtime_labels),
    },
  )


@app.get("/latest")
def latest():
  if _LATEST_RESULT is None:
    return jsonify({"success": False, "error": "Belum ada data telemetry yang masuk ke ML service."}), 404
  return jsonify(_LATEST_RESULT)


@app.post("/predict")
def predict():
  payload = request.get_json(silent=True)
  sequence, received_len = _parse_sequence(payload)
  try:
    response_payload = _predict_from_sequence(sequence, received_len, payload)
  except Exception as exc:
    return jsonify({"success": False, "error": str(exc)}), 500

  update_blynk = bool((payload or {}).get("update_blynk", False))
  blynk_result = None
  if update_blynk:
    meta = _read_model_meta()
    try:
      _blynk_update("V6", response_payload["label"])
      _blynk_update("V7", response_payload["confidence"])
      _blynk_update("V8", meta.get("model_name") or "N/A")
      _blynk_update("V9", _now_iso())
      blynk_result = {"updated": True, "pins": ["V6", "V7", "V8", "V9"]}
    except Exception as exc:
      blynk_result = {"updated": False, "error": str(exc)}

  response_payload["blynk"] = blynk_result
  return jsonify(_make_json_safe(response_payload))


@app.post("/ingest")
def ingest():
  global _LATEST_RESULT
  payload = request.get_json(silent=True) or {}
  sample = _normalize_sample(payload)

  with _LOCK:
    if _should_reset_buffer_for_device_change(sample):
      _SEQ_BUFFER.clear()
      _EMA_PROBS = None

    try:
      features = _extract_features_from_sample(sample)
    except Exception as exc:
      return jsonify({"success": False, "error": str(exc)}), 400

    _SEQ_BUFFER.append(features.tolist())
    sequence, received_len = _parse_sequence({"sequence": list(_SEQ_BUFFER)})

  try:
    response_payload = _predict_from_sequence(sequence, received_len, payload)
  except Exception as exc:
    return jsonify({"success": False, "error": str(exc)}), 500

  update_blynk = bool(payload.get("update_blynk", True))
  blynk_result = None
  if update_blynk:
    meta = _read_model_meta()
    try:
      _blynk_update("V6", response_payload["label"])
      _blynk_update("V7", response_payload["confidence"])
      _blynk_update("V8", meta.get("model_name") or "N/A")
      _blynk_update("V9", _now_iso())
      blynk_result = {"updated": True, "pins": ["V6", "V7", "V8", "V9"]}
    except Exception as exc:
      blynk_result = {"updated": False, "error": str(exc)}

  _LATEST_RESULT = _build_latest_result(sample, response_payload)
  response_payload["blynk"] = blynk_result
  response_payload["sample"] = _sample_to_dict(sample)
  return jsonify(_make_json_safe(response_payload))


@app.post("/thingsboard/ingest")
def thingsboard_ingest():
  return ingest()


@app.post("/reset")
def reset():
  global _EMA_PROBS, _LAST_RAW_SAMPLE, _PREV_POWER, _LATEST_RESULT
  with _LOCK:
    _SEQ_BUFFER.clear()
    _EMA_PROBS = None
    _LAST_RAW_SAMPLE = None
    _PREV_POWER = None
    _LATEST_RESULT = None
  return jsonify({"success": True})


@app.get("/demo/dummy")
def demo_dummy():
  payload = dict(request.args)
  payload["update_blynk"] = str(payload.get("update_blynk", "false")).lower() in ("1", "true", "yes")
  if "stride" in payload:
    try:
      payload["stride"] = int(payload["stride"])
    except Exception:
      payload["stride"] = 1

  try:
    samples = _load_dummy_samples()
    last, timeline = _run_samples(samples, payload)
  except Exception as exc:
    return jsonify({"success": False, "error": str(exc)}), 500

  return jsonify(
    {
      "success": True,
      "file": str(_DUMMY_FILE),
      "total_samples": len(samples),
      "result": {
        "label": last["label"],
        "confidence": last["confidence"],
        "buffer": last["buffer"],
      },
      "timeline": timeline,
    },
  )


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5001")))
