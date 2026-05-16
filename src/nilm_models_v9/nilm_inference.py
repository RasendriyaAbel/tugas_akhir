
"""
NILM Inference Utilities — v9 (multi-label)
Auto-generated dari training notebook nilm_training_v9.ipynb.

Penggunaan:
    from nilm_inference import NILMPredictor

    predictor = NILMPredictor(
        model_path  = "path/best_nilm_model.keras",
        meta_path   = "path/meta_nilm.json",
        scaler_path = "path/scaler_nilm.pkl",
    )

    raw = {"voltage": 211.5, "current": 1.07, "power": 225.0,
           "power_factor": 1.00, "frequency": 50.0}

    result = predictor.predict(raw)
    print(result)
    # {"label": "hair_dryer", "active_devices": ["hair_dryer"],
    #  "probs": [("charger_hp", 0.02), ("hair_dryer", 0.95), ...],
    #  "buffer_fill": 30, "power_w": 225.0, "is_ready": True}
"""

import json
import numpy as np
import joblib
import tensorflow as tf
from collections import deque
from typing import Optional, List


def build_feature_vector(raw: dict) -> np.ndarray:
    """
    Bangun feature vector dari dict sensor PZEM-004T.

    Input dict keys: voltage, current, power, power_factor, frequency
    Output: np.ndarray shape (8,)

    ⚠️  Urutan elemen ini WAJIB identik dengan FEATURE_COLS saat training.
        Jangan ubah urutan.
    """
    def _f(d, k, fallback=0.0):
        try:
            v = float(d[k])
            return v if np.isfinite(v) else fallback
        except Exception:
            return fallback

    v  = _f(raw, "voltage",      220.0)
    i  = _f(raw, "current",        0.0)
    p  = _f(raw, "power",          0.0)
    pf = _f(raw, "power_factor",   0.9)
    hz = _f(raw, "frequency",     50.0)

    apparent_power = v * i
    reactive_power = apparent_power * np.sqrt(max(0.0, 1.0 - pf ** 2))
    power_ratio    = p / (apparent_power + 1e-6)

    return np.array([
        v, i, p, pf, hz,
        apparent_power,
        reactive_power,
        power_ratio,
    ], dtype=np.float32)


def _weighted_bce_placeholder(y_true, y_pred):
    """Placeholder loss untuk load_model — tidak dipakai saat inference."""
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

def _exact_match_placeholder(y_true, y_pred):
    """Placeholder metric untuk load_model — tidak dipakai saat inference."""
    pred_bin = tf.cast(y_pred >= 0.5, tf.float32)
    match    = tf.reduce_all(tf.equal(pred_bin, y_true), axis=1)
    return tf.reduce_mean(tf.cast(match, tf.float32))


class NILMPredictor:
    """
    Stateful NILM predictor dengan rolling buffer dan smoothing.

    Satu instance per sesi monitoring. Panggil reset() saat koneksi putus.
    """

    def __init__(
        self,
        model_path:  str,
        meta_path:   str,
        scaler_path: str,
    ):
        # Custom objects diperlukan karena loss & metric adalah fungsi custom
        # TemporalSum harus didefinisikan sebelum load_model
        try:
            _reg = tf.keras.saving.register_keras_serializable(package='nilm_v9')
        except AttributeError:
            try:
                _reg = tf.keras.utils.register_keras_serializable(package='nilm_v9')
            except AttributeError:
                _reg = lambda cls: cls

        @_reg
        class _TemporalSum(tf.keras.layers.Layer):
            def call(self, x):
                return tf.reduce_sum(x, axis=1)
            def get_config(self):
                return super().get_config()

        self.model  = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'TemporalSum'  : _TemporalSum,
                'weighted_bce' : _weighted_bce_placeholder,
                'exact_match'  : _exact_match_placeholder,
            }
        )
        self.scaler = joblib.load(scaler_path)

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.window_size      = self.meta["window_size"]
        self.n_features       = self.meta["n_features"]
        self.devices          = self.meta["devices"]           # ['charger_hp', 'hair_dryer', 'kipas', 'laptop']
        self.threshold        = self.meta.get("threshold", 0.5)
        self.noise_floor_w    = self.meta.get("noise_floor_w", 3.0)
        self.smooth_n         = self.meta.get("smooth_n", 5)
        self.transition_delta = self.meta.get("transition_delta", 30.0)
        self.power_range      = self.meta.get("power_range", {})

        self._buffer     = deque(maxlen=self.window_size)
        self._pred_queue = deque(maxlen=self.smooth_n)
        self._prev_power: Optional[float] = None

    def reset(self) -> None:
        """Reset buffer dan smoothing queue."""
        self._buffer.clear()
        self._pred_queue.clear()
        self._prev_power = None

    def predict(self, raw: dict) -> dict:
        """
        Inferensi satu step dari dict data sensor.

        FIX: Deteksi idle & transisi sekarang menggunakan feature vector
        lengkap (8 fitur) konsisten dengan input model, bukan hanya power.

        Returns:
            label          : str        — label prediksi akhir (e.g. "kipas+laptop")
            active_devices : list[str]  — device yang diprediksi aktif
            probs          : list[tuple]— prob tiap device [(device, prob), ...]
            buffer_fill    : int        — jumlah sampel dalam buffer
            power_w        : float      — power saat ini (W)
            apparent_w     : float      — apparent power saat ini (VA)
            is_ready       : bool       — True jika buffer penuh
        """
        def _f(k, fb=0.0):
            try:
                v = float(raw[k])
                return v if np.isfinite(v) else fb
            except Exception:
                return fb

        # Bangun feature vector lengkap (8 fitur) — konsisten dengan training
        feat  = build_feature_vector(raw)
        p_now = feat[2]   # index 2 = power (lihat FEATURE_COLS di training)
        s_now = feat[5]   # index 5 = apparent_power

        # ── Deteksi transisi beban menggunakan apparent_power ──────────────
        # Apparent power (V×I) lebih stabil daripada active power untuk
        # mendeteksi perubahan beban induktif (kipas, dll).
        # Jika apparent_power berubah > transition_delta → reset buffer.
        if self._prev_power is not None:
            if abs(s_now - self._prev_power) > self.transition_delta:
                self._buffer.clear()
                self._pred_queue.clear()
        self._prev_power = s_now  # simpan apparent_power sebagai referensi

        # ── Deteksi idle menggunakan active power ──────────────────────────
        # Tetap pakai active power (W) untuk idle karena threshold dalam Watt.
        if p_now < self.noise_floor_w:
            self.reset()
            return {
                "label"          : "idle",
                "active_devices" : [],
                "probs"          : [(d, 0.0) for d in self.devices],
                "buffer_fill"    : 0,
                "power_w"        : round(p_now, 2),
                "apparent_va"    : round(s_now, 2),
                "is_ready"       : False,
            }

        # ── Isi buffer dengan feature vector lengkap ───────────────────────
        # feat sudah dibangun di atas — tidak perlu build ulang
        self._buffer.append(feat)
        buf_len = len(self._buffer)

        MIN_BUF = max(10, self.window_size // 3)
        if buf_len < MIN_BUF:
            return {
                "label"          : "filling_buffer",
                "active_devices" : [],
                "probs"          : [(d, 0.0) for d in self.devices],
                "buffer_fill"    : buf_len,
                "power_w"        : round(p_now, 2),
                "apparent_va"    : round(s_now, 2),
                "is_ready"       : False,
            }

        # Pad / slice buffer ke window_size
        X_raw = np.array(self._buffer, dtype="float32")
        if buf_len < self.window_size:
            repeat = int(np.ceil(self.window_size / buf_len))
            X_pad  = np.tile(X_raw, (repeat, 1))[:self.window_size]
        else:
            X_pad = X_raw[-self.window_size:]

        # Scale & predict
        X_sc  = self.scaler.transform(X_pad)
        X_in  = np.expand_dims(X_sc, axis=0)
        probs = self.model.predict(X_in, verbose=0)[0]   # shape: (N_DEVICES,)

        # Decode multi-label: tiap device independen
        active_devices = [
            dev for dev, p in zip(self.devices, probs)
            if p >= self.threshold
        ]
        raw_label = '+'.join(active_devices) if active_devices else 'idle'

        # Majority-vote smoothing (per kombinasi label string)
        self._pred_queue.append(raw_label)
        final_label = max(set(self._pred_queue), key=self._pred_queue.count)

        # Decode active_devices dari final_label (setelah smoothing)
        if final_label == 'idle':
            final_active = []
        else:
            final_active = final_label.split('+')

        probs_out = [(dev, round(float(p), 4)) for dev, p in zip(self.devices, probs)]

        return {
            "label"          : final_label,
            "active_devices" : final_active,
            "probs"          : probs_out,
            "buffer_fill"    : buf_len,
            "power_w"        : round(p_now, 2),
            "apparent_va"    : round(s_now, 2),
            "is_ready"       : buf_len >= self.window_size,
        }
