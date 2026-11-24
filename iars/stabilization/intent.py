"""Learned intent gate for the head-motion stabiliser.

The heuristic gate in :mod:`iars.stabilization.head_motion` thresholds the
low-pass rate with hysteresis.  It is cheap and predictable, but it conflates
"the wearer is turning slowly" with "the wearer is being shaken hard enough that
some low-frequency energy leaks through the split".  On the POC data the two
cases overlap: a hard buffet at 6 Hz leaves a visible low-frequency envelope.

This module replaces the threshold with a small classifier over windowed gyro
features.  It is deliberately tiny - eight hand-designed features and a
logistic regression trained offline - because it has to run inside the 400 Hz
loop on the flight computer's MCU, not on a GPU.

Features (computed over a 0.32 s window, 0.1 s hop):

===================  ==========================================================
``lp_mag``           magnitude of the low-pass (voluntary-band) rate
``hp_rms``           RMS of the high-pass (buffeting-band) rate
``hp_lp_ratio``      ``hp_rms / (lp_mag + eps)`` - how oscillatory the motion is
``raw_rms``          RMS of the raw rate
``zcr_hz``           zero-crossing rate of the dominant axis
``centroid_hz``      spectral centroid of the raw rate
``mean_abs_diff``    mean absolute sample-to-sample change
``acf_peak``         normalised autocorrelation peak in the 5-25 Hz lag band
===================  ==========================================================

The model is trained on episodes where the voluntary and buffeting components
are known *by construction*, so the label is the generator's ground truth, not
the heuristic's own output.  That makes the reported accuracy a fair
comparison rather than a self-fulfilling one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .head_motion import HeadMotionSeparator, biquad_filt, design_biquad

FEATURE_NAMES = [
    "lp_mag",
    "hp_rms",
    "hp_lp_ratio",
    "raw_rms",
    "zcr_hz",
    "centroid_hz",
    "mean_abs_diff",
    "acf_peak",
]

DEFAULT_WEIGHTS = Path(__file__).with_name("intent_weights.json")


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def extract_features(
    gyro_dps: np.ndarray,
    fs: float,
    window: int = 128,
    hop: int = 40,
    lp_cutoff_hz: float = 1.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Windowed features and window-centre sample indices.

    ``gyro_dps`` is ``(N, 3)``.  Returns ``X`` of shape ``(M, 8)`` and the centre
    index of every window so predictions can be mapped back onto the stream.
    """
    gyro = np.atleast_2d(np.asarray(gyro_dps, dtype=float))
    if gyro.shape[0] < gyro.shape[1]:  # transpose a single 3-vector
        gyro = gyro.T
    n = len(gyro)
    window = min(window, n)
    hop = max(1, hop)

    b, a = design_biquad("lowpass", fs, lp_cutoff_hz)
    lp = np.column_stack([biquad_filt(gyro[:, k], b, a) for k in range(3)])
    hp = gyro - lp

    mag = np.linalg.norm(gyro, axis=1)
    lp_mag_series = np.linalg.norm(lp, axis=1)
    hp_rms_series = np.sqrt(np.sum(hp * hp, axis=1))

    # Dominant axis for zero-crossing / spectral descriptors.
    axis = int(np.argmax(np.var(gyro, axis=0)))
    sig = gyro[:, axis]

    freq = np.fft.rfftfreq(window, 1.0 / fs)
    win = np.hanning(window)
    band_lo = max(1, int(np.searchsorted(freq, 5.0)))
    band_hi = max(band_lo + 1, int(np.searchsorted(freq, 25.0)))
    min_lag = max(1, int(fs / 25.0))
    max_lag = max(min_lag + 1, int(fs / 5.0))

    centres, rows = [], []
    for start in range(0, max(1, n - window + 1), hop):
        end = start + window
        seg = sig[start:end]
        seg_mag = mag[start:end]
        centre = start + window // 2
        centres.append(centre)

        lp_mag = float(np.mean(lp_mag_series[start:end]))
        hp_rms = float(np.sqrt(np.mean(hp_rms_series[start:end] ** 2)))
        raw_rms = float(np.sqrt(np.mean(seg_mag**2)))

        # Zero crossings of the mean-removed dominant-axis signal.
        centred = seg - seg.mean()
        sign = np.sign(centred)
        sign[sign == 0] = 1
        zcr = float(np.count_nonzero(np.diff(sign)) / (window / fs))

        # Spectral centroid.
        spec = np.abs(np.fft.rfft(centred * win))
        denom = float(np.sum(spec))
        centroid = float(np.sum(freq * spec) / denom) if denom > 1e-9 else 0.0

        # Autocorrelation peak over the buffeting lag band.  On very short
        # windows the band exceeds the window length, so the feature degrades to
        # zero rather than raising.
        x = centred - centred.mean()
        acf = np.correlate(x, x, mode="full")[window - 1 :]
        acf = acf / (acf[0] + 1e-9)
        lag_hi = min(max_lag, window)
        acf_peak = float(np.max(acf[min_lag:lag_hi])) if lag_hi > min_lag else 0.0

        rows.append(
            [
                lp_mag,
                hp_rms,
                hp_rms / (lp_mag + 1e-3),
                raw_rms,
                zcr,
                centroid,
                float(np.mean(np.abs(np.diff(seg)))) if window > 1 else 0.0,
                acf_peak,
            ]
        )

    return np.asarray(rows, dtype=float), np.asarray(centres, dtype=int)


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
class IntentClassifier:
    """L2-regularised logistic regression on standardised window features."""

    def __init__(self, mean, scale, coef, intercept, threshold: float = 0.5):
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = float(intercept)
        self.threshold = float(threshold)

    # -- training ---------------------------------------------------------
    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1500,
        lr: float = 0.5,
        l2: float = 1e-3,
        threshold: float = 0.5,
    ) -> "IntentClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-9] = 1.0
        Z = (X - mean) / scale

        # Balance the classes: the positive (turning) class is the minority.
        pos = float(np.sum(y == 1))
        neg = float(np.sum(y == 0))
        w_pos = 0.5 * len(y) / max(pos, 1.0)
        w_neg = 0.5 * len(y) / max(neg, 1.0)
        sample_w = np.where(y > 0.5, w_pos, w_neg)

        coef = np.zeros(Z.shape[1])
        intercept = 0.0
        for _ in range(epochs):
            logits = Z @ coef + intercept
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            err = (p - y) * sample_w
            grad = Z.T @ err / len(y) + l2 * coef
            grad_b = float(np.mean(err))
            coef -= lr * grad
            intercept -= lr * grad_b
        return cls(mean, scale, coef, intercept, threshold)

    # -- inference --------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = (np.atleast_2d(X) - self.mean) / self.scale
        logits = Z @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X) >= self.threshold

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "feature_names": FEATURE_NAMES,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "threshold": self.threshold,
            "training": "synthetic episodes with known voluntary/vibration components",
        }

    def save(self, path: str | Path = DEFAULT_WEIGHTS) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path = DEFAULT_WEIGHTS) -> "IntentClassifier":
        d = json.loads(Path(path).read_text())
        return cls(d["mean"], d["scale"], d["coef"], d["intercept"], d.get("threshold", 0.5))

    @classmethod
    def load_default(cls) -> "IntentClassifier":
        return cls.load(DEFAULT_WEIGHTS)

    # -- convenience ------------------------------------------------------
    def predict_stream(self, gyro_dps: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
        """Predict on a stream; returns (centre indices, boolean intent)."""
        X, centres = extract_features(gyro_dps, fs)
        if len(X) == 0:
            return centres, np.zeros(0, dtype=bool)
        return centres, self.predict(X)


def learned_intent_gate(gyro_dps: np.ndarray, fs: float, clf: IntentClassifier | None = None) -> np.ndarray:
    """Per-sample intent mask by hold-last-value between windows."""
    clf = clf or IntentClassifier.load_default()
    gyro = np.atleast_2d(np.asarray(gyro_dps, dtype=float))
    if gyro.shape[0] < gyro.shape[1]:
        gyro = gyro.T
    centres, pred = clf.predict_stream(gyro, fs)
    if len(centres) == 0:
        return np.zeros(len(gyro), dtype=bool)
    idx = np.clip(centres, 0, len(gyro) - 1)
    return np.interp(np.arange(len(gyro)), idx, pred.astype(float)) > 0.5


def heuristic_intent_gate(gyro_dps: np.ndarray, fs: float, threshold_dps: float = 8.0) -> np.ndarray:
    """Reference gate: the 2024 threshold-on-low-pass heuristic."""
    sep = HeadMotionSeparator()
    sep.cfg = type(sep.cfg)(
        sample_rate_hz=fs,
        intent_threshold_dps=threshold_dps,
        intent_hold_s=sep.cfg.intent_hold_s,
    )
    voluntary, _ = sep.separate(gyro_dps)
    return sep.detect_intent(voluntary)
