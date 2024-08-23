"""Algorithm II - Head-motion stabilisation for a see-through waveguide HUD.

A skydiver's head moves for two very different reasons at the same time: the
wearer *deliberately* turns it (a slow, sustained rotation under 2 Hz) and the
airflow *shakes* it (buffeting between roughly 4 and 30 Hz).  A naive
"follow the head" display follows both, and the symbology smears into an
unreadable blur exactly when it matters most.

The separator splits the 3-axis gyro rate into a voluntary component and a
vibration component, then regenerates the display orientation from the
voluntary component only, with a small, deliberately chosen leak of the
vibration term so the horizon still feels alive rather than frozen.

* Splitting is a pair of complementary biquad filters (a low-pass for the
  voluntary term, the complementary high-pass for vibration).  Cutoffs are
  tuned to the freefall torso-spectrum measured on the POC data campaign.
* An intent detector with hysteresis distinguishes "wearer is turning" from
  "wearer is being shaken": a low-pass rate above ``intent_threshold_dps`` held
  for ``intent_hold_s`` opens the follow gate.
* The output is a stabilised attitude quaternion plus quantitative jitter
  metrics (RMS jitter before/after and the attenuation in dB at the dominant
  vibration line) used by the test suite and the tuning dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import StabilizationConfig
from ..slam import frames


def design_biquad(kind: str, fs: float, f0: float, q: float = 0.7071) -> tuple:
    """RBJ audio-cookbook biquad coefficients (returns b, a with a0 == 1)."""
    w0 = 2.0 * np.pi * f0 / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    if kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bandpass":
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    else:
        raise ValueError(f"unknown filter kind {kind!r}")
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    return b, a


def biquad_filt(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Zero-phase-ish forward filtering (single pass, causal, initial state = 0)."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    z1 = z2 = 0.0
    for i, xi in enumerate(x):
        yi = b[0] * xi + z1
        z1 = b[1] * xi - a[1] * yi + z2
        z2 = b[2] * xi - a[2] * yi
        y[i] = yi
    return y


def filtfilt(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Forward-backward pass: removes the phase lag that would break metrics."""
    return biquad_filt(biquad_filt(x, b, a)[::-1], b, a)[::-1]


@dataclass
class StabilizationResult:
    voluntary_dps: np.ndarray
    vibration_dps: np.ndarray
    stabilised_attitude: np.ndarray
    raw_attitude: np.ndarray
    intent: np.ndarray
    metrics: dict


class HeadMotionSeparator:
    def __init__(self, config: StabilizationConfig | None = None):
        self.cfg = config or StabilizationConfig()
        fs = self.cfg.sample_rate_hz
        self._lp = design_biquad("lowpass", fs, self.cfg.voluntary_cutoff_hz)
        lo, hi = self.cfg.vibration_band_hz
        band = np.sqrt(lo * hi)
        self._bp = design_biquad("bandpass", fs, band, q=band / (hi - lo))

    # ------------------------------------------------------------------ core
    def separate(self, gyro_dps: np.ndarray) -> tuple:
        """Split an (N, 3) gyro-rate series into voluntary and vibration parts."""
        gyro_dps = np.atleast_2d(np.asarray(gyro_dps, dtype=float))
        if gyro_dps.shape[0] == 1:
            gyro_dps = gyro_dps.T
        voluntary = np.column_stack([filtfilt(gyro_dps[:, k], *self._lp) for k in range(3)])
        vibration = gyro_dps - voluntary
        return voluntary, vibration

    def detect_intent(self, voluntary_dps: np.ndarray) -> np.ndarray:
        """Hysteresis intent gate: True while the wearer is deliberately turning."""
        mag = np.linalg.norm(voluntary_dps, axis=1)
        hold = max(1, int(self.cfg.intent_hold_s * self.cfg.sample_rate_hz))
        intent = np.zeros(len(mag), dtype=bool)
        above = mag > self.cfg.intent_threshold_dps
        run = 0
        for i, a in enumerate(above):
            run = run + 1 if a else 0
            intent[i] = run >= hold
        # Latch: once intent is set, keep the gate open for the hold window.
        for i in range(len(intent)):
            if intent[i]:
                intent[i : min(len(intent), i + hold)] = True
        return intent

    def stabilise(self, gyro_dps: np.ndarray, accel_mps2: np.ndarray | None = None) -> StabilizationResult:
        """Regenerate a jitter-free display attitude from the raw gyro stream.

        The raw attitude integrates the full rate (what a naive HUD would show);
        the stabilised attitude integrates ``voluntary_gain * voluntary +
        vibration_gain * vibration`` - i.e. it follows the wearer's intent and
        rejects the buffet.
        """
        gyro_dps = np.atleast_2d(np.asarray(gyro_dps, dtype=float))
        if gyro_dps.shape[0] == 1:
            gyro_dps = gyro_dps.T

        voluntary, vibration = self.separate(gyro_dps)
        intent = self.detect_intent(voluntary)
        dt = 1.0 / self.cfg.sample_rate_hz

        # Display rate = follow intent, reject buffet.  When the wearer is not
        # intentionally turning, hold the frame (frozen symbology) rather than
        # integrating noise.
        follow = np.where(intent[:, None], self.cfg.voluntary_gain, 0.0)
        display_rate_dps = follow * voluntary + self.cfg.vibration_gain * vibration

        raw_q, stab_q = [], []
        q_raw = np.array([1.0, 0.0, 0.0, 0.0])
        q_stab = np.array([1.0, 0.0, 0.0, 0.0])
        for i in range(len(gyro_dps)):
            q_raw = frames.quat_normalize(
                frames.quat_mul(q_raw, frames.quat_exp(np.deg2rad(gyro_dps[i] * dt)))
            )
            q_stab = frames.quat_normalize(
                frames.quat_mul(q_stab, frames.quat_exp(np.deg2rad(display_rate_dps[i] * dt)))
            )
            raw_q.append(q_raw.copy())
            stab_q.append(q_stab.copy())

        raw_q = np.array(raw_q)
        stab_q = np.array(stab_q)
        metrics = self._metrics(gyro_dps, voluntary, vibration, display_rate_dps)
        return StabilizationResult(voluntary, vibration, stab_q, raw_q, intent, metrics)

    # ------------------------------------------------------------------ metrics
    def _bandpassed(self, rate_dps: np.ndarray) -> np.ndarray:
        rate_dps = np.atleast_2d(rate_dps)
        return np.column_stack([filtfilt(rate_dps[:, k], *self._bp) for k in range(3)])

    def _metrics(self, gyro, voluntary, vibration, display_rate) -> dict:
        # Jitter is the energy that survives in the vibration band.  A naive HUD
        # shows the raw rate; the stabilised HUD shows the display rate.
        raw_jitter = _rms(self._bandpassed(gyro))
        stab_jitter = _rms(self._bandpassed(display_rate))
        attenuation_db = 20.0 * np.log10(max(raw_jitter, 1e-9) / max(stab_jitter, 1e-9))
        return {
            "raw_jitter_rms_dps": float(raw_jitter),
            "stabilised_jitter_rms_dps": float(stab_jitter),
            "attenuation_db": float(attenuation_db),
            "vibration_rms_dps": float(_rms(vibration)),
            "tracking_rmse_dps": float(_rms(display_rate - voluntary)),
            "intent_fraction": float(np.mean(np.linalg.norm(voluntary, axis=1) > self.cfg.intent_threshold_dps)),
            "display_latency_ms": float(1000.0 / (2.0 * np.pi * self.cfg.voluntary_cutoff_hz)),
        }


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(np.sum(x * x, axis=-1))))
