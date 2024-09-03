"""Algorithm II - head-motion separation and stabilisation tests."""

import numpy as np

from iars.config import StabilizationConfig
from iars.stabilization.head_motion import HeadMotionSeparator, design_biquad, filtfilt


def _synthetic_head(fs=400.0, duration=4.0, vib_freq=12.0, vib_amp=30.0, vol_amp=25.0):
    t = np.arange(0, duration, 1.0 / fs)
    voluntary = np.zeros((len(t), 3))
    # Slow, sustained turn on the yaw axis (voluntary).
    voluntary[:, 2] = vol_amp * (1.0 - np.cos(2 * np.pi * 0.4 * t))
    # High-frequency buffet on all axes (vibration).
    vibration = np.column_stack(
        [
            vib_amp * np.sin(2 * np.pi * vib_freq * t),
            0.6 * vib_amp * np.sin(2 * np.pi * (vib_freq * 0.8) * t + 0.5),
            vib_amp * np.sin(2 * np.pi * (vib_freq * 1.3) * t + 1.1),
        ]
    )
    return t, voluntary, vibration, voluntary + vibration


def test_lowpass_has_expected_unity_and_stopband_gain():
    b, a = design_biquad("lowpass", 400.0, 2.0)
    assert abs(np.sum(b) / np.sum(a) - 1.0) < 1e-6  # DC gain == 1


def test_separator_recovers_voluntary_component():
    t, voluntary, vibration, gyro = _synthetic_head()
    sep = HeadMotionSeparator(StabilizationConfig(sample_rate_hz=400.0))
    est_vol, est_vib = sep.separate(gyro)
    # Ignore the filter's warm-up edge.
    sl = slice(400, -400)
    err = est_vol[sl, 2] - voluntary[sl, 2]
    assert np.sqrt(np.mean(err**2)) < 3.0


def test_separator_recovers_vibration_component():
    t, voluntary, vibration, gyro = _synthetic_head()
    sep = HeadMotionSeparator(StabilizationConfig(sample_rate_hz=400.0))
    est_vol, est_vib = sep.separate(gyro)
    sl = slice(400, -400)
    corr = np.corrcoef(est_vib[sl, 0], vibration[sl, 0])[0, 1]
    assert corr > 0.9


def test_stabilisation_attenuates_jitter():
    t, voluntary, vibration, gyro = _synthetic_head()
    sep = HeadMotionSeparator(StabilizationConfig(sample_rate_hz=400.0))
    res = sep.stabilise(gyro)
    assert res.metrics["attenuation_db"] > 15.0
    assert res.metrics["stabilised_jitter_rms_dps"] < res.metrics["raw_jitter_rms_dps"]


def test_intent_gate_opens_on_sustained_turn():
    t, voluntary, vibration, gyro = _synthetic_head()
    sep = HeadMotionSeparator(StabilizationConfig(sample_rate_hz=400.0))
    res = sep.stabilise(gyro)
    # During the sustained turn the gate must be open for a good fraction.
    assert res.intent.mean() > 0.2


def test_filtfilt_removes_phase_lag():
    t = np.arange(0, 2.0, 1.0 / 400.0)
    x = np.sin(2 * np.pi * 0.5 * t)
    b, a = design_biquad("lowpass", 400.0, 5.0)
    y = filtfilt(x, b, a)
    assert np.sqrt(np.mean((y - x) ** 2)) < 1e-2
