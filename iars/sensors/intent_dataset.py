"""Labelled episodes for the learned intent gate.

The point of this generator is to have **known ground truth**: each episode is
synthesised from an explicit voluntary component and an explicit buffeting
component, so the label ("the wearer is deliberately turning") comes from the
generator rather than from the heuristic we are trying to beat.  That is what
makes the accuracy comparison in ``tools/train_intent.py`` meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TURN_THRESHOLD_DPS = 8.0


@dataclass
class Episode:
    gyro_dps: np.ndarray   # (N, 3) measured rate
    voluntary_dps: np.ndarray  # (N, 3) ground-truth voluntary component
    vibration_dps: np.ndarray  # (N, 3) ground-truth buffeting component
    label: np.ndarray      # (N,) bool - voluntary magnitude above threshold
    kind: str


def _turn_profile(t: np.ndarray, amp: float, freq: float, axes: np.ndarray) -> np.ndarray:
    shape = 0.5 * (1.0 - np.cos(2.0 * np.pi * freq * t))
    out = np.zeros((len(t), 3))
    for axis, gain in zip(axes, (1.0, 0.5, 0.8)):
        out[:, axis] = amp * gain * shape
    return out


def generate_episode(
    rng: np.random.Generator,
    fs: float = 400.0,
    duration: float = 6.0,
) -> Episode:
    n = int(duration * fs)
    t = np.arange(n) / fs
    kind = str(rng.choice(["turn", "mixed", "shake", "still"], p=[0.30, 0.35, 0.25, 0.10]))

    # ---- buffeting: one or two tones in the 5-25 Hz band -----------------
    vib_amp = rng.uniform(8.0, 55.0)
    if kind == "still":
        vib_amp *= 0.25
    vibration = np.zeros((n, 3))
    for axis in range(3):
        f1 = rng.uniform(5.0, 12.0)
        f2 = rng.uniform(12.0, 26.0)
        p1, p2 = rng.uniform(0, 2 * np.pi, size=2)
        vibration[:, axis] = vib_amp * (
            0.7 * np.sin(2 * np.pi * f1 * t + p1) + 0.5 * np.sin(2 * np.pi * f2 * t + p2)
        )

    # ---- voluntary: none / sustained / toggling -------------------------
    axes = rng.permutation(3)[: rng.integers(1, 3)]
    amp = rng.uniform(12.0, 60.0)
    freq = rng.uniform(0.2, 1.0)
    voluntary = np.zeros((n, 3))
    if kind == "turn":
        voluntary = _turn_profile(t, amp, freq, axes)
    elif kind == "mixed":
        cursor = 0.0
        while cursor < duration - 0.6:
            if rng.random() < 0.6:
                span = rng.uniform(0.8, 2.0)
                start = int(cursor * fs)
                stop = min(n, start + int(span * fs))
                seg_t = t[start:stop] - t[start]
                voluntary[start:stop] = _turn_profile(seg_t, amp, freq, axes)
                cursor += span
            cursor += rng.uniform(0.4, 1.2)

    noise = 0.6 * rng.standard_normal((n, 3))
    gyro = voluntary + vibration + noise
    label = np.linalg.norm(voluntary, axis=1) > TURN_THRESHOLD_DPS
    return Episode(gyro, voluntary, vibration, label, kind)


def make_episodes(n_episodes: int = 160, seed: int = 0, fs: float = 400.0) -> list[Episode]:
    rng = np.random.default_rng(seed)
    return [generate_episode(rng, fs=fs) for _ in range(n_episodes)]
