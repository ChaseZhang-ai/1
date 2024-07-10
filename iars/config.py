"""Tuning parameters for the IARS sensor stack.

The defaults are the values used by the 2024 POC firmware profile, re-expressed
in SI units.  They are deliberately conservative: the flight computer would
tighten them per discipline (freefall, smoke diving, motorcycle) during
bench calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RedundancyConfig:
    # 1-sigma noise of each altitude source, metres.
    source_sigma: dict = field(
        default_factory=lambda: {
            "baro": 0.35,
            "gnss": 1.20,
            "inertial": 0.90,
            "pitot": 1.60,
        }
    )
    # Gross-error gate: two sources are compatible within this many sigmas.
    mad_gate: float = 4.5
    # Innovation gate for the sequential combiner (chi-square, 1 dof, 99.9%).
    chi2_gate: float = 10.83
    # EWMA coefficient used to track slowly varying per-source bias.
    bias_alpha: float = 0.01
    # Below this many healthy sources the output is flagged degraded.
    min_healthy: int = 2


@dataclass(frozen=True)
class StabilizationConfig:
    sample_rate_hz: float = 400.0
    # Split between voluntary head rotation and buffeting vibration.
    voluntary_cutoff_hz: float = 1.8
    vibration_band_hz: tuple = (4.0, 30.0)
    # Display follow gain for each component.
    voluntary_gain: float = 1.0
    vibration_gain: float = 0.05
    # Intent detector: |low-pass rate| above this for `hold` seconds means the
    # wearer is deliberately turning rather than being shaken.
    intent_threshold_dps: float = 8.0
    intent_hold_s: float = 0.15


@dataclass(frozen=True)
class StereoConfig:
    # Physical panel geometry of the binocular waveguide module.
    ipd_m: float = 0.064
    focal_px: float = 1180.0
    image_size_px: tuple = (1920, 1080)
    # Acceptance thresholds (ANSI/HFES-style viewing comfort limits).
    max_vertical_divergence_arcmin: float = 10.0
    max_relative_tilt_deg: float = 0.50
    max_scale_error_pct: float = 1.00
    max_horizontal_divergence_deg: float = 1.50


@dataclass(frozen=True)
class FilterConfig:
    gyro_noise_dps: float = 0.08
    accel_noise_mps2: float = 0.03
    gyro_bias_rw: float = 2.0e-5
    accel_bias_rw: float = 3.0e-4
    gnss_pos_sigma_m: float = 1.5
    gnss_vel_sigma_mps: float = 0.15
    baro_sigma_m: float = 0.40
    visual_pos_sigma_m: float = 0.12
    visual_att_sigma_rad: float = 0.004
    zupt_sigma_mps: float = 0.02
    chi2_gate_3d: float = 16.27  # 99.9%, 3 dof
    chi2_gate_1d: float = 10.83
    chi2_gate_6d: float = 22.46


def default_params() -> dict:
    return {
        "redundancy": RedundancyConfig(),
        "stabilization": StabilizationConfig(),
        "stereo": StereoConfig(),
        "filter": FilterConfig(),
    }


def deg2rad(x):
    return np.deg2rad(x)


def rad2deg(x):
    return np.rad2deg(x)
