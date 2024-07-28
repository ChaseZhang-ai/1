"""Synthetic flight data generator.

The 2024 POC campaign logged ~40 minutes of real freefall, canopy and smoke
diving data, but those logs contain personal telemetry and cannot be published.
This module regenerates an equivalent, fully synthetic dataset with the same
salient statistics (entry-to-terminal velocity profile, buffeting spectrum,
GNSS outage under canopy) so the algorithms can be exercised and benchmarked in
the open.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..slam import frames

G = 9.80665


@dataclass
class FlightData:
    t: np.ndarray
    dt: float
    pos: np.ndarray          # (N, 3) ENU truth, metres
    vel: np.ndarray          # (N, 3) ENU truth, m/s
    quat: np.ndarray         # (N, 4) body->world truth
    gyro_true: np.ndarray    # (N, 3) rad/s
    accel_true: np.ndarray   # (N, 3) m/s^2, specific force
    gyro_meas: np.ndarray    # (N, 3) rad/s, noisy
    accel_meas: np.ndarray   # (N, 3) m/s^2, noisy
    baro: np.ndarray         # (N,) static-pressure altitude, metres
    gnss_pos: np.ndarray     # (N, 3) NaN during outage
    gnss_vel: np.ndarray     # (N, 3) NaN during outage
    pitot_airspeed: np.ndarray
    visual_valid: np.ndarray  # (N,) bool
    truth_altitude: np.ndarray


def _vterm_profile(t: np.ndarray) -> np.ndarray:
    """Terminal-velocity envelope: fast freefall -> deployment -> canopy."""
    vterm = np.full_like(t, 55.0)
    # Slow to aircraft-equivalent before exit.
    vterm[t < 3.0] = 70.0
    # Deployment ramp between t=40 s and t=43 s.
    ramp = np.clip((t - 40.0) / 3.0, 0.0, 1.0)
    vterm = vterm * (1.0 - ramp) + 5.0 * ramp
    return vterm


def simulate_flight(duration_s: float = 60.0, fs: float = 400.0, seed: int = 7) -> FlightData:
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    n = int(duration_s * fs)
    t = np.arange(n) * dt

    vterm = _vterm_profile(t)

    # ---- translational dynamics (world ENU, z up) -------------------------
    vel = np.zeros((n, 3))
    pos = np.zeros((n, 3))
    accel_world = np.zeros((n, 3))
    pos[0] = np.array([0.0, 0.0, 4000.0])  # jump run altitude, 4 km AGL
    vz = 0.0
    vx = vy = 0.0
    for i in range(1, n):
        airborne = t[i] > 3.0
        if airborne:
            drag = G * (vz / vterm[i]) ** 2
            az = -G + drag
            ax = 0.6 * np.sin(0.35 * t[i]) * (t[i] > 8.0)
            ay = 0.6 * np.cos(0.27 * t[i]) * (t[i] > 8.0)
        else:
            az = ax = ay = 0.0
        vx += ax * dt
        vy += ay * dt
        vz += az * dt
        vel[i] = [vx, vy, vz]
        pos[i] = pos[i - 1] + vel[i - 1] * dt + 0.5 * np.array([ax, ay, az]) * dt**2
        accel_world[i] = [ax, ay, az]

    # ---- attitude --------------------------------------------------------
    quat = np.zeros((n, 4))
    quat[0] = [1.0, 0.0, 0.0, 0.0]
    gyro_true = np.zeros((n, 3))
    for i in range(1, n):
        airborne = t[i] > 3.0
        # Voluntary head/body rotation: slow tracking turns.
        yaw_rate = 0.25 * np.sin(0.12 * t[i]) if airborne else 0.0
        pitch_rate = 0.10 * np.sin(0.20 * t[i]) if airborne else 0.0
        w = np.array([pitch_rate, 0.05 * np.cos(0.3 * t[i]) if airborne else 0.0, yaw_rate])
        # Buffeting: 5-25 Hz vibration, strong during freefall and deployment.
        if airborne:
            amp = np.deg2rad(35.0)
            if 38.0 < t[i] < 46.0:
                amp *= 1.6  # deployment shock
            if t[i] > 43.0:
                amp *= 0.25  # calmer under canopy
            w = w + amp * np.array(
                [
                    np.sin(2 * np.pi * 11.0 * t[i]),
                    np.sin(2 * np.pi * 7.5 * t[i] + 0.7),
                    np.sin(2 * np.pi * 9.3 * t[i] + 1.9),
                ]
            )
        gyro_true[i] = w
        quat[i] = frames.quat_normalize(frames.quat_mul(quat[i - 1], frames.quat_exp(w * dt)))

    # ---- specific force and noisy sensors --------------------------------
    accel_true = np.zeros((n, 3))
    for i in range(n):
        R = frames.quat_to_rotmat(quat[i])
        accel_true[i] = R.T @ (accel_world[i] - frames.GRAVITY)

    gyro_bias_true = np.array([3e-4, -2e-4, 1.5e-4])
    accel_bias_true = np.array([0.02, -0.015, 0.03])
    gyro_meas = (
        gyro_true
        + gyro_bias_true
        + np.deg2rad(0.08) * rng.standard_normal((n, 3))
    )
    accel_meas = accel_true + accel_bias_true + 0.03 * rng.standard_normal((n, 3))

    # ---- aiding sensors --------------------------------------------------
    truth_altitude = pos[:, 2]
    baro = truth_altitude + 0.35 * rng.standard_normal(n)

    gnss_pos = np.full((n, 3), np.nan)
    gnss_vel = np.full((n, 3), np.nan)
    gnss_stride = int(fs / 10)
    outage = (t > 42.0) & (t < 49.0)  # canopy transit: GNSS lost
    for i in range(0, n, gnss_stride):
        if not outage[i]:
            gnss_pos[i] = pos[i] + 1.5 * rng.standard_normal(3)
            gnss_vel[i] = vel[i] + 0.15 * rng.standard_normal(3)

    pitot_airspeed = np.abs(vel[:, 2]) + 0.5 * rng.standard_normal(n)
    pitot_airspeed[np.abs(vel[:, 2]) < 10.0] = np.nan  # unreadable at low speed

    visual_valid = (t > 3.0) & (t < 58.0)
    visual_stride = int(fs / 30)
    visual_valid = visual_valid & (np.arange(n) % visual_stride == 0)

    return FlightData(
        t=t,
        dt=dt,
        pos=pos,
        vel=vel,
        quat=quat,
        gyro_true=gyro_true,
        accel_true=accel_true,
        gyro_meas=gyro_meas,
        accel_meas=accel_meas,
        baro=baro,
        gnss_pos=gnss_pos,
        gnss_vel=gnss_vel,
        pitot_airspeed=pitot_airspeed,
        visual_valid=visual_valid,
        truth_altitude=truth_altitude,
    )


def inertial_altitude_channel(baro: np.ndarray, accel_z: np.ndarray, dt: float, tau: float = 2.0) -> np.ndarray:
    """A complementary baro-inertial altitude: the fourth redundant source.

    Integrates the vertical accelerometer and continuously pulls the result back
    towards the barometer with time constant ``tau`` so it does not diverge.
    """
    alt = np.zeros_like(baro)
    vel = 0.0
    alt[0] = baro[0]
    for i in range(1, len(baro)):
        # accel_z is the specific force in the body z axis; subtract gravity.
        a_up = accel_z[i] - (-G)  # -(accel + g) with z-down body convention
        vel += a_up * dt
        alt_pred = alt[i - 1] + vel * dt
        alt[i] = alt_pred + (baro[i] - alt_pred) * (dt / tau)
        vel += (baro[i] - alt[i]) * (dt / tau)
    return alt
