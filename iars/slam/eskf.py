"""Algorithm IV - Multi-sensor SLAM / INS fusion (error-state Kalman filter).

This is the navigator that gives the HUD a single, coherent 6-DoF pose.  It
fuses four measurement families at very different rates:

* an inertial strapdown (IMU pre-integration) at 400 Hz,
* pressure altitude at ~50 Hz,
* GNSS/BeiDou position and velocity at ~10 Hz,
* a stereo visual pose (from the waveguide cameras / a body-worn stereo pair)
  at ~30 Hz, which is the only module that is drift-free indoors and under
  canopy where GNSS is gone.

The filter is a 15-state error-state Kalman filter (ESKF) on ``[p, v, q, ba,
bg]``.  Working in the error state keeps the attitude error small and therefore
linear, so the covariance stays valid through a 50 m/s freefall and violent
buffeting where a naive EKF on the full quaternion would diverge.  Every
measurement passes a chi-square innovation gate first, which is what makes the
fusion robust to the GNSS outages and visual dropouts recorded in the POC data.

Reference: Sola, "Quaternion kinematics for the error-state Kalman filter".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import FilterConfig
from . import frames


@dataclass
class FilterDiagnostics:
    updates: dict = field(default_factory=dict)

    def bump(self, key):
        self.updates[key] = self.updates.get(key, 0) + 1


class ESKF:
    def __init__(self, config: FilterConfig | None = None, p0=None, v0=None, q0=None):
        self.cfg = config or FilterConfig()
        self.p = np.zeros(3) if p0 is None else np.asarray(p0, float).copy()
        self.v = np.zeros(3) if v0 is None else np.asarray(v0, float).copy()
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) if q0 is None else frames.quat_normalize(np.asarray(q0, float))
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)

        self.P = np.eye(15)
        self.P[0:3, 0:3] *= 4.0
        self.P[3:6, 3:6] *= 0.25
        self.P[6:9, 6:9] *= np.deg2rad(3.0) ** 2
        self.P[9:12, 9:12] *= 0.1**2
        self.P[12:15, 12:15] *= np.deg2rad(0.5) ** 2

        self._initialised = False
        self.diag = FilterDiagnostics()

    # ------------------------------------------------------------------ accessors
    @property
    def position(self) -> np.ndarray:
        return self.p.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.v.copy()

    @property
    def attitude(self) -> np.ndarray:
        return self.q.copy()

    # ------------------------------------------------------------------ propagate
    def predict(self, gyro_rps: np.ndarray, accel_mps2: np.ndarray, dt: float) -> None:
        R = frames.quat_to_rotmat(self.q)
        a_body = np.asarray(accel_mps2, float) - self.ba
        a_world = R @ a_body + frames.GRAVITY

        self.q = frames.quat_normalize(
            frames.quat_mul(self.q, frames.quat_exp((np.asarray(gyro_rps, float) - self.bg) * dt))
        )
        # Trapezoidal position update: use the pre-update velocity for the
        # linear term and the acceleration for the quadratic term.
        self.p = self.p + self.v * dt + 0.5 * a_world * dt**2
        self.v = self.v + a_world * dt

        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = -0.5 * R @ frames.skew(a_body) * dt**2
        F[0:3, 9:12] = -0.5 * R * dt**2
        F[3:6, 6:9] = -R @ frames.skew(a_body) * dt
        F[3:6, 9:12] = -R * dt
        w = (np.asarray(gyro_rps, float) - self.bg) * dt
        F[6:9, 6:9] = np.eye(3) - frames.skew(w)
        F[6:9, 12:15] = -np.eye(3) * dt

        G = np.zeros((15, 12))
        G[0:3, 0:3] = 0.5 * R * dt**2
        G[3:6, 0:3] = R * dt
        G[6:9, 3:6] = -np.eye(3) * dt
        G[9:12, 6:9] = np.eye(3) * dt
        G[12:15, 9:12] = np.eye(3) * dt

        sa = self.cfg.accel_noise_mps2
        sg = np.deg2rad(self.cfg.gyro_noise_dps)
        sba = self.cfg.accel_bias_rw
        sbg = np.deg2rad(self.cfg.gyro_bias_rw)
        Qc = np.diag([sa**2] * 3 + [sg**2] * 3 + [sba**2] * 3 + [sbg**2] * 3)
        Q = G @ Qc @ G.T

        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------ update
    def _update(self, H: np.ndarray, residual: np.ndarray, R: np.ndarray, gate: float, tag: str) -> bool:
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        # Chi-square innovation gate: reject a measurement that is inconsistent
        # with the current covariance (outage re-acquisition, vision dropout).
        # The very first measurement bypasses the gate - it initialises the state.
        if self._initialised and float(residual @ S_inv @ residual) > gate:
            return False
        K = self.P @ H.T @ S_inv
        dx = K @ residual
        self._inject(dx)
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form
        self._initialised = True
        self.diag.bump(tag)
        return True

    def _inject(self, dx: np.ndarray) -> None:
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.q = frames.quat_normalize(frames.quat_mul(self.q, frames.quat_exp(dx[6:9])))
        self.ba = self.ba + dx[9:12]
        self.bg = self.bg + dx[12:15]

    # ------------------------------------------------------------------ sensors
    def update_gnss_position(self, pos_m: np.ndarray, sigma: float | None = None) -> bool:
        R = np.eye(3) * (sigma or self.cfg.gnss_pos_sigma_m) ** 2
        H = np.zeros((3, 15))
        H[:, 0:3] = np.eye(3)
        return self._update(H, np.asarray(pos_m, float) - self.p, R, self.cfg.chi2_gate_3d, "gnss_pos")

    def update_gnss_velocity(self, vel_mps: np.ndarray, sigma: float | None = None) -> bool:
        R = np.eye(3) * (sigma or self.cfg.gnss_vel_sigma_mps) ** 2
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        return self._update(H, np.asarray(vel_mps, float) - self.v, R, self.cfg.chi2_gate_3d, "gnss_vel")

    def update_baro(self, altitude_m: float, sigma: float | None = None) -> bool:
        R = np.array([[((sigma or self.cfg.baro_sigma_m)) ** 2]])
        H = np.zeros((1, 15))
        H[0, 2] = 1.0  # z up
        return self._update(H, np.array([altitude_m - self.p[2]]), R, self.cfg.chi2_gate_1d, "baro")

    def update_attitude(self, q_meas: np.ndarray, sigma: float | None = None) -> bool:
        R = np.eye(3) * (sigma or self.cfg.visual_att_sigma_rad) ** 2
        H = np.zeros((3, 15))
        H[:, 6:9] = np.eye(3)
        residual = frames.quat_log(frames.quat_mul(_qconj(self.q), np.asarray(q_meas, float)))
        return self._update(H, residual, R, self.cfg.chi2_gate_3d, "attitude")

    def update_visual_pose(self, pos_m: np.ndarray, q_meas: np.ndarray) -> bool:
        ok_p = self.update_gnss_position(pos_m, self.cfg.visual_pos_sigma_m)
        ok_q = self.update_attitude(q_meas, self.cfg.visual_att_sigma_rad)
        return ok_p or ok_q

    def zupt(self) -> bool:
        """Zero-velocity update: pins the INS while the wearer stands still."""
        R = np.eye(3) * self.cfg.zupt_sigma_mps**2
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        return self._update(H, -self.v, R, self.cfg.chi2_gate_3d, "zupt")

    # ------------------------------------------------------------------ diagnostics
    def state_dict(self) -> dict:
        return {
            "position": self.position,
            "velocity": self.velocity,
            "attitude": self.attitude,
            "accel_bias": self.ba.copy(),
            "gyro_bias": self.bg.copy(),
            "cov_trace": float(np.trace(self.P)),
            "updates": dict(self.diag.updates),
        }


def _qconj(q: np.ndarray) -> np.ndarray:
    q = frames.quat_normalize(q)
    return np.array([q[0], -q[1], -q[2], -q[3]])
