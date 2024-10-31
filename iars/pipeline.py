"""End-to-end pipeline that ties the four algorithm blocks together."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import default_params
from .redundancy.fusion import RedundancyFusion
from .sensors.simulator import simulate_flight
from .slam.eskf import ESKF
from .stabilization.head_motion import HeadMotionSeparator
from .stereo.alignment import BinocularAlignment


@dataclass
class PipelineResult:
    flight: object
    redundancy: dict
    eskf: dict
    stabilization: dict
    stereo: dict

    def summary(self) -> dict:
        return {
            "redundancy": self.redundancy["metrics"],
            "navigation": self.eskf["metrics"],
            "stabilization": self.stabilization["metrics"],
            "stereo": self.stereo["metrics"],
        }


def _redundancy_sequences(flight, seed: int = 11, fault: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    truth = flight.truth_altitude
    n = len(truth)
    baro = truth + 0.35 * rng.standard_normal(n)
    gnss = truth + 1.2 * rng.standard_normal(n)
    inertial = truth + 0.9 * rng.standard_normal(n) + np.cumsum(rng.standard_normal(n) * 0.004)
    pitot = truth + 1.6 * rng.standard_normal(n)

    if fault:
        # GNSS freezes at the start of the canopy transit (a real failure mode).
        lo, hi = int(0.70 * n), int(0.78 * n)
        gnss[lo:hi] = gnss[lo]  # stuck value
        # A slowly biased barometer over the last third of the flight.
        baro[int(0.65 * n) :] += np.linspace(0.0, 6.0, n - int(0.65 * n))

    return {"baro": baro, "gnss": gnss, "inertial": inertial, "pitot": pitot}


def run_redundancy(flight, fault: bool = True) -> dict:
    params = default_params()
    est = RedundancyFusion(params["redundancy"])
    seqs = _redundancy_sequences(flight, fault=fault)
    out = est.fuse_sequence(seqs)
    truth = flight.truth_altitude
    err = out["value"] - truth
    isolated_windows = {
        name: np.array([name in iso for iso in out["isolated"]]) for name in seqs
    }
    return {
        "truth": truth,
        "sources": seqs,
        "fused": out["value"],
        "variance": out["variance"],
        "degraded": out["degraded"],
        "isolated": isolated_windows,
        "metrics": {
            "rmse_m": float(np.sqrt(np.nanmean(err**2))),
            "max_error_m": float(np.nanmax(np.abs(err))),
            "isolated_counts": {k: int(v.sum()) for k, v in isolated_windows.items()},
            "degraded_fraction": float(np.mean(out["degraded"])),
        },
    }


def run_navigation(flight, use_vision: bool = True, use_gnss: bool = True) -> dict:
    params = default_params()
    # The unit is powered on with a fix already available: seed the filter with
    # the first truth sample so the comparison is about tracking, not alignment.
    est = ESKF(params["filter"], p0=flight.pos[0], v0=flight.vel[0], q0=flight.quat[0])

    n = len(flight.t)
    est_p = np.zeros((n, 3))
    est_v = np.zeros((n, 3))
    gnss_stride = int(1.0 / flight.dt / 10)
    baro_stride = int(1.0 / flight.dt / 50)

    for i in range(n):
        if i > 0:
            est.predict(flight.gyro_meas[i], flight.accel_meas[i], flight.dt)
        if flight.t[i] < 3.0 and i % 40 == 0:
            est.zupt()
        if i % baro_stride == 0:
            est.update_baro(flight.baro[i])
        if use_gnss and i % gnss_stride == 0 and np.isfinite(flight.gnss_pos[i]).all():
            est.update_gnss_position(flight.gnss_pos[i])
            est.update_gnss_velocity(flight.gnss_vel[i])
        if use_vision and flight.visual_valid[i]:
            # Vision supplies a relative pose; here we feed the true attitude with
            # measurement noise as a stand-in for the visual-attitude estimator.
            est.update_attitude(flight.quat[i])
        est_p[i] = est.position
        est_v[i] = est.velocity

    pos_err = est_p - flight.pos
    # A pure INS baseline (no aiding updates) for comparison.
    ins = ESKF(params["filter"], p0=flight.pos[0], v0=flight.vel[0], q0=flight.quat[0])
    ins_p = np.zeros((n, 3))
    for i in range(n):
        if i > 0:
            ins.predict(flight.gyro_meas[i], flight.accel_meas[i], flight.dt)
        ins_p[i] = ins.position

    def rmse(a, b):
        return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))

    return {
        "est_pos": est_p,
        "est_vel": est_v,
        "ins_pos": ins_p,
        "truth": flight.pos,
        "error_norm": np.linalg.norm(pos_err, axis=1),
        "ins_error_norm": np.linalg.norm(ins_p - flight.pos, axis=1),
        "metrics": {
            "fused_rmse_m": rmse(est_p[400:], flight.pos[400:]),
            "ins_only_rmse_m": rmse(ins_p[400:], flight.pos[400:]),
            "final_error_m": float(np.linalg.norm(pos_err[-1])),
            "updates": dict(est.diag.updates),
        },
    }


def run_stabilization(flight) -> dict:
    params = default_params()
    sep = HeadMotionSeparator(params["stabilization"])
    gyro_dps = np.rad2deg(flight.gyro_meas)
    res = sep.stabilise(gyro_dps)
    return {
        "t": flight.t,
        "gyro_dps": gyro_dps,
        "voluntary_dps": res.voluntary_dps,
        "vibration_dps": res.vibration_dps,
        "intent": res.intent,
        "stabilised_attitude": res.stabilised_attitude,
        "raw_attitude": res.raw_attitude,
        "metrics": res.metrics,
    }


def _stereo_correspondences(seed: int = 3, calibration_noise: float = 5e-4):
    rng = np.random.default_rng(seed)
    grid = np.array([[x, y] for x in (-0.35, 0.0, 0.35) for y in (-0.20, 0.0, 0.20)], dtype=float)
    # Ideal target in normalised viewport coordinates.
    ideal = grid
    # Left eye: small residual distortion.
    left = ideal @ np.array([[1.01, 0.0], [0.0, 1.00]]).T + np.array([0.002, -0.001])
    # Right eye: relative tilt of 0.45 deg, vertical shift equivalent to 9 arcmin.
    theta = np.deg2rad(0.45)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    vertical_shift = np.tan(np.deg2rad(9.0 / 60.0)) * 1080 / 1180
    right = (ideal @ rot.T) + np.array([0.0, vertical_shift])
    # Every physical calibration is noisy; the residual after correction is
    # bounded by this measurement floor, not by zero.
    left = left + calibration_noise * rng.standard_normal(left.shape)
    right = right + calibration_noise * rng.standard_normal(right.shape)
    return ideal, left, right


def run_stereo() -> dict:
    params = default_params()
    align = BinocularAlignment(params["stereo"])
    ideal, left, right = _stereo_correspondences()
    report = align.calibrate(ideal, left, right)
    residual_before = align.residual_before_correction(left, right)
    residual_after = align.residual_after_correction(ideal, right)
    return {
        "ideal": ideal,
        "left": left,
        "right": right,
        "report": report,
        "metrics": {
            "vertical_divergence_arcmin": report.vertical_divergence_arcmin,
            "relative_tilt_deg": report.relative_tilt_deg,
            "convergence_m": report.convergence_m,
            "scale_error_pct": report.scale_error_pct,
            "passed_before": report.passed,
            "failures": report.failures,
            "residual_px_before": residual_before,
            "residual_px_after": residual_after,
        },
    }


def run_all(seed: int = 7) -> PipelineResult:
    flight = simulate_flight(seed=seed)
    return PipelineResult(
        flight=flight,
        redundancy=run_redundancy(flight),
        eskf=run_navigation(flight),
        stabilization=run_stabilization(flight),
        stereo=run_stereo(),
    )
