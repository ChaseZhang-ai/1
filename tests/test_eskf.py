"""Algorithm IV - ESKF navigation tests."""

import numpy as np

from iars.config import FilterConfig
from iars.pipeline import run_navigation
from iars.sensors.simulator import simulate_flight
from iars.slam import frames
from iars.slam.eskf import ESKF


def test_quaternion_roundtrip():
    q = frames.quat_normalize(np.array([0.7, 0.2, -0.3, 0.55]))
    assert np.allclose(frames.rotmat_to_quat(frames.quat_to_rotmat(q)), q, atol=1e-9)


def test_exp_log_roundtrip():
    phi = np.array([0.01, -0.02, 0.03])
    assert np.allclose(frames.quat_log(frames.quat_exp(phi)), phi, atol=1e-9)


def test_stationary_filter_does_not_drift_with_zupt():
    cfg = FilterConfig()
    eskf = ESKF(cfg)
    dt = 1.0 / 400.0
    rng = np.random.default_rng(0)
    for _ in range(400 * 20):
        gyro = 0.001 * rng.standard_normal(3)
        accel = np.array([0.0, 0.0, 9.80665]) + 0.02 * rng.standard_normal(3)
        eskf.predict(gyro, accel, dt)
        eskf.zupt()
    assert np.linalg.norm(eskf.position) < 0.5
    assert np.linalg.norm(eskf.velocity) < 0.1


def test_fusion_beats_inertial_only():
    flight = simulate_flight(duration_s=40.0, seed=5)
    fused = run_navigation(flight, use_vision=False, use_gnss=True)
    assert fused["metrics"]["fused_rmse_m"] < fused["metrics"]["ins_only_rmse_m"]


def test_chi_square_gate_rejects_outlier():
    cfg = FilterConfig()
    eskf = ESKF(cfg)
    for _ in range(2000):
        eskf.predict(np.zeros(3), np.array([0.0, 0.0, 9.80665]), 1.0 / 400.0)
    ok = eskf.update_gnss_position(np.array([0.0, 0.0, 0.0]))
    assert ok
    rejected = eskf.update_gnss_position(np.array([500.0, 500.0, 500.0]))
    assert rejected is False


def test_barometer_update_corrects_altitude():
    cfg = FilterConfig()
    eskf = ESKF(cfg, p0=[0.0, 0.0, 100.0])
    for _ in range(400):
        eskf.predict(np.zeros(3), np.array([0.0, 0.0, 9.80665]), 1.0 / 400.0)
    for _ in range(5):
        eskf.update_baro(120.0)
        eskf.predict(np.zeros(3), np.array([0.0, 0.0, 9.80665]), 1.0 / 400.0)
    assert abs(eskf.position[2] - 120.0) < 5.0
