"""Algorithm III - binocular display alignment tests."""

import numpy as np

from iars.config import StereoConfig
from iars.stereo.alignment import BinocularAlignment


def _grid():
    return np.array([[x, y] for x in (-0.35, 0.0, 0.35) for y in (-0.20, 0.0, 0.20)], dtype=float)


def _rot(deg):
    th = np.deg2rad(deg)
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def test_affine_fit_recovers_known_transform():
    ideal = _grid()
    A = np.array([[1.02, 0.01], [-0.01, 0.99]])
    t = np.array([0.003, -0.002])
    measured = ideal @ A.T + t
    warp = BinocularAlignment._fit_affine(ideal, measured)
    assert np.allclose(warp.A, A, atol=1e-6)
    assert np.allclose(warp.t, t, atol=1e-6)


def test_detects_vertical_divergence():
    cfg = StereoConfig()
    align = BinocularAlignment(cfg)
    ideal = _grid()
    left = ideal
    shift = np.tan(np.deg2rad(20.0 / 60.0)) * cfg.image_size_px[1] / cfg.focal_px
    right = ideal + np.array([0.0, shift])
    report = align.calibrate(ideal, left, right)
    assert report.vertical_divergence_arcmin > cfg.max_vertical_divergence_arcmin
    assert "vertical_divergence" in report.failures
    assert not report.passed


def test_detects_relative_tilt():
    cfg = StereoConfig()
    align = BinocularAlignment(cfg)
    ideal = _grid()
    report = align.calibrate(ideal, ideal, ideal @ _rot(0.8).T)
    assert abs(report.relative_tilt_deg) > cfg.max_relative_tilt_deg
    assert "relative_tilt" in report.failures


def test_within_tolerance_passes():
    cfg = StereoConfig()
    align = BinocularAlignment(cfg)
    ideal = _grid()
    right = ideal + np.array([0.0, 0.0005])
    report = align.calibrate(ideal, ideal, right)
    assert report.passed
    assert report.failures == []


def test_correction_reduces_residual_error():
    cfg = StereoConfig()
    align = BinocularAlignment(cfg)
    ideal = _grid()
    theta = np.deg2rad(0.45)
    shift = np.tan(np.deg2rad(9.0 / 60.0)) * cfg.image_size_px[1] / cfg.focal_px
    right = (ideal @ _rot(0.45).T) + np.array([0.0, shift])
    left = ideal * 1.01
    align.calibrate(ideal, left, right)
    before = np.sqrt(np.mean(np.sum(((left - right) * np.array(cfg.image_size_px)) ** 2, axis=1)))
    after = align.residual_after_correction(ideal, right)
    assert after < before * 0.5
