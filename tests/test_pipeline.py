"""End-to-end pipeline smoke test."""

import numpy as np

from iars.pipeline import run_all
from iars.telemetry.hud import HudState, render_hud


def test_pipeline_runs_and_reports_metrics():
    result = run_all(seed=7)
    summary = result.summary()
    assert set(summary) == {"redundancy", "navigation", "stabilization", "stereo"}
    assert summary["redundancy"]["rmse_m"] < 3.0
    assert summary["navigation"]["fused_rmse_m"] < summary["navigation"]["ins_only_rmse_m"]
    assert summary["stabilization"]["attenuation_db"] > 10.0
    assert summary["stereo"]["residual_px_after"] < summary["stereo"]["residual_px_before"]


def test_redundancy_detects_injected_faults():
    result = run_all(seed=7)
    counts = result.redundancy["metrics"]["isolated_counts"]
    assert counts["gnss"] > 0


def test_hud_renders_without_error():
    img = render_hud(HudState())
    assert img.size[0] > 0 and img.size[1] > 0
    arr = np.asarray(img)
    assert arr.shape[2] == 3
