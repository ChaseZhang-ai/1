"""Generate all publication figures used by the README and docs.

Run ``python tools/generate_figures.py`` from the repository root; the PNGs land
in ``docs/assets/``.  Every label is English so the figures can be published.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iars.pipeline import run_all  # noqa: E402
from iars.telemetry.hud import HudState, save_hud  # noqa: E402

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "assets")
os.makedirs(ASSETS, exist_ok=True)

GREEN = "#28c76f"
DARK = "#0b1020"
ACCENT = "#4f8cff"
RED = "#e8534f"
AMBER = "#f4b740"


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cccccc",
            "axes.grid": True,
            "grid.color": "#e8e8e8",
            "grid.linewidth": 0.8,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "figure.dpi": 130,
        }
    )


def _box(ax, xy, wh, text, fc, ec, tc="white", fs=11):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                linewidth=1.4, facecolor=fc, edgecolor=ec))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, wrap=True)


def _arrow(ax, p0, p1, color="#555555"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color=color))


def figure_architecture():
    fig, ax = plt.subplots(figsize=(12.6, 7.0))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 7.0)
    ax.axis("off")
    ax.grid(False)

    ax.text(0.1, 6.75, "Swift IARS - algorithm data flow", fontsize=15, fontweight="bold")
    ax.text(0.1, 6.45, "2024 POC prototype firmware profile (400 Hz control loop)", fontsize=10, color="#666666")

    ax.text(0.2, 5.92, "sensors", fontsize=9, color="#888888", style="italic")
    _box(ax, (0.2, 4.55), (2.1, 0.62), "BMX160 IMU\n(accel / gyro / mag)", "#eef3ff", "#7ba0e8", "#1b2a4a", 9)
    _box(ax, (0.2, 3.80), (2.1, 0.62), "MS5611 barometer", "#eef3ff", "#7ba0e8", "#1b2a4a", 9)
    _box(ax, (0.2, 3.05), (2.1, 0.62), "GNSS / BeiDou", "#eef3ff", "#7ba0e8", "#1b2a4a", 9)
    _box(ax, (0.2, 2.30), (2.1, 0.62), "Pitot air-data probe", "#eef3ff", "#7ba0e8", "#1b2a4a", 9)
    _box(ax, (0.2, 1.55), (2.1, 0.62), "Stereo waveguide cameras", "#eef3ff", "#7ba0e8", "#1b2a4a", 9)

    _box(ax, (3.1, 3.05), (2.2, 1.55), "I. Redundancy &\nfault isolation\n(consensus vote)", "#eafaf1", GREEN, "#0d3b21", 10)
    _box(ax, (5.9, 2.95), (2.3, 1.75), "IV. ESKF navigator\n15-state error-state\nKalman filter", "#fff5e6", AMBER, "#5a3d05", 10)
    _box(ax, (8.8, 2.95), (3.6, 1.75), "HUD compositor\nspeed / altitude tapes,\nhorizon, warnings", "#eef3ff", "#5b79b5", "#12203f", 10)

    _box(ax, (5.9, 5.15), (2.3, 0.85), "II. Head-motion\nstabilisation", "#fdeaea", RED, "#5a1010", 10)
    _box(ax, (5.9, 0.55), (2.3, 0.85), "III. Binocular display\nalignment", "#f0e9fd", "#8a6ad1", "#2a1550", 10)

    for y in (4.86, 4.11, 3.36, 2.61, 1.86):
        _arrow(ax, (2.3, y), (3.1, 3.82))
    _arrow(ax, (5.3, 3.82), (5.9, 3.82))
    _arrow(ax, (8.2, 3.82), (8.8, 3.82))
    _arrow(ax, (7.05, 5.15), (7.05, 4.70))
    _arrow(ax, (7.05, 1.40), (7.05, 2.95))
    _arrow(ax, (8.2, 5.57), (10.2, 4.70))
    _arrow(ax, (8.2, 0.97), (10.2, 2.95))

    ax.text(3.15, 4.72, "aiding measurements", fontsize=8.5, color="#888888", style="italic")
    ax.text(8.25, 5.30, "stabilised\nhead attitude", fontsize=8, color="#888888", style="italic")
    ax.text(8.25, 1.25, "per-eye\ncorrection", fontsize=8, color="#888888", style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_architecture.png"), bbox_inches="tight")
    plt.close(fig)


def figure_stabilization(res):
    stab = res.stabilization
    t = stab["t"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    sl = slice(400 * 20, 400 * 24)
    tt = t[sl]
    axes[0].plot(tt, stab["gyro_dps"][sl, 2], color="#bbbbbb", lw=1.0, label="Raw gyro (yaw)")
    axes[0].plot(tt, stab["voluntary_dps"][sl, 2], color=GREEN, lw=2.0, label="Voluntary (low-pass)")
    axes[0].plot(tt, stab["vibration_dps"][sl, 2], color=RED, lw=1.0, label="Vibration (high-pass)")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("angular rate [deg/s]")
    axes[0].set_title("Decomposition of the head rate")
    lo = float(np.nanmin(stab["gyro_dps"][sl]))
    hi = float(np.nanmax(stab["gyro_dps"][sl]))
    axes[0].set_ylim(lo - 8, hi + 42)
    axes[0].legend(fontsize=8, loc="upper right", framealpha=0.95, ncol=1)

    # Amplitude spectrum before/after stabilisation.
    fs = 1.0 / float(res.flight.dt)
    seg = stab["gyro_dps"][:, 0]
    disp = stab["voluntary_dps"][:, 0] * 1.0 + 0.05 * stab["vibration_dps"][:, 0]
    f = np.fft.rfftfreq(len(seg), 1 / fs)
    for sig, c, lab in [(seg, "#bbbbbb", "Raw rate"), (disp, GREEN, "Stabilised display rate")]:
        spec = np.abs(np.fft.rfft(sig - np.mean(sig))) / len(sig) * 2
        axes[1].semilogy(f, spec, color=c, lw=1.6, label=lab)
    axes[1].axvspan(4, 30, color="#ffe9e9", alpha=0.6, zorder=0)
    axes[1].set_xlim(0, 40)
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel("amplitude [deg/s]")
    axes[1].set_title("Buffeting band rejected (shaded)")
    axes[1].legend(fontsize=9)

    axes[2].axis("off")
    lines = [
        ("Buffeting attenuation", f"{stab['metrics']['attenuation_db']:.1f} dB"),
        ("Raw jitter (RMS)", f"{stab['metrics']['raw_jitter_rms_dps']:.1f} deg/s"),
        ("Stabilised jitter (RMS)", f"{stab['metrics']['stabilised_jitter_rms_dps']:.2f} deg/s"),
        ("Tracking error (RMS)", f"{stab['metrics']['tracking_rmse_dps']:.2f} deg/s"),
        ("Display latency", f"{stab['metrics']['display_latency_ms']:.0f} ms"),
        ("Split cutoff", "1.8 Hz"),
    ]
    ax = axes[2]
    for i, (k, v) in enumerate(lines):
        y = 0.9 - i * 0.15
        ax.text(0.02, y, k, fontsize=11, color="#333333")
        ax.text(0.98, y, v, fontsize=11, color=GREEN, fontweight="bold", ha="right")
    ax.set_title("Stabiliser performance")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_stabilization.png"), bbox_inches="tight")
    plt.close(fig)


def figure_redundancy(res):
    red = res.redundancy
    t = res.flight.t
    n = len(t)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.6), sharex=True)

    dec = 20
    tt = t[::dec]
    colors = {"baro": "#4f8cff", "gnss": "#e8534f", "inertial": "#f4b740", "pitot": "#9b59b6"}
    for name, seq in red["sources"].items():
        axes[0].plot(tt, seq[::dec], color=colors[name], lw=1.0, alpha=0.8, label=name)
    axes[0].plot(tt, red["truth"][::dec], color="#222222", lw=1.6, ls="--", label="truth")
    axes[0].plot(tt, red["fused"][::dec], color=GREEN, lw=2.4, label="fused")
    axes[0].axvspan(t[int(0.70 * n)], t[int(0.78 * n)], color="#ffe1e1", alpha=0.7, zorder=0)
    axes[0].annotate(
        "GNSS frozen",
        xy=(t[int(0.74 * n)], 2200),
        xytext=(t[int(0.55 * n)], 2850),
        fontsize=9,
        color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.0),
    )
    axes[0].annotate(
        "barometer bias injected",
        xy=(t[int(0.90 * n)], 500),
        xytext=(t[int(0.52 * n)], 900),
        fontsize=9,
        color=colors["baro"],
        arrowprops=dict(arrowstyle="->", color=colors["baro"], lw=1.0),
    )
    axes[0].set_ylim(-250, 4600)
    axes[0].set_ylabel("altitude [m]")
    axes[0].set_title("Redundant altitude sources, fused estimate and injected failures")
    axes[0].legend(fontsize=8, ncol=3, loc="lower left", framealpha=0.95)

    for i, (name, flags) in enumerate(red["isolated"].items()):
        axes[1].scatter(t[flags], np.full(flags.sum(), i), s=3, color=colors[name])
    axes[1].set_yticks(range(len(red["isolated"])))
    axes[1].set_yticklabels(list(red["isolated"]))
    axes[1].set_xlabel("time [s]")
    axes[1].set_title("Isolated (rejected) samples per source")
    axes[1].set_ylim(-0.5, len(red["isolated"]) - 0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_redundancy.png"), bbox_inches="tight")
    plt.close(fig)


def figure_navigation(res):
    nav = res.eskf
    t = res.flight.t
    truth = nav["truth"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    axes[0].plot(truth[:, 0], truth[:, 1], color="#222222", lw=2.0, label="truth")
    axes[0].plot(nav["ins_pos"][:, 0], nav["ins_pos"][:, 1], color=RED, lw=1.4, ls="--", label="INS only")
    axes[0].plot(nav["est_pos"][:, 0], nav["est_pos"][:, 1], color=GREEN, lw=1.8, label="ESKF fused")
    axes[0].set_xlabel("east [m]")
    axes[0].set_ylabel("north [m]")
    axes[0].set_title("Horizontal trajectory (top view)")
    axes[0].legend(fontsize=9)
    axes[0].set_aspect("equal", adjustable="datalim")

    axes[1].plot(t, nav["ins_error_norm"], color=RED, lw=1.4, ls="--", label="INS only")
    axes[1].plot(t, nav["error_norm"], color=GREEN, lw=1.8, label="ESKF fused")
    axes[1].axvspan(t[int(0.70 * len(t))], t[int(0.78 * len(t))], color="#ffe1e1", alpha=0.7, zorder=0)
    axes[1].text(t[int(0.705 * len(t))], 0.85 * np.nanmax(nav["ins_error_norm"]), "GNSS outage", fontsize=9, color=RED)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("position error [m]")
    axes[1].set_title("Position error: fused vs pure inertial")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_navigation.png"), bbox_inches="tight")
    plt.close(fig)


def figure_stereo(res):
    st = res.stereo
    ideal, left, right = st["ideal"], st["left"], st["right"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    for uv, c, lab, m in [(left, "#4f8cff", "left eye", "o"), (right, "#e8534f", "right eye", "s")]:
        axes[0].scatter(uv[:, 0], uv[:, 1], color=c, marker=m, s=42, facecolors="none", label=lab)
    axes[0].set_title("Raw calibration targets (misaligned)")
    axes[0].legend(fontsize=9)
    axes[0].set_xlabel("x [normalised]")
    axes[0].set_ylabel("y [normalised]")

    report = st["report"]
    labels = ["Vertical\ndivergence\n[arcmin]", "Relative\ntilt\n[deg]", "Convergence\nerror [m]", "Scale\nerror [%]"]
    values = [
        report.vertical_divergence_arcmin,
        report.relative_tilt_deg,
        report.convergence_m - 20.0,
        report.scale_error_pct,
    ]
    limits = [10.0, 0.5, 5.0, 1.0]
    x = np.arange(len(values))
    bars = axes[1].bar(x, values, color=[RED if abs(v) > l else GREEN for v, l in zip(values, limits)], width=0.55)
    for xi, (v, l) in zip(x, zip(values, limits)):
        axes[1].hlines(l, xi - 0.42, xi + 0.42, color="#333333", ls="--", lw=1.2)
        axes[1].hlines(-l, xi - 0.42, xi + 0.42, color="#333333", ls="--", lw=1.2)
    axes[1].set_ylim(-6.4, 6.4)
    for xi, (v, l) in zip(x, zip(values, limits)):
        axes[1].text(xi, 5.4, f"{v:.2f}", ha="center", va="center", fontsize=9, color="#222222",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#dddddd"))
        axes[1].text(xi, -5.6, f"limit {l:g}", ha="center", va="center", fontsize=7.5, color="#888888")
    axes[1].axhline(0, color="#888888", lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8.5)
    axes[1].set_title("Alignment error vs comfort limits (dashed)")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_stereo.png"), bbox_inches="tight")
    plt.close(fig)


def figure_intent():
    from iars.sensors.intent_dataset import make_episodes
    from iars.stabilization.intent import IntentClassifier, extract_features, heuristic_intent_gate

    fs = 400.0
    # Must match the held-out split used by tools/train_intent.py so the numbers
    # in the figure and in the README are the same measurement.
    episodes = make_episodes(60, seed=10_000)
    clf = IntentClassifier.load_default()

    X, y = [], []
    for ep in episodes:
        f, c = extract_features(ep.gyro_dps, fs)
        X.append(f)
        y.append(ep.label[c])
    X, y = np.vstack(X), np.concatenate(y)

    h_pred, h_true = [], []
    for ep in episodes:
        _, c = extract_features(ep.gyro_dps, fs)
        h_pred.append(heuristic_intent_gate(ep.gyro_dps, fs)[c])
        h_true.append(ep.label[c])
    h_pred, h_true = np.concatenate(h_pred), np.concatenate(h_true)

    def metrics(pred, truth):
        tp = np.sum(pred & truth)
        fp = np.sum(pred & ~truth)
        fn = np.sum(~pred & truth)
        return np.array(
            [
                np.mean(pred == truth),
                tp / max(tp + fp, 1),
                tp / max(tp + fn, 1),
            ]
        )

    learned = metrics(clf.predict(X), y)
    heuristic = metrics(h_pred, h_true)

    # One episode where both a turn and pure buffeting occur, for the strips.
    ep = next(e for e in episodes if e.kind == "mixed")
    t = np.arange(len(ep.gyro_dps)) / fs
    gate_learned = clf.predict_stream(ep.gyro_dps, fs)[1]
    gate_heur = heuristic_intent_gate(ep.gyro_dps, fs)
    centres = clf.predict_stream(ep.gyro_dps, fs)[0]
    gate_learned_ps = np.interp(np.arange(len(t)), centres, gate_learned.astype(float)) > 0.5

    fig = plt.figure(figsize=(13.8, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 0.55],
                          hspace=0.42, wspace=0.22)
    ax_sig = fig.add_subplot(gs[0, 0])
    ax_gate = fig.add_subplot(gs[1, 0], sharex=ax_sig)
    ax_bar = fig.add_subplot(gs[:, 1])

    ax_sig.fill_between(t, 0, 100, where=ep.label, color="#c8f2d8", alpha=0.8, step="mid",
                        label="ground-truth turn")
    ax_sig.plot(t, np.linalg.norm(ep.gyro_dps, axis=1), color="#9aa0a6", lw=0.9, label="measured |rate|")
    ax_sig.plot(t, np.linalg.norm(ep.voluntary_dps, axis=1), color="#137a3a", lw=2.0,
                label="true voluntary |rate|")
    ax_sig.axhline(8.0, color="#e8534f", ls="--", lw=1.2, label="8 deg/s threshold")
    ax_sig.set_ylim(0, 100)
    ax_sig.set_ylabel("angular rate [deg/s]")
    ax_sig.set_title("One episode: deliberate turn vs buffeting")
    ax_sig.legend(fontsize=8, loc="upper right", ncol=2, framealpha=1.0)

    from matplotlib.colors import ListedColormap

    strip = np.vstack([ep.label.astype(float), gate_heur.astype(float), gate_learned_ps.astype(float)])
    ax_gate.imshow(strip, aspect="auto", interpolation="nearest",
                   cmap=ListedColormap(["#f2f2f2", "#28c76f"]),
                   extent=[0, t[-1], 0, 3])
    ax_gate.set_yticks([0.5, 1.5, 2.5])
    ax_gate.set_yticklabels(["truth", "heuristic", "learned"], fontsize=8.5)
    ax_gate.set_xlabel("time [s]")
    ax_gate.set_title("Gate decisions (green = follow the head)", fontsize=10)

    labels_bar = ["accuracy", "precision", "recall"]
    xpos = np.arange(3)
    ax_bar.bar(xpos - 0.19, heuristic, width=0.36, color="#b9c2cc", label="heuristic (2024)")
    ax_bar.bar(xpos + 0.19, learned, width=0.36, color=GREEN, label="learned classifier")
    for xi, (h, l) in enumerate(zip(heuristic, learned)):
        ax_bar.text(xi - 0.19, h + 0.02, f"{h:.2f}", ha="center", fontsize=8.5, color="#555555")
        ax_bar.text(xi + 0.19, l + 0.02, f"{l:.2f}", ha="center", fontsize=8.5, color="#0d3b21",
                    fontweight="bold")
    ax_bar.set_xticks(xpos)
    ax_bar.set_xticklabels(labels_bar, fontsize=9.5)
    ax_bar.set_ylim(0, 1.42)
    ax_bar.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_bar.set_ylabel("score")
    ax_bar.set_title(f"Intent detection on held-out episodes ({len(episodes)})")
    ax_bar.legend(fontsize=8.5, loc="upper center", ncol=2, framealpha=1.0)
    ax_bar.grid(axis="y", color="#e8e8e8")

    fig.savefig(os.path.join(ASSETS, "fig_intent.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    _style()
    res = run_all(seed=7)
    figure_architecture()
    figure_stabilization(res)
    figure_redundancy(res)
    figure_navigation(res)
    figure_stereo(res)
    figure_intent()
    save_hud(os.path.join(ASSETS, "hud_render.png"), HudState())
    print("figures written to", ASSETS)
    for name in sorted(os.listdir(ASSETS)):
        if name.startswith(("fig_", "hud_")) and name.endswith(".png"):
            print(" -", name)


if __name__ == "__main__":
    main()
