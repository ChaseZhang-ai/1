# Algorithm notes

Design rationale and derivations for the four blocks in `iars/`. Units are SI throughout; the
world frame is a local ENU frame (x east, y north, z up, **gravity along −z**). Quaternions are
`[w, x, y, z]`, body→world.

---

## I. Redundancy design and fault-tolerant fusion

**Sources.** Four independent height estimates: barometric (`σ ≈ 0.35 m`), GNSS ellipsoidal
(`σ ≈ 1.2 m`), baro-aided inertial (`σ ≈ 0.9 m`), pitot-derived (`σ ≈ 1.6 m`). The baro-aided
inertial channel is a complementary filter — integrate the vertical specific force, pull the
result back toward the barometer with time constant `τ`:

```
v̂ ← v̂ + a_up dt
ĥ ← ĥ + v̂ dt + (h_baro − ĥ) · dt/τ
v̂ ← v̂ + (h_baro − ĥ) · dt/τ
```

This is the "keep the good high-frequency of the accelerometer, the good low-frequency of the
barometer" trick, and it is what makes the inertial channel a usable *redundant* source rather
than a divergent integrator.

**Consensus clustering.** Rather than a single median/MAD detector (which fails when two
channels fail in opposite directions), sources are grouped by pairwise compatibility:
`i ~ j  ⇔  |xᵢ − xⱼ| ≤ k·√(σᵢ² + σⱼ²)`, `k = 4.5`. The largest mutually-compatible group — the
largest clique, found greedily — is the consensus; members off it are isolated. This is the
classic "signal plausibility / 2-out-of-n monitor" of flight-critical redundancy, made explicit.

**Fusion.** With bias-corrected measurements `zᵢ = xᵢ − b̂ᵢ` and inverse-variance weights
`wᵢ = 1/(σᵢ² + b̂ᵢ²)`:

```
x̂ = Σ wᵢ zᵢ / Σ wᵢ ,      Var(x̂) = 1 / Σ wᵢ
```

A sequential chi-square gate then drops any survivor whose normalised residual against `x̂`
exceeds `γ²` (1 dof, 99.9 % → 10.83), and the fit is repeated. Bias trackers are EWMA-updated
toward the consensus, so a slowly drifting instrument is progressively down-weighted.

**Degraded mode.** When fewer than two sources form a consensus the output is flagged degraded
and the HUD falls back to the last fused value plus a computed-margin altitude warning.

---

## II. Head-motion stabilisation

**Signal model.** The measured head rate decomposes as
`ω_meas = ω_voluntary + ω_vibration + n`. In the POC data the voluntary band sits below ≈2 Hz
(deliberate tracking turns) and the buffeting band spans ≈4–30 Hz (torso/airflow coupling).

**Decomposition.** Complementary biquad pair — RBJ low-pass at `f_c = 1.8 Hz` for the voluntary
term, and the complementary high-pass (`ω − ω_lp`) for vibration. A forward–backward pass
(`filtfilt`) removes the phase lag that would otherwise corrupt the metrics and add display
latency. A band-pass (geometric-centre `√(4·30)`, `Q = f₀/Δf`) shapes the vibration term used
for the jitter metrics.

**Intent detector.** Hysteresis on the low-pass magnitude: the follow gate opens when
`|ω_lp| > 8 deg/s` has held for `0.15 s`, and latches for the same window. When the gate is shut
the frame is *frozen* instead of integrating noise — a deliberate UX choice, since an inertially
frozen horizon reads as steadier than one that follows a shaking head.

**Display regeneration.**

```
ω_display = 1{intent} · ω_voluntary + g_vib · ω_vibration ,   g_vib = 0.05
q_display ← q_display ⊗ Exp(ω_display dt)
```

`g_vib = 0.05` corresponds to −26 dB; the measured −25.7 dB is close because the small residual
comes from the band edges. Display latency is `1/(2π f_c) ≈ 88 ms` at `f_c = 1.8 Hz`, which is
below the ≈100 ms threshold where a lagging horizon starts to feel wrong.

---

## III. Binocular display alignment

**Model.** Each eye's virtual image is an affine map of the ideal normalised viewport:
`m_e = A_e u + t_e`, fitted by least squares from ≥3 (in practice 9+) calibration
correspondences. The relative map from left to right eye is `D = A_R A_L⁻¹`, `d = t_R − D t_L`.

**Sensitivities.**

| Parameter | Definition | Comfort limit |
|---|---|---|
| Vertical divergence | `atan(d_y · H / f)`, arc-minutes | 10 arcmin |
| Relative image tilt | `atan2(D₁₀, D₀₀)`, degrees | 0.5° |
| Convergence | `IPD / (2 tan(θ/2))`, metres | — (reported) |
| Scale (aniseikonia) | `100·(√det A_R − √det A_L)/√det A_L` | 1 % |

`f = 1180 px` is the panel focal length in pixels, `H = 1080` the panel height, `IPD = 64 mm`.

**Correction.** The compositor bakes a per-eye affine `D·A_L, D·t_L + d` into its warp LUT so the
two image planes coincide; with real (noisy) calibration the residual is bounded by the
measurement floor, not by zero. In the demo that is **8.30 px → 0.80 px** while the pre-correction
parameters still trip the vertical-divergence limit, so the block both *detects* and *fixes*.

---

## IV. Multi-sensor SLAM / INS fusion (error-state Kalman filter)

**Why error-state.** The nominal state `X = [p, v, q, b_a, b_g]` is propagated non-linearly, but
the *error* `δX = [δp, δv, δθ, δb_a, δb_g]` stays small and linear, so a linear KF on the error
is valid through 50 m/s and high angular rate where an EKF on the full quaternion drifts out of
its linearisation. After each update the error is injected into the nominal state and reset to
zero.

**Propagation** (`a_b = a_meas − b_a`, `R = R(q)`):

```
a_world = R a_b + g
p ← p + v dt + ½ a_world dt²
v ← v + a_world dt
q ← q ⊗ Exp((ω_meas − b_g) dt)
```

Error transition `F` and noise map `G`:

```
F = [ I   I dt   −½R[a_b]×dt²  −½R dt²   0
      0   I      −R[a_b]× dt     −R dt    0
      0   0       I − [ω]×dt       0     −I dt
      0   0          0           I        0
      0   0          0           0        I ]
G = diag-blocks { ½R dt² , R dt , −I dt , I dt , I dt } ,  Q = G diag(σ_a², σ_g², σ_ba², σ_bg²) Gᵀ
```

**Updates.** All are linear in the error state:

| Measurement | Residual | H block |
|---|---|---|
| GNSS position | `p_gnss − p̂` | `[I 0 0 0 0]` |
| GNSS velocity | `v_gnss − v̂` | `[0 I 0 0 0]` |
| Barometric altitude | `h_baro − p̂_z` | `[0 0 1 0 0]` |
| Stereo attitude | `Log(q̂⁻¹ ⊗ q_vis)` | `[0 0 I 0 0]` (attitude) |
| Zero-velocity (ZUPT) | `−v̂` | `[0 I 0 0 0]` |

Each update is gated: if `rᵀ (H P Hᵀ + R)⁻¹ r > γ²` the measurement is rejected — this is what
rejects the GNSS re-acquisition jump after the canopy outage rather than letting it yank the
state. Covariance is updated in Joseph form for numerical stability.

**Result.** Over the 60 s synthetic profile, fused RMSE is **0.25 m** (0.13 m final) versus
**31 m** for the same IMU with no aiding.

---

## Reproducing

```bash
python -m pytest -q            # 25 tests
python tools/run_demo.py       # prints the metric table
python tools/generate_figures.py
```

Every number quoted in the README is emitted by `tools/run_demo.py` on the synthetic profile
(`seed = 7`).
