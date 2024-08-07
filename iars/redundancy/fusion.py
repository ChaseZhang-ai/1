"""Algorithm I - Redundancy design and fault-tolerant sensor fusion.

The POC flight computer carries four independent ways of knowing how high the
wearer is:

* ``baro``     - MS5611 static-pressure altitude,
* ``gnss``     - GPS/BeiDou ellipsoidal altitude,
* ``inertial`` - the double integral of the vertical accelerometer channel
                 anchored by the barometric reference,
* ``pitot``    - dynamic pressure integrated through the air-data model.

None of them is trustworthy on its own: the barometer lags during a fast
deployment, GNSS freezes or drops under canopy transit, the inertial channel
diverges as 0.5*a*t^2, and the pitot probe is unreadable below ~10 m/s.

This module fuses them with a two-stage scheme:

1. *Robust gross-error rejection.*  A consensus-clustering detector groups the
   sources that agree to within their combined uncertainty and isolates the
   rest, so a single stuck or saturated channel is dropped without knowing
   which one it is - and even two channels failing in opposite directions are
   handled.
2. *Bias-aware sequential least squares.*  Surviving samples are combined with
   inverse-variance weights while an EWMA tracks the slowly varying per-source
   bias, so a mildly biased instrument is down-weighted instead of being thrown
   away.

The estimator reports the fused value, its variance, the set of isolated
sources and a ``degraded`` flag that is asserted when fewer than two sources
agree - the HUD then falls back to the last fused value plus a computed-margin
altitude warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import RedundancyConfig


@dataclass
class FusionResult:
    value: float
    variance: float
    healthy: list
    isolated: list
    degraded: bool
    residuals: dict = field(default_factory=dict)


class RedundancyFusion:
    """Fault-tolerant scalar fusion for a redundant measurement group."""

    def __init__(self, config: RedundancyConfig | None = None):
        self.cfg = config or RedundancyConfig()
        self._bias = {k: 0.0 for k in self.cfg.source_sigma}
        self._healthy_streak = {k: 0 for k in self.cfg.source_sigma}

    # ------------------------------------------------------------------ helpers
    def _sigma(self, name: str) -> float:
        return float(self.cfg.source_sigma.get(name, 1.0))

    def _weight(self, name: str, bias_penalty: bool = True) -> float:
        sigma = self._sigma(name)
        if bias_penalty:
            sigma = np.hypot(sigma, self._bias[name])
        return 1.0 / (sigma * sigma)

    # ------------------------------------------------------------------ stage 1
    def _isolate(self, samples: dict) -> tuple[list, list, dict]:
        """Consensus clustering over the current measurement set.

        Two sources are *compatible* when they agree to within ``mad_gate`` times
        the quadrature sum of their 1-sigma accuracies.  The largest mutually
        compatible group is taken as the consensus; everything else is reported
        as isolated.  Unlike a plain median/MAD detector this still works when
        two channels fail in opposite directions.
        """
        names = sorted(samples)
        if len(samples) <= 2:
            return names, [], {n: 0.0 for n in names}

        n = len(names)
        adjacency = {a: set() for a in names}
        for i in range(n):
            for j in range(i + 1, n):
                a, b = names[i], names[j]
                tol = self.cfg.mad_gate * np.hypot(self._sigma(a), self._sigma(b))
                if abs(samples[a] - samples[b]) <= tol:
                    adjacency[a].add(b)
                    adjacency[b].add(a)

        # Greedy largest-clique-ish search: repeatedly grow from the source that
        # has the largest compatible neighbourhood.
        best = []
        for seed in sorted(adjacency, key=lambda k: -len(adjacency[k])):
            cluster = [seed]
            for cand in sorted(adjacency[seed], key=lambda k: -len(adjacency[k])):
                if all(cand in adjacency[m] for m in cluster):
                    cluster.append(cand)
            if len(cluster) > len(best) or (
                len(cluster) == len(best)
                and self._cluster_variance(cluster) < self._cluster_variance(best)
            ):
                best = cluster

        isolated = [a for a in names if a not in best]
        residuals = {a: float(samples[a] - np.mean([samples[b] for b in best])) for a in names}
        return sorted(best), isolated, residuals

    def _cluster_variance(self, cluster: list) -> float:
        if not cluster:
            return np.inf
        return float(sum(self._sigma(a) ** 2 for a in cluster))

    # ------------------------------------------------------------------ stage 2
    def update(self, samples: dict) -> FusionResult:
        """Fuse one measurement set.  ``samples`` maps source name -> value."""
        samples = {k: float(v) for k, v in samples.items() if np.isfinite(v)}
        if not samples:
            return FusionResult(np.nan, np.inf, [], [], degraded=True)

        healthy, isolated, residuals = self._isolate(samples)

        # Bias-corrected measurements and inverse-variance weights.
        corrected, weights = [], []
        for n in healthy:
            corrected.append(samples[n] - self._bias[n])
            weights.append(self._weight(n))
        corrected = np.array(corrected)
        weights = np.array(weights)

        fused = float(np.sum(weights * corrected) / np.sum(weights))
        variance = float(1.0 / np.sum(weights))

        # Sequential chi-square gate: drop any healthy source whose normalised
        # residual against the fused estimate exceeds the gate, then refit.
        for n in list(healthy):
            r = (samples[n] - self._bias[n] - fused) / np.sqrt(self._sigma(n) ** 2 + variance)
            if r * r > self.cfg.chi2_gate:
                idx = healthy.index(n)
                healthy.remove(n)
                isolated.append(n)
                weights = np.delete(weights, idx)
                corrected = np.delete(corrected, idx)
                if len(healthy) >= 1:
                    fused = float(np.sum(weights * corrected) / np.sum(weights))
                    variance = float(1.0 / np.sum(weights))

        # Update per-source bias trackers towards the current consensus.
        a = self.cfg.bias_alpha
        for n, v in samples.items():
            if n in healthy:
                resid = v - fused
                self._bias[n] = (1 - a) * self._bias[n] + a * resid
                self._healthy_streak[n] = self._healthy_streak.get(n, 0) + 1
            else:
                # Decay the streak so a recovered channel is not barred forever.
                self._healthy_streak[n] = max(0, self._healthy_streak.get(n, 0) - 1)

        return FusionResult(
            value=fused,
            variance=variance,
            healthy=healthy,
            isolated=isolated,
            degraded=len(healthy) < self.cfg.min_healthy,
            residuals=residuals,
        )

    def fuse_sequence(self, sequences: dict) -> dict:
        """Convenience wrapper running the estimator over aligned time series."""
        n = len(next(iter(sequences.values())))
        out = {
            "value": np.full(n, np.nan),
            "variance": np.full(n, np.nan),
            "degraded": np.zeros(n, dtype=bool),
            "isolated": [[] for _ in range(n)],
        }
        for i in range(n):
            res = self.update({k: v[i] for k, v in sequences.items()})
            out["value"][i] = res.value
            out["variance"][i] = res.variance
            out["degraded"][i] = res.degraded
            out["isolated"][i] = res.isolated
        return out
