"""Algorithm III - Binocular (dual-waveguide) display alignment.

IARS projects two independent virtual images, one per eye, through a pair of
geometric waveguides.  Mechanical mounting tolerance, waveguide wedge error and
per-eye optical distortion all introduce a small misalignment between the two
images.  The eye is sensitive to three of these parameters in particular:

* **vertical divergence** - an image-height mismatch (arc-minutes),
* **relative image tilt** - a roll difference between the two images (degrees),
* **convergence error** - the implied fusion distance drifting away from the
  intended optical infinity / near plane (metres).

This module recovers those parameters from a pair of calibration targets and
synthesises a per-eye corrective warp that drives the residual back under the
comfort limits (ANSI/HFES 100 style thresholds, see :class:`StereoConfig`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import StereoConfig


@dataclass
class EyeWarp:
    """Affine mapping from ideal normalised coords to one eye's image plane."""

    A: np.ndarray  # 2x2 linear part
    t: np.ndarray  # 2 offset

    def apply(self, uv: np.ndarray) -> np.ndarray:
        return (self.A @ np.atleast_2d(uv).T).T + self.t

    def inverse(self) -> "EyeWarp":
        A_inv = np.linalg.inv(self.A)
        return EyeWarp(A_inv, -A_inv @ self.t)


@dataclass
class AlignmentReport:
    vertical_divergence_arcmin: float
    relative_tilt_deg: float
    convergence_m: float
    scale_error_pct: float
    horizontal_divergence_deg: float
    passed: bool
    failures: list


class BinocularAlignment:
    def __init__(self, config: StereoConfig | None = None):
        self.cfg = config or StereoConfig()
        self.left: EyeWarp | None = None
        self.right: EyeWarp | None = None

    # ------------------------------------------------------------------ calibration
    @staticmethod
    def _fit_affine(ideal_uv: np.ndarray, measured_uv: np.ndarray) -> EyeWarp:
        """Least-squares affine fit measured = A @ ideal + t."""
        uv = np.atleast_2d(ideal_uv).T if np.atleast_2d(ideal_uv).shape[0] == 2 else np.atleast_2d(ideal_uv)
        mv = np.atleast_2d(measured_uv).T if np.atleast_2d(measured_uv).shape[0] == 2 else np.atleast_2d(measured_uv)
        if uv.shape[1] != mv.shape[1]:
            raise ValueError("needs at least 3 correspondences")
        X = np.column_stack([uv, np.ones(len(uv))])  # (N, 3)
        # Solve for all three output rows at once via normal equations.
        coef, *_ = np.linalg.lstsq(X, mv, rcond=None)  # (3, 2)
        A = coef[:2].T
        t = coef[2]
        return EyeWarp(A, t)

    def calibrate(self, ideal_uv: np.ndarray, left_uv: np.ndarray, right_uv: np.ndarray) -> AlignmentReport:
        self.left = self._fit_affine(ideal_uv, left_uv)
        self.right = self._fit_affine(ideal_uv, right_uv)
        return self.evaluate()

    # ------------------------------------------------------------------ analysis
    def relative_warp(self) -> tuple:
        """Return (D, d) mapping the left image into the right image plane."""
        if self.left is None or self.right is None:
            raise RuntimeError("calibrate() first")
        D = self.right.A @ np.linalg.inv(self.left.A)
        d = self.right.t - D @ self.left.t
        return D, d

    def evaluate(self) -> AlignmentReport:
        D, d = self.relative_warp()

        # Vertical divergence sampled at image centre, converted to arc-minutes.
        # d is in normalised image units (image width == 1); one normalised unit
        # subtends 2*atan(0.5 * panel_width / focal) -> use focal in px directly.
        vertical_px = d[1] * self.cfg.image_size_px[1]
        vertical_arcmin = np.rad2deg(np.arctan(vertical_px / self.cfg.focal_px)) * 60.0

        # Relative image tilt from the rotation part of D.
        tilt_deg = np.degrees(np.arctan2(D[1, 0], D[0, 0]))

        # Scale (aniseikonia) error between the two images.
        scale_l = np.sqrt(abs(np.linalg.det(self.left.A)))
        scale_r = np.sqrt(abs(np.linalg.det(self.right.A)))
        scale_error_pct = 100.0 * (scale_r - scale_l) / scale_l

        # Horizontal divergence -> implied convergence distance.  Zero
        # divergence means the two eye images are parallel (fused at optical
        # infinity); a small inward vergence angle theta fuses at ipd / theta.
        horizontal_px = d[0] * self.cfg.image_size_px[0]
        horizontal_div_rad = np.arctan(horizontal_px / self.cfg.focal_px)
        horizontal_div_deg = np.degrees(horizontal_div_rad)
        if abs(horizontal_div_rad) < 1e-6:
            depth_m = 1000.0
        else:
            depth_m = float(np.clip(self.cfg.ipd_m / (2.0 * np.tan(abs(horizontal_div_rad) / 2.0)), 0.2, 1000.0))

        failures = []
        if abs(vertical_arcmin) > self.cfg.max_vertical_divergence_arcmin:
            failures.append("vertical_divergence")
        if abs(tilt_deg) > self.cfg.max_relative_tilt_deg:
            failures.append("relative_tilt")
        if abs(scale_error_pct) > self.cfg.max_scale_error_pct:
            failures.append("scale_error")
        if abs(horizontal_div_deg) > self.cfg.max_horizontal_divergence_deg:
            failures.append("horizontal_divergence")

        return AlignmentReport(
            vertical_divergence_arcmin=float(vertical_arcmin),
            relative_tilt_deg=float(tilt_deg),
            convergence_m=depth_m,
            scale_error_pct=float(scale_error_pct),
            horizontal_divergence_deg=float(horizontal_div_deg),
            passed=len(failures) == 0,
            failures=failures,
        )

    # ------------------------------------------------------------------ correction
    def correction(self) -> tuple:
        """Per-eye warps that null the relative misalignment.

        The right eye is taken as the reference; the left eye's warp is adjusted
        so that the two image planes coincide.  Returns ``(warp_left, warp_right)``
        suitable for baking into the compositor's per-eye transform.
        """
        D, d = self.relative_warp()
        # Warp applied in right-eye coordinates that maps left-eye content onto it.
        right_fix = EyeWarp(D, d)
        left_fix = EyeWarp(np.eye(2), np.zeros(2))
        # Express the fix as an additional transform on top of the existing warps.
        corrected_left = EyeWarp(right_fix.A @ self.left.A, right_fix.A @ self.left.t + right_fix.t)
        return corrected_left, self.right

    def residual_before_correction(self, left_uv: np.ndarray, right_uv: np.ndarray) -> float:
        """RMS pixel disparity between the raw left and right eye images."""
        l = np.atleast_2d(left_uv)
        r = np.atleast_2d(right_uv)
        err_px = (l - r) * np.asarray(self.cfg.image_size_px)
        return float(np.sqrt(np.mean(np.sum(err_px**2, axis=1))))

    def residual_after_correction(self, ideal_uv: np.ndarray, right_uv: np.ndarray) -> float:
        """RMS pixel disparity between the two eyes after applying the correction.

        The corrected left-eye warp is evaluated at the ideal target positions and
        compared with the *measured* right-eye points, so the result reflects the
        real calibration noise floor rather than the algebraic identity that the
        fitted warps would otherwise satisfy.
        """
        corrected_left, _ = self.correction()
        ref = np.atleast_2d(ideal_uv)
        r = np.atleast_2d(right_uv)
        err_px = (corrected_left.apply(ref) - r) * np.asarray(self.cfg.image_size_px)
        return float(np.sqrt(np.mean(np.sum(err_px**2, axis=1))))
