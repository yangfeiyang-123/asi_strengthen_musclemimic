"""Calibrate diagonal policy exploration in decoded physical-action space.

The policy samples a raw action perturbation ``delta_z`` with diagonal
covariance and a deterministic decoder maps it to the physical excitation
space.  Around the decoder operating point, with Jacobian ``J``,

``E[||J delta_z||^2] = trace(J diag(std**2) J.T)``.

This module chooses the raw-action standard deviations so that the expected
root-mean-square physical perturbation has an explicit target.  Keeping this
calibration outside PPO also makes the fixed-synergy experiment directly
comparable with the full-muscle baseline.
"""

from __future__ import annotations

import numpy as np

SUPPORTED_STD_MODES: tuple[str, ...] = (
    "scalar_calibrated",
    "per_dimension",
    "gram_whitened",
)


def physical_exploration_rms(
    decoder_jacobian: np.ndarray,
    std_vector: np.ndarray,
) -> float:
    """Return the linearized expected physical RMS for a diagonal Gaussian.

    Args:
        decoder_jacobian: Matrix with shape ``[physical_dim, action_dim]``.
        std_vector: Positive raw-action standard deviations with shape
            ``[action_dim]``.

    Returns:
        ``sqrt(E[||J delta_z||^2] / physical_dim)``.
    """

    jacobian = _validated_jacobian(decoder_jacobian)
    std = np.asarray(std_vector, dtype=np.float64)
    if std.shape != (jacobian.shape[1],):
        raise ValueError(
            "std_vector shape must match decoder Jacobian action dimension: "
            f"expected {(jacobian.shape[1],)}, got {std.shape}"
        )
    if not np.all(np.isfinite(std)) or np.any(std <= 0.0):
        raise ValueError("std_vector must contain only finite positive values")

    column_energy = np.sum(np.square(jacobian), axis=0)
    mean_square = float(np.dot(column_energy, np.square(std))) / jacobian.shape[0]
    return float(np.sqrt(max(mean_square, 0.0)))


def calibrate_exploration_std(
    decoder_jacobian: np.ndarray,
    target_physical_rms: float,
    *,
    mode: str = "scalar_calibrated",
    residual_dim: int = 0,
    residual_std_scale: float = 1.0,
    gram_epsilon: float = 1e-6,
    min_std: float = 1e-4,
    max_std: float = 10.0,
) -> np.ndarray:
    """Calibrate per-action initial std from a decoder Jacobian.

    The modes differ only in their *relative* raw-action std profile.  A final
    positive scalar is solved for so that the expected physical excitation RMS
    equals ``target_physical_rms``:

    - ``scalar_calibrated`` starts with equal std in every action dimension.
    - ``per_dimension`` scales dimension ``i`` inversely with the norm of
      decoder-Jacobian column ``i``, equalizing each dimension's first-order
      physical contribution.
    - ``gram_whitened`` uses the diagonal approximation
      ``sqrt(diag((J.T @ J + epsilon I)^-1))``.

    If ``residual_dim`` is non-zero, the final ``residual_dim`` columns are the
    structured-residual action.  Their relative profile is multiplied by
    ``residual_std_scale`` *before* the global physical-RMS calibration.  Thus
    a value below one suppresses residual exploration without changing the
    requested aggregate RMS, unless a std bound becomes active.  Bounds are
    enforced while solving the global scale, rather than clipping afterward.

    Args:
        decoder_jacobian: Matrix with shape ``[physical_dim, action_dim]``.
        target_physical_rms: Desired linearized expected RMS in physical space.
        mode: One of :data:`SUPPORTED_STD_MODES`.
        residual_dim: Number of structured-residual dimensions at the end of
            the action vector.
        residual_std_scale: Positive relative scale for residual dimensions.
        gram_epsilon: Positive Tikhonov regularizer for inverse-norm modes.
        min_std: Inclusive lower bound for every returned std.
        max_std: Inclusive upper bound for every returned std.

    Returns:
        A finite, positive ``float64`` vector of length ``action_dim``.

    Raises:
        ValueError: If inputs are invalid, the decoder has no local influence,
            or the target RMS cannot be attained within the requested bounds.
    """

    jacobian = _validated_jacobian(decoder_jacobian)
    physical_dim, action_dim = jacobian.shape

    if mode not in SUPPORTED_STD_MODES:
        raise ValueError(f"unsupported exploration std mode {mode!r}; expected one of {SUPPORTED_STD_MODES}")
    target = _positive_finite_scalar(target_physical_rms, "target_physical_rms")
    epsilon = _positive_finite_scalar(gram_epsilon, "gram_epsilon")
    lower = _positive_finite_scalar(min_std, "min_std")
    upper = _positive_finite_scalar(max_std, "max_std")
    residual_scale = _positive_finite_scalar(residual_std_scale, "residual_std_scale")
    if lower > upper:
        raise ValueError(f"min_std must be <= max_std, got {lower} > {upper}")
    if isinstance(residual_dim, bool) or not isinstance(residual_dim, int | np.integer):
        raise ValueError("residual_dim must be an integer")
    residual_dim = int(residual_dim)
    if residual_dim < 0 or residual_dim > action_dim:
        raise ValueError(f"residual_dim must be in [0, {action_dim}], got {residual_dim}")

    gram = jacobian.T @ jacobian
    column_energy = np.diag(gram).copy()
    total_energy = float(np.sum(column_energy))
    if not np.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError("decoder_jacobian must have non-zero finite physical influence")

    if mode == "scalar_calibrated":
        relative_std = np.ones(action_dim, dtype=np.float64)
    elif mode == "per_dimension":
        relative_std = np.reciprocal(np.sqrt(column_energy + epsilon))
    else:
        regularized_gram = gram + epsilon * np.eye(action_dim, dtype=np.float64)
        try:
            inverse_gram = np.linalg.inv(regularized_gram)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - epsilon normally prevents this
            raise ValueError("regularized decoder Gram matrix is not invertible") from exc
        inverse_diagonal = np.diag(inverse_gram)
        # Numerical roundoff can make a theoretically positive diagonal a tiny
        # negative number.  Anything materially non-positive is invalid.
        tolerance = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(inverse_diagonal)))) * 32.0
        if np.any(inverse_diagonal < -tolerance):
            raise ValueError("regularized decoder Gram inverse has a negative diagonal")
        relative_std = np.sqrt(np.maximum(inverse_diagonal, tolerance))

    if residual_dim:
        relative_std[-residual_dim:] *= residual_scale
    if not np.all(np.isfinite(relative_std)) or np.any(relative_std <= 0.0):
        raise ValueError("calibrated relative std profile is not finite and positive")

    target_energy = target**2 * physical_dim
    min_energy = float(np.dot(column_energy, np.full(action_dim, lower**2)))
    max_energy = float(np.dot(column_energy, np.full(action_dim, upper**2)))
    energy_tolerance = max(1e-14, target_energy * 1e-10)
    if target_energy < min_energy - energy_tolerance or target_energy > max_energy + energy_tolerance:
        min_rms = float(np.sqrt(min_energy / physical_dim))
        max_rms = float(np.sqrt(max_energy / physical_dim))
        raise ValueError(
            "target_physical_rms is unattainable within std bounds: "
            f"target={target:.12g}, attainable=[{min_rms:.12g}, {max_rms:.12g}]"
        )

    std = _bounded_energy_normalize(
        relative_std,
        column_energy,
        target_energy,
        lower,
        upper,
    )
    if not np.all(np.isfinite(std)) or np.any(std <= 0.0):
        raise RuntimeError("exploration std calibration produced invalid values")
    if np.any(std < lower) or np.any(std > upper):
        raise RuntimeError("exploration std calibration violated requested bounds")

    achieved_rms = physical_exploration_rms(jacobian, std)
    if not np.isclose(achieved_rms, target, rtol=1e-8, atol=1e-10):
        raise RuntimeError(
            "exploration std calibration missed physical RMS target: "
            f"target={target:.12g}, achieved={achieved_rms:.12g}"
        )
    return std


def _bounded_energy_normalize(
    relative_std: np.ndarray,
    column_energy: np.ndarray,
    target_energy: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Scale a positive profile to a target weighted energy under box bounds."""

    def scaled(multiplier: float) -> np.ndarray:
        return np.clip(multiplier * relative_std, lower, upper)

    def energy(multiplier: float) -> float:
        values = scaled(multiplier)
        return float(np.dot(column_energy, np.square(values)))

    # The unconstrained answer avoids bisection and preserves ratios exactly in
    # the common case where no bound is active.
    profile_energy = float(np.dot(column_energy, np.square(relative_std)))
    multiplier = float(np.sqrt(target_energy / profile_energy))
    candidate = scaled(multiplier)
    if np.isclose(
        float(np.dot(column_energy, np.square(candidate))),
        target_energy,
        rtol=1e-12,
        atol=max(1e-15, target_energy * 1e-12),
    ):
        return candidate

    # Find a bracket.  At zero all dimensions sit at the lower bound; as the
    # multiplier grows they monotonically approach the upper bound.
    lo = 0.0
    hi = max(multiplier, upper / float(np.min(relative_std)), 1.0)
    while energy(hi) < target_energy:
        hi *= 2.0
        if not np.isfinite(hi):  # pragma: no cover - guarded by feasibility check
            raise RuntimeError("failed to bracket bounded std calibration")

    for _ in range(128):
        mid = (lo + hi) / 2.0
        if energy(mid) < target_energy:
            lo = mid
        else:
            hi = mid
    return scaled((lo + hi) / 2.0)


def _validated_jacobian(value: np.ndarray) -> np.ndarray:
    jacobian = np.asarray(value, dtype=np.float64)
    if jacobian.ndim != 2 or jacobian.shape[0] == 0 or jacobian.shape[1] == 0:
        raise ValueError("decoder_jacobian must be a non-empty rank-2 matrix with shape [physical_dim, action_dim]")
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("decoder_jacobian must contain only finite values")
    return jacobian


def _positive_finite_scalar(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return result


__all__ = [
    "SUPPORTED_STD_MODES",
    "calibrate_exploration_std",
    "physical_exploration_rms",
]
