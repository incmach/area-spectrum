"""
per_coordinate.py

Per-coordinate gradient descent on the scaled area-spectrum objective

    F(I) = sum_k ( (area_spectrum(I)[k] - target[k]) / sqrt(N[k]) )^2

where N[k] is the exact integer squared row-norm of the Jacobian of
area_spectrum at the current image I.

Each iteration:
  1. Compute grad = dF/dI (via area_spectrum_scaled_gradient).
  2. Pick the coordinate with the largest |grad| value
     (first one on ties, in row-major order).
  3. Move that one coordinate by 1 unit in the descent direction
     (-sign of that gradient component).

Only one coordinate changes per step; all others stay put.
"""

import math
import numpy as np

from definition import area_spectrum
from definition import (
    area_spectrum_scaled_gradient,
    area_spectrum_jacobian_row_squared_norms,
)


def scaled_objective(I, target):
    """
    F(I) = sum_k ((A_k(I) - target_k) / sqrt(N_k(I)))^2,
    with N_k as exact integer squared row norms.
    """
    A = area_spectrum(I)
    N = area_spectrum_jacobian_row_squared_norms(I)
    total = 0.0
    for a, t, n in zip(A, target, N):
        if n != 0:
            r = (a - t) / math.sqrt(n)
            total += r * r
    return total


def per_coordinate_descent(
    I,
    target,
    max_iter=1000,
    lower_bound=None,
    upper_bound=None,
    check_decrease=True,
    record=False,
):
    """
    Parameters
    ----------
    I : array-like of ints
        Initial image (any shape).
    target : sequence
        Target area spectrum. Must have the same length as area_spectrum(I).
    max_iter : int
        Maximum number of single-coordinate steps.
    lower_bound : int or None
        Forbid a coordinate dropping below this value (None = unbounded).
    upper_bound : int or None
        Forbid a coordinate rising above this value (None = unbounded).
    check_decrease : bool
        If True, reject a step that does not strictly decrease F and stop.
        If False, accept the step unconditionally and keep going until
        max_iter or a zero gradient is reached.
    record : bool
        If True, return (I, history) where history is a list of dicts.

    Returns
    -------
    I : np.ndarray of int
        Final image.
    history : list of dicts (only when record=True)
        Each entry: {'iter', 'F', 'coord', 'old', 'new', 'grad'}.
    """
    I = np.array(I, dtype=int)
    shape = I.shape

    history = []
    F = scaled_objective(I, target)

    for it in range(max_iter):
        # 1. Gradient of F w.r.t. I (uses exact-integer row norms internally).
        grad = np.asarray(area_spectrum_scaled_gradient(I, target), dtype=float)

        # 2. First largest by absolute value.
        flat = int(np.argmax(np.abs(grad)))
        coord = np.unravel_index(flat, shape)
        g = float(grad[coord])

        # 3. Descent direction: -sign(g).  If g == 0, we are at a
        #    coordinate-wise stationary point.
        if g == 0.0:
            break
        step = -1 if g > 0 else 1

        old_val = int(I[coord])
        new_val = old_val + step

        # Respect bounds (if any) by stopping when we would violate them.
        if lower_bound is not None and new_val < lower_bound:
            break
        if upper_bound is not None and new_val > upper_bound:
            break

        I[coord] = new_val

        if check_decrease:
            new_F = scaled_objective(I, target)
            if new_F >= F:
                # Step did not help; revert and stop.
                I[coord] = old_val
                break
        else:
            new_F = F  # not evaluated; keep previous value only for history

        if record:
            history.append({
                'iter': it,
                'F': float(F),
                'coord': tuple(int(c) for c in coord),
                'old': old_val,
                'new': new_val,
                'grad': g,
            })

        F = new_F

    if record:
        return I, history
    return I