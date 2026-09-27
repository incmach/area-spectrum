from collections import namedtuple
from functools import cache
import itertools
import math

import numpy as np
import sympy

MAX_DIMENSION = 4


@cache
def volume_function(d):
    vs = [[sympy.Symbol(f"v_{i}_{j}") for j in range(d)] for i in range(d + 1)]
    m = sympy.Matrix([[v_j - v_0_j for v_0_j, v_j in zip(vs[0], v)] for v in vs[1:]])
    return sympy.lambdify(tuple(itertools.chain(vs)), abs(m.det()))


def volume(vs):
    f = volume_function(len(vs) - 1)
    return f(*vs)


@cache
def dimension_multiplier(d):
    """Upper bound on the determinant of a d x d 0/1 matrix.

    A parallelepiped whose vertices are grid points has
        |det| <= prod(n_i - 1) * D_d
    with D_d the largest determinant of a d x d 0/1 matrix. Hadamard's
    inequality bounds it by (d+1)^((d+1)/2) / 2^d, which is exact at d=3 and
    d=7 and loose in between. Since prod(n_i - 1) < I.size, a bin count of
    I.size * D_d is always sufficient. Exact values for small d are 1, 1, 2.

    Past MAX_DIMENSION the bound is no longer offered, since the spectrum would
    be padded far beyond anything a real image needs.
    """
    if d > MAX_DIMENSION:
        raise ValueError(f"dimension {d} exceeds supported maximum {MAX_DIMENSION}")
    if d <= 2:
        return 1
    return math.ceil((d + 1) ** ((d + 1) / 2) / 2 ** d)


def spectrum_length(I):
    return I.size * dimension_multiplier(len(I.shape))


def coordinates(I):
    return list(itertools.product(*(range(n) for n in I.shape)))


def area_spectrum(I):
    """Exact area spectrum: R[k] = sum over ordered (d+1)-tuples S of prod I[S].

    The points of S are required to be pairwise distinct. For k >= 1 that is
    automatic, because a nonzero determinant makes the tuple affinely
    independent and affine independence implies distinctness. The requirement
    therefore only ever removes tuples from bin 0, namely the collinear ones
    naming some point twice.

    The consequence is that every bin is trilinear in the pixels: each term is
    a product of d+1 *distinct* pixel values, so no term carries a square or a
    cube, and a sum of such terms is a multilinear form. In particular the
    spectrum is exactly linear in any single pixel, so the derivative in
    jacobian_transpose is exact rather than merely first-order.
    """
    d = len(I.shape)
    coords = coordinates(I)
    n_spectrum = spectrum_length(I)
    result = [0] * n_spectrum
    for S in itertools.permutations(coords, d + 1):
        k = volume(S)
        result[k] += math.prod(int(I[v]) for v in S)
    return result


def jacobian_transpose(I, w):
    """Exact J^T w for the area-spectrum Jacobian: result[p] = sum_k J[k][p] w[k].

    The Jacobian is never materialized. An ordered (d+1)-tuple S contributes
    the product of the values it omits to row k = volume(S), once for each
    position i at which S carries the pixel S[i], so the prefix/suffix products
    yield all d+1 of those contributions together.

    S ranges over tuples of distinct points, matching area_spectrum. Because
    those points are distinct, the values they omit are d+1 distinct pixel
    values and the contribution is linear in any single one of them; the
    spectrum being trilinear, this is the exact derivative and not just a
    first-order one.

    The result is a flat list ordered like I.ravel(). Arithmetic stays exact
    whenever w is integral.
    """
    d = len(I.shape)
    coords = coordinates(I)
    index = {v: i for i, v in enumerate(coords)}
    result = [0] * I.size
    for S in itertools.permutations(coords, d + 1):
        wk = w[volume(S)]
        if not wk:
            continue
        vals = [int(I[v]) for v in S]
        m = len(vals)
        prefix = [1] * (m + 1)
        for i in range(m):
            prefix[i + 1] = prefix[i] * vals[i]
        suffix = [1] * (m + 1)
        for i in range(m - 1, -1, -1):
            suffix[i] = suffix[i + 1] * vals[i]
        for i in range(m):
            result[index[S[i]]] += wk * prefix[i] * suffix[i + 1]
    return result


def jacobian_row_sums(I):
    """Exact L1 row sums R[k] = sum_p ( d area_spectrum(I)[k] / d I[p] ).

    Crude scaling factor: all row entries are nonnegative, so this is the
    L1 norm of row k of the Jacobian. Used to divide out the huge dynamic
    range of the raw spectrum.

    This is J applied to the all-ones vector: it sums within each bin and is
    indexed by bin, so it returns n_spectrum entries. jacobian_transpose is
    the other orientation, summing across bins and indexed by pixel, and the
    two are not interchangeable despite both involving an all-ones weighting.
    """
    d = len(I.shape)
    coords = coordinates(I)
    n_spectrum = spectrum_length(I)
    result = [0] * n_spectrum
    for S in itertools.permutations(coords, d + 1):
        k = volume(S)
        vals = [int(I[v]) for v in S]
        m = len(vals)
        prefix = [1] * (m + 1)
        for i in range(m):
            prefix[i + 1] = prefix[i] * vals[i]
        suffix = [1] * (m + 1)
        for i in range(m - 1, -1, -1):
            suffix[i] = suffix[i + 1] * vals[i]
        for i in range(m):
            result[k] += prefix[i] * suffix[i + 1]
    return result


def normalized_spectrum_gradient(I, delta):
    """Steepest-descent gradient of a normalized-spectrum target.

    With scale = jacobian_row_sums(I) the target is

        target[k] = area_spectrum(I)[k] + delta[k] * scale[k]

    and the loss is the sum of squared normalized residuals, over the bins the
    image can actually reach:

        L(J) = sum_k (( area_spectrum(J)[k] - target[k] ) / scale[k])**2

    At the reference image the normalized residual is exactly -delta, so the
    target spectrum itself drops out of the derivative:

        dL/dJ[p] = sum_k J[k][p] * 2*(area_spectrum(I)[k] - target[k])/scale[k]**2
                 = -2 * sum_k J[k][p] * delta[k] / scale[k]

    delta is a bulk displacement, not a per-pixel one. Moving one pixel by 1
    changes bin k by J[k][p] <= scale[k], short by a factor of about I.size,
    and J[k][p] varies across pixels within a row. Dividing by scale[k] makes
    the bins commensurate with each other, which is what the sum over pixels
    needs; it is not a predictor of any single pixel's effect.

    Bins with scale[k] == 0 are unreachable, so a request against one is a
    no-op rather than a division by zero. A zero image therefore has an
    identically zero gradient.

    delta may be a full-length sequence, a mapping {bin: value}, or an
    iterable of (bin, value) pairs; the sparse forms spare a single-bin edit
    the allocation of a full vector.

    Returns a flat list ordered like I.ravel().
    """
    n_spectrum = spectrum_length(I)
    if hasattr(delta, "items"):
        pairs = list(delta.items())
    else:
        delta = list(delta)
        if delta and isinstance(delta[0], tuple):
            pairs = delta
        elif len(delta) == n_spectrum:
            pairs = [(k, v) for k, v in enumerate(delta) if v]
        else:
            pairs = delta
    scale = jacobian_row_sums(I)
    w = [0.0] * n_spectrum
    for k, value in pairs:
        if not 0 <= k < n_spectrum:
            raise IndexError(f"bin {k} outside spectrum of length {n_spectrum}")
        if scale[k] > 0:
            w[k] = -2.0 * value / scale[k]
    return jacobian_transpose(I, w)


def spectrum_residual_weights(spectrum, target, scale):
    """Bin weights w[k] = 2 * (spectrum[k] - target[k]) / scale[k]**2.

    The common factor of the gradient of the squared normalized residual. Kept
    separate so the loss and the gradient cannot drift apart in how they treat
    unreachable bins, and so descent can reuse one spectrum for both.

    Bins with scale[k] == 0 are unreachable, so their weight is 0 rather than a
    division by zero.
    """
    return [
        2.0 * (a - t) / (s * s) if s > 0 else 0.0
        for a, t, s in zip(spectrum, target, scale)
    ]

def spectrum_gradient(I, target, scale):
    """Gradient of the normalized-residual loss, as a flat list like I.ravel().

    The bin weights are the derivative of the squared normalized residual, so
    the pixel gradient is J^T w, with w from spectrum_residual_weights. Same
    enumeration as the spectrum itself, weighted instead of summed, which is
    what makes this the adjoint of the forward pass rather than a separate
    computation.

    This is an exact quantity, not a first-order one. area_spectrum sums over
    tuples of distinct points, so no term names a pixel twice and the spectrum
    is multilinear; it is therefore linear in each pixel separately, and the
    change a unit step at p makes to bin k is exactly J[k][p]. Ranking by this
    gradient orders steps by their true effect on the residual, though the loss
    is quadratic in the spectrum, so the best step by loss is not always the
    best by gradient. Confirm a step against the loss before committing to it.
    """
    w = spectrum_residual_weights(area_spectrum(I), target, scale)
    return jacobian_transpose(I, w)

def spectrum_loss(I, target, scale):
    """Sum of squared normalized residuals, over the bins the image can reach.

    L(I) = sum_{scale[k] > 0} ((area_spectrum(I)[k] - target[k]) / scale[k])**2
    """
    return sum(
        ((a - t) / s) ** 2
        for a, t, s in zip(area_spectrum(I), target, scale)
        if s > 0
    )


def ranked_gradient_steps(I, target, scale, min_value=0, max_value=255):
    """Every feasible (pixel, step) the gradient would allow, best first.

    For a step s in {-1, +1} at pixel p the loss changes by s * dL/dI[p], so
    the ordering key is -s * gradient[p]. Returned best first, which is what
    integer_gradient_direction takes the head of.

    Pixels pinned at a bound in the direction they want are omitted rather than
    clamped, since stepping the other way raises the loss.
    """
    gradient = spectrum_gradient(I, target, scale)
    candidates = []
    for p, g in zip(coordinates(I), gradient):
        if g == 0:
            continue
        value = int(I[p])
        for step in (-1, 1):
            if min_value <= value + step <= max_value:
                candidates.append((-step * g, p, step))
    candidates.sort(key=lambda c: -c[0])
    return [(p, step) for _, p, step in candidates]


def integer_gradient_direction(I, target, scale, min_value=0, max_value=255):
    """The single pixel and +-1 step that the gradient most wants to take.

    The head of ranked_gradient_steps, or None when no pixel has room to move in
    a direction the gradient wants. This ranks by the exact change the step
    makes to the spectrum, but the loss is quadratic in the spectrum, so the
    best step by loss is not always the best by gradient. Descent confirms a
    step against the real spectrum before keeping it.
    """
    candidates = ranked_gradient_steps(I, target, scale, min_value, max_value)
    return candidates[0] if candidates else None


Descent = namedtuple("Descent", "image loss accepted_steps history stopped_early")


def descent(I, target, scale=None, max_steps=1000, patience=25,
            min_value=0, max_value=255, width=1):
    """Reduce the discrepancy between the image's spectrum and a fixed target,
    moving one pixel at a time and keeping only moves that actually help.

    Each iteration takes the width best-ranked steps from ranked_gradient_steps
    and keeps the one that most reduces the true integer-spectrum loss. The
    spectrum is trilinear in the pixels, so the gradient gives a step's effect
    on the spectrum exactly, and its top-ranked step is the single best move
    available: measured on a 4x4, the head pick and the best available step
    both decrease the loss by 0.1535. The repeated-point spectrum this replaced
    was not so lucky -- its top pick gave 0.061 where 0.153 was available.

    Checking width candidates and keeping the best is therefore insurance
    against a mis-ranked step rather than a correction for nonlinearity, and
    width=1 follows the gradient exactly. Raising width costs that many spectrum
    evaluations per accepted step and typically reaches a lower loss in fewer
    steps; a width covering every feasible step makes each move exactly the best
    available one.

    Because the winner is chosen by measured loss, the returned history is
    non-increasing by construction.

    Two stopping conditions, per T0.1-06: max_steps, and patience consecutive
    iterations in which no candidate improved. The second is a local-minimum
    test, and it is reported in stopped_early so the caller can widen the search
    or raise patience rather than assume optimality. A target can be attainable
    and still unreachable this way; a larger edit is a longer path than a unit
    step can always cover.

    scale defaults to jacobian_row_sums(I) at the starting image and is held
    fixed for the whole run, which is what keeps the target's meaning stable
    across steps. Returns a Descent with the final image, its loss, the number
    of steps accepted, the loss history, and whether patience ran out.
    """
    if scale is None:
        scale = jacobian_row_sums(I)
    current = np.array(I, dtype=np.int64, copy=True)
    loss = spectrum_loss(current, target, scale)
    history = [loss]
    accepted = 0
    stalled = 0
    for _ in range(max_steps):
        candidates = ranked_gradient_steps(current, target, scale, min_value, max_value)
        best = None
        for pixel, step in candidates[:width]:
            trial = current.copy()
            trial[pixel] += step
            trial_loss = spectrum_loss(trial, target, scale)
            if trial_loss < loss and (best is None or trial_loss < best[0]):
                best = (trial_loss, trial)
        if best is None:
            stalled += 1
            if stalled >= patience:
                return Descent(current, loss, accepted, history, True)
            continue
        loss, current = best
        accepted += 1
        stalled = 0
        history.append(loss)
    return Descent(current, loss, accepted, history, False)
