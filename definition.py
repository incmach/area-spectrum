from functools import cache
import itertools
import math

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
    """Exact area spectrum: R[k] = sum over ordered (d+1)-tuples S of prod I[S]."""
    d = len(I.shape)
    coords = coordinates(I)
    n_spectrum = spectrum_length(I)
    result = [0] * n_spectrum
    for S in itertools.product(coords, repeat=d + 1):
        k = volume(S)
        result[k] += math.prod(int(I[v]) for v in S)
    return result


def jacobian_transpose(I, w):
    """Exact J^T w for the area-spectrum Jacobian: result[p] = sum_k J[k][p] w[k].

    The Jacobian is never materialized. An ordered (d+1)-tuple S contributes
    the product of the values it omits to row k = volume(S), once for each
    position i at which S carries the pixel S[i], so the prefix/suffix products
    yield all d+1 of those contributions together.

    The result is a flat list ordered like I.ravel(). Arithmetic stays exact
    whenever w is integral.
    """
    d = len(I.shape)
    coords = coordinates(I)
    index = {v: i for i, v in enumerate(coords)}
    result = [0] * I.size
    for S in itertools.product(coords, repeat=d + 1):
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
    for S in itertools.product(coords, repeat=d + 1):
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
