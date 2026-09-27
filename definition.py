"""The area spectrum, its Jacobian, and the gradient of a normalized target.

This is the reference implementation: exact, slow, and written to be read
against. It enumerates every ordered tuple of distinct grid points and bins by
volume, which costs pixels**(d+1) and is not meant for real images.

The spectrum and the gradient are here together on purpose. The gradient is
J^T w, the same offset enumeration as the spectrum with per-bin weights applied
instead of summed, so under a triple-correlation or NTT formulation it is the
adjoint of the forward pass rather than a separate problem. Whatever speeds up
the spectrum speeds up the gradient by the same route, so they are written,
optimised and reviewed as one thing. The search that consumes a gradient lives
in descent.py, which knows only this module's interface.

Distinct points only. A nonzero determinant already forces the points of a
tuple to be distinct, so that requirement is a no-op for every bin above 0; it
removes the collinear tuples in bin 0 that name a point twice, which is what
leaves the spectrum trilinear and the gradient exact.
"""
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
    definition.py's jacobian_transpose is exact rather than merely first-order.
    """
    d = len(I.shape)
    coords = coordinates(I)
    n_spectrum = spectrum_length(I)
    result = [0] * n_spectrum
    for S in itertools.permutations(coords, d + 1):
        result[volume(S)] += math.prod(int(I[v]) for v in S)
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

    Bins no tuple of distinct points reaches come out 0, so an image can be
    nonzero while its row sums are all zero. That is the same situation as an
    unreachable bin, and it leaves the gradient identically zero.
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


def jacobian_column(I, pixel):
    """Exact Jacobian column C[k] = d area_spectrum(I)[k] / d I[pixel], for one pixel.

    The third orientation of the Jacobian, after jacobian_transpose (indexed by
    pixel) and jacobian_row_sums (indexed by bin, all pixels at once): this one is
    indexed by bin but selects a single pixel, so it is the transpose of one row
    of the Jacobian and the same length as a spectrum.

    A tuple of d+1 distinct points contributes the product of the d values it
    omits when it names `pixel`, once per position at which it carries that
    pixel, which is the same prefix/suffix product jacobian_transpose uses.
    Because the points are distinct, no term squares a value and the derivative
    is exact rather than first-order.

    What buys this is trilinearity. Every term is a product of distinct pixel
    values, so the spectrum is linear in any one pixel and

        area_spectrum(I + step * e_p)[k] == area_spectrum(I)[k] + step * C[k]

    holds exactly for an integer step, in both directions. A search that needs
    the spectrum of a one-pixel edit can therefore add this column instead of
    recomputing the spectrum, and the resulting value is not an approximation of
    the spectrum but the spectrum.

    `pixel` is a coordinate tuple, and is required to be one of them.
    """
    d = len(I.shape)
    coords = coordinates(I)
    if pixel not in coords:
        raise ValueError(f"pixel {pixel} is not a coordinate of an image of shape {I.shape}")
    n_spectrum = spectrum_length(I)
    result = [0] * n_spectrum
    for S in itertools.permutations(coords, d + 1):
        if pixel not in S:
            continue
        result[volume(S)] += math.prod(int(I[v]) for v in S if v != pixel)
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


def spectrum_gradient(I, target, scale, spectrum=None):
    """Gradient of the normalized-residual loss, as a flat list like I.ravel().

    The bin weights are the derivative of the squared normalized residual, so
    the pixel gradient is J^T w, with w from spectrum_residual_weights. Same
    enumeration as the spectrum itself, weighted instead of summed, which is
    what makes this the adjoint of the forward pass rather than a separate
    computation: an implementation that speeds up the spectrum by triple
    correlation or NTT speeds this up by the same route.

    This is an exact quantity, not a first-order one. area_spectrum sums over
    tuples of distinct points, so no term names a pixel twice and the spectrum
    is multilinear; it is therefore linear in each pixel separately, and the
    change a unit step at p makes to bin k is exactly J[k][p], which
    jacobian_column returns. Ranking by this gradient orders steps by their true
    effect on the residual, though the loss is quadratic in the spectrum, so the
    best step by loss is not always the best by gradient. Confirm a step against
    the loss before committing to it.

    Pass `spectrum` to supply a spectrum of I that the caller already has -- by
    trilinearity, or as the sum of the columns of the steps taken so far. The
    weights need it and it is the expensive half of this call, so a caller that
    has it should not pay for it twice. It must be the spectrum of this I;
    passing anything else returns a gradient of the wrong loss.
    """
    if spectrum is None:
        spectrum = area_spectrum(I)
    w = spectrum_residual_weights(spectrum, target, scale)
    return jacobian_transpose(I, w)
