"""
ntt.py

NTT-based (galois) computation of area_spectrum and of the exact integer
squared row norms of the Jacobian of area_spectrum, using per-first-diff
slices of the (d+1)-way correlation.

The NTT prime is chosen from the dtype's maximum representable value and
the image size.  No extra safety margin is applied.
"""

import itertools
import math

import numpy as np
import galois

from definition import volume


# ====================================================================
# Prime selection
# ====================================================================

def _dtype_max(dtype):
    """Largest representable value of a NumPy integer dtype."""
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return 1
    if np.issubdtype(dtype, np.integer):
        return int(np.iinfo(dtype).max)
    raise TypeError(f"Unsupported dtype for NTT: {dtype}")


def find_ntt_prime(min_value, transform_sizes):
    """
    Return a prime p > min_value such that p - 1 is divisible by every
    element of transform_sizes (so primitive roots of unity of those
    orders exist mod p).
    """
    L = 1
    for s in transform_sizes:
        L = L * s // math.gcd(L, s)
    target = max(2, int(min_value) + 1)
    rem = (target - 1) % L
    p = target + ((L - rem) % L)
    if p <= target:
        p += L
    while not galois.is_prime(p):
        p += L
    return int(p)


def choose_ntt_prime(I):
    """
    Choose an NTT prime large enough to hold the largest possible
    (d+1)-way correlation value given I.dtype and I.shape.

    Bound: n_total * V**(d+1), with V = max representable value of
    I.dtype (not the observed max in I).  No safety multiplier.

    Note: callers must not widen I (e.g. to int64) before calling this,
    or V becomes the int64 maximum and the required prime is
    astronomically large.
    """
    I = np.asarray(I)
    d = I.ndim
    n_total = I.size
    V = _dtype_max(I.dtype)
    max_corr = 1 if V <= 0 else n_total * (V ** (d + 1))
    ntt_shape = tuple(1 << max(0, 2 * n - 2).bit_length() for n in I.shape)
    return find_ntt_prime(max_corr, list(ntt_shape)), ntt_shape


# ====================================================================
# Multi-dimensional NTT (galois.ntt operates on the last axis)
# ====================================================================

def _ntt_along_axis(x_gf, axis):
    x_gf = np.moveaxis(x_gf, axis, -1)
    shape = x_gf.shape
    flat = x_gf.reshape(-1, shape[-1])
    out = flat.copy()
    for i in range(flat.shape[0]):
        out[i] = galois.ntt(flat[i])
    out = out.reshape(shape)
    return np.moveaxis(out, -1, axis)


def _intt_along_axis(x_gf, axis):
    x_gf = np.moveaxis(x_gf, axis, -1)
    shape = x_gf.shape
    flat = x_gf.reshape(-1, shape[-1])
    out = flat.copy()
    for i in range(flat.shape[0]):
        out[i] = galois.intt(flat[i])
    out = out.reshape(shape)
    return np.moveaxis(out, -1, axis)


def _nd_ntt(x_gf):
    result = x_gf
    for ax in range(x_gf.ndim):
        result = _ntt_along_axis(result, ax)
    return result


def _nd_intt(x_gf):
    result = x_gf
    for ax in range(x_gf.ndim):
        result = _intt_along_axis(result, ax)
    return result


# ====================================================================
# Difference helpers
# ====================================================================

def _all_diffs(shape):
    """All difference vectors fitting inside an image of the given shape."""
    return list(itertools.product(
        *[range(-(n - 1), n) for n in shape]
    ))


def _valid_grid(shape, diffs):
    """
    Origins x such that x + a is inside the image for every a in diffs.
    Returns a list of 1-D index arrays (one per axis), or None if empty.
    """
    lo, hi = [], []
    for ax, n in enumerate(shape):
        if diffs:
            l = max(0, max(-a[ax] for a in diffs))
            h = min(n, min(n - a[ax] for a in diffs))
        else:
            l, h = 0, n
        lo.append(l)
        hi.append(h)
    if any(l >= h for l, h in zip(lo, hi)):
        return None
    return [np.arange(l, h) for l, h in zip(lo, hi)]


def _shifted_product(I, xs, diffs):
    """
    Compute I[x] * prod_{a in diffs} I[x + a] over the grid xs.
    """
    grid = np.ix_(*xs)
    prod_arr = I[grid].copy()
    for a in diffs:
        shifted = [x + a[ax] for ax, x in enumerate(xs)]
        prod_arr = prod_arr * I[np.ix_(*shifted)]
    return prod_arr


def _diff_product(I, xs, diffs):
    """
    Compute prod_{a in diffs} I[x + a] over the grid xs, WITHOUT the
    origin factor I[x].

    This is the term appearing in the derivative fields
        D_k(p) = sum_a prod_i I[p + a_i]
    where the sum runs over difference d-tuples a.  The origin factor is
    only wanted in _shifted_product() (used by the area_spectrum NTT).
    diffs must be non-empty.
    """
    first = diffs[0]
    shifted = [x + first[ax] for ax, x in enumerate(xs)]
    prod_arr = I[np.ix_(*shifted)].copy()
    for a in diffs[1:]:
        shifted = [x + a[ax] for ax, x in enumerate(xs)]
        prod_arr = prod_arr * I[np.ix_(*shifted)]
    return prod_arr


# ====================================================================
# area_spectrum via NTT
# ====================================================================

def area_spectrum_ntt(I, prime=None):
    """
    NTT-based area spectrum.

    For 2D images:

        A[k] = sum_{p,q,r} I[p] I[q] I[r] * [|det(q-p, r-p)| == k]

    Group by first difference a = q - p.  For fixed a, the inner sum over
    r is a cross-correlation of J_a(p) = I[p] I[p + a] with I, evaluated
    at every lag b = r - p; the lag b is binned by |det(a, b)|.  The NTT
    computes all lags b at once.  The generalization to d dimensions is
    obtained by fixing the first d-1 differences.
    """
    if prime is None:
        # Choose the prime from the ORIGINAL dtype; casting to int64 first
        # would make _dtype_max() return 2**63-1 and the required prime
        # astronomically large for typical image dtypes.
        prime, ntt_shape = choose_ntt_prime(I)
    else:
        ntt_shape = tuple(1 << max(0, 2 * n - 2).bit_length() for n in I.shape)

    I = np.asarray(I, dtype=np.int64)
    shape = I.shape
    d = len(shape)
    n_total = I.size

    GF = galois.GF(prime)

    # Zero-pad I to ntt_shape, forward NTT once.
    I_pad = np.zeros(ntt_shape, dtype=np.int64)
    I_pad[tuple(slice(0, n) for n in shape)] = I
    I_hat = _nd_ntt(GF(I_pad))

    result = [0] * n_total
    all_diffs = _all_diffs(shape)

    for prefix in itertools.product(all_diffs, repeat=d - 1):
        xs = _valid_grid(shape, prefix)
        if xs is None:
            continue

        # J(x) = I[x] * prod_i I[x + prefix_i] over valid origins.
        J_block = _shifted_product(I, xs, prefix)

        J_pad = np.zeros(ntt_shape, dtype=np.int64)
        J_pad[np.ix_(*xs)] = J_block

        # Reverse-and-roll to turn convolution into cross-correlation.
        rev = tuple(slice(None, None, -1) for _ in range(d))
        J_rev = np.roll(J_pad[rev], (1,) * d, axis=tuple(range(d)))
        J_rev_hat = _nd_ntt(GF(J_rev))

        T_hat = J_rev_hat * I_hat
        T_np = np.asarray(_nd_intt(T_hat), dtype=np.int64)

        for a_d in all_diffs:
            idx = tuple(a % m for a, m in zip(a_d, ntt_shape))
            val = int(T_np[idx])
            if val == 0:
                continue
            k = volume(((0,) * d,) + tuple(prefix) + (a_d,))
            if 0 <= k < n_total:
                result[k] += val

    return result


# ====================================================================
# Jacobian row squared norms (reorganized per difference pattern)
# ====================================================================

def area_spectrum_jacobian_row_squared_norms(I):
    """
    Exact integer squared row norms of the Jacobian of area_spectrum.

        N_k = sum_p ( dA_k / dI_p )^2

    By the product rule and vertex symmetry,

        dA_k / dI_p = (d+1) * sum_{a: |det(a)|=k} prod_i I[p + a_i]

    where the outer sum is over d-tuples of differences.  We build, for
    each bin k, the derivative field D_k(p) = sum_a prod_i I[p + a_i]
    and then N_k = (d+1)^2 * sum_p D_k(p)^2.
    """
    I = np.asarray(I, dtype=np.int64)
    shape = I.shape
    d = len(shape)
    n_total = I.size

    all_diffs = _all_diffs(shape)

    # Bucket difference d-tuples by their volume bin.
    patterns_by_bin = [[] for _ in range(n_total)]
    for a in itertools.product(all_diffs, repeat=d):
        k = volume(((0,) * d,) + a)
        if 0 <= k < n_total:
            patterns_by_bin[k].append(a)

    N = [0] * n_total
    for k in range(n_total):
        D = np.zeros(shape, dtype=np.int64)
        for a in patterns_by_bin[k]:
            xs = _valid_grid(shape, a)
            if xs is None:
                continue
            prod_arr = _diff_product(I, xs, a)
            D[np.ix_(*xs)] += prod_arr
        N[k] = (d + 1) ** 2 * sum(int(x) * int(x) for x in D.flat)
    return N


# ====================================================================
# Combined objective and scaled gradient
# ====================================================================

def area_spectrum_and_jacobian_norms_ntt(I, prime=None):
    """
    Returns (A, N) where A = area_spectrum_ntt(I) and
    N = area_spectrum_jacobian_row_squared_norms(I).
    """
    A = area_spectrum_ntt(I, prime=prime)
    N = area_spectrum_jacobian_row_squared_norms(I)
    return A, N


def scaled_objective_ntt(I, target, prime=None):
    """
        F(I) = sum_k ((A_k(I) - target_k) / sqrt(N_k(I)))^2
    with N_k the exact integer squared row norms.
    """
    A, N = area_spectrum_and_jacobian_norms_ntt(I, prime=prime)
    total = 0.0
    for a, t, n in zip(A, target, N):
        if n != 0:
            r = (a - t) / math.sqrt(n)
            total += r * r
    return total


def area_spectrum_scaled_gradient_ntt(I, target, prime=None):
    """
    Gradient of  F(I) = sum_k ((A_k - t_k) / sqrt(N_k))^2  w.r.t. I,
    treating N_k as a fixed preconditioner computed at the current I.

    Uses the difference-pattern reorganization:

        dF/dI_p = sum_a coef_{|det(a)|} * prod_i I[p + a_i]

    with coef_k = 2 (A_k - t_k) (d+1) / N_k.  Memory O(n^d),
    time O(n^{d^2}).
    """
    I, prime = np.asarray(I), prime
    shape = I.shape
    d = len(shape)
    n_total = I.size

    # Do NOT widen I to int64 before the NTT helper runs: choose_ntt_prime
    # sizes the prime from I.dtype, and an int64 array would demand a
    # prime far larger than needed.  Cast only for the arithmetic below.
    A, N = area_spectrum_and_jacobian_norms_ntt(I, prime=prime)

    coef = [0.0] * n_total
    for k in range(n_total):
        if N[k] != 0:
            coef[k] = 2.0 * (A[k] - target[k]) * (d + 1) / float(N[k])

    I = np.asarray(I, dtype=np.int64)
    grad = np.zeros(shape, dtype=float)
    all_diffs = _all_diffs(shape)

    for a in itertools.product(all_diffs, repeat=d):
        k = volume(((0,) * d,) + a)
        if k >= n_total or coef[k] == 0.0:
            continue
        xs = _valid_grid(shape, a)
        if xs is None:
            continue
        prod_arr = _diff_product(I, xs, a)
        grad[np.ix_(*xs)] += coef[k] * prod_arr.astype(float)

    return grad.tolist()