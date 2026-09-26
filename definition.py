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


def jacobian_row_sums(I):
    """Exact L1 row sums R[k] = sum_p ( d area_spectrum(I)[k] / d I[p] ).

    Crude scaling factor: all row entries are nonnegative, so this is the
    L1 norm of row k of the Jacobian. Used to divide out the huge dynamic
    range of the raw spectrum.
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
