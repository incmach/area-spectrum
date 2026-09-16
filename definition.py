from functools import cache, lru_cache
import itertools
import math
import numpy as np
import sympy

@cache
def volume_function(d):
    vs = [ [ sympy.Symbol(f'v_{i}_{j}') for j in range(d) ] for i in range(d+1) ] 
    m = sympy.Matrix([ [ v_j - v_0_j for v_0_j, v_j in zip(vs[0],v) ] for v in vs[1:] ])
    return sympy.lambdify(tuple(itertools.chain(vs)), abs(m.det()))

def volume(vs):
    f = volume_function(len(vs)-1)
    return f(*vs)


def area_spectrum(I):
    result = [0]*I.size
    for vs in itertools.product(itertools.product(*(range(n) for n in I.shape)), repeat = len(I.shape)+1):
        result[volume(vs)] += math.prod(int(I[v]) for v in vs)
    return result;


def area_spectrum_jacobian_row_squared_norms(I):
    """
    Returns N where

        N[k] = sum_p ( d area_spectrum(I)[k] / d I[p] )^2

    computed exactly with Python unbounded ints.

    This is the squared scaling factor for spectrum element k:
        scaling_factor(I, k) = sqrt(N[k])
    """
    shape = I.shape
    d = len(shape)

    coords = list(itertools.product(*(range(n) for n in shape)))
    coord_to_idx = {v: i for i, v in enumerate(coords)}
    spectrum_len = I.size

    # J[k][p] = derivative of spectrum element k w.r.t. pixel p
    # Use dicts to avoid allocating a dense spectrum_len x n_pixels matrix.
    J = [dict() for _ in range(spectrum_len)]

    for S in itertools.product(coords, repeat=d + 1):
        k = volume(S)
        if k >= spectrum_len:
            # area_spectrum as written has only I.size entries
            continue

        vals = [int(I[v]) for v in S]
        m = len(vals)

        # prefix and suffix products for O(m) product-except-one
        pref = [1] * (m + 1)
        for i in range(m):
            pref[i + 1] = pref[i] * vals[i]

        suff = [1] * (m + 1)
        for i in range(m - 1, -1, -1):
            suff[i] = suff[i + 1] * vals[i]

        for i, v in enumerate(S):
            prod_excl = pref[i] * suff[i + 1]
            p = coord_to_idx[v]
            row = J[k]
            row[p] = row.get(p, 0) + prod_excl

    return [sum(x * x for x in row.values()) for row in J]


def scaling_factor_squared(I, sigma):
    """Exact integer squared scaling factor for one spectrum index."""
    return area_spectrum_jacobian_row_squared_norms(I)[sigma]


def scaling_factor(I, sigma):
    """Floating point scaling factor = sqrt(row squared norm)."""
    return math.sqrt(scaling_factor_squared(I, sigma))


def target_function_squared_norm(I, target):
    """
    Computes

        sum_sigma ((area_spectrum(I)[sigma] - target[sigma])
                   / scaling_factor(I, sigma))^2
    """
    current = area_spectrum(I)
    N = area_spectrum_jacobian_row_squared_norms(I)

    total = 0.0
    for a, t, n in zip(current, target, N):
        if n != 0:
            r = (a - t) / math.sqrt(n)
            total += r * r
    return total


def area_spectrum_scaled_gradient(I, target):
    """
    Computes the gradient of

        F(I) = sum_k ((area_spectrum(I)[k] - target[k]) / sqrt(N[k]))^2

    with respect to I, treating N[k] = area_spectrum_jacobian_row_squared_norms(I)[k]
    as a fixed preconditioner computed at the current I.

    Returns a nested list of floats with the same shape as I.
    """
    shape = I.shape
    d = len(shape)
    coords = list(itertools.product(*(range(n) for n in shape)))

    # Compute spectrum and exact squared row norms ONCE.
    A = area_spectrum(I)
    N = area_spectrum_jacobian_row_squared_norms(I)   # exact Python ints

    # directions[k] = 2 * (A[k] - target[k]) / N[k]
    # If N[k] == 0, the scaling factor is zero; we set the direction to 0.
    directions = [0.0] * len(A)
    for k in range(len(A)):
        if N[k] != 0:
            directions[k] = 2.0 * (A[k] - target[k]) / float(N[k])

    # Build gradient: for each pixel p, sum over T (length d) of
    # prod(I[T]) * directions[volume((p,) + T)]
    result = np.zeros_like(I, dtype=float).tolist()

    for p in coords:
        assignment_target = result
        for coordinate in p[:-1]:
            assignment_target = assignment_target[coordinate]

        # Each ordered d-tuple T with p at any of the (d+1) vertex positions
        # contributes equally (volume() is permutation-symmetric), so the
        # true derivative includes a factor of (d+1).
        for T in itertools.product(coords, repeat=d):
            idx = volume((p,) + T)
            if idx < len(directions):
                assignment_target[p[-1]] += (
                    (d + 1) * math.prod(int(I[u]) for u in T) * directions[idx]
                )

    return result