"""
Tests for definition.py.

Covers volume(), area_spectrum() and the Jacobian-scaling machinery
(area_spectrum_jacobian_row_squared_norms, scaling_factor*,
target_function_squared_norm, area_spectrum_scaled_gradient).

Runnable directly (python test_definition.py) or via pytest.
All images are small (<= 8x8); the algorithms are intentionally naive.
"""

import itertools
import math

import numpy as np

import definition

VOLUME = definition.volume
AREA_SPECTRUM = definition.area_spectrum
ROW_NORMS = definition.area_spectrum_jacobian_row_squared_norms
SCALING_FACTOR_SQUARED = definition.scaling_factor_squared
SCALING_FACTOR = definition.scaling_factor
TARGET_SQUARED_NORM = definition.target_function_squared_norm
SCALED_GRADIENT = definition.area_spectrum_scaled_gradient


# ---------------------------------------------------------------------------
# volume()
# ---------------------------------------------------------------------------

def test_volume_1d():
    assert VOLUME(((0,), (3,))) == 3
    assert VOLUME(((2,), (2,))) == 0
    assert VOLUME(((5,), (1,))) == 4


def test_volume_2d():
    assert VOLUME(((0, 0), (1, 0), (0, 1))) == 1
    assert VOLUME(((0, 0), (2, 0), (0, 2))) == 4
    assert VOLUME(((1, 1), (1, 1), (0, 0))) == 0
    assert VOLUME(((0, 0), (1, 1), (2, 2))) == 0


def test_volume_3d():
    assert VOLUME(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))) == 1
    assert VOLUME(((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2))) == 8


def test_volume_permutation_invariant():
    pts = ((0, 0), (3, 0), (1, 4))
    expected = VOLUME(pts)
    for perm in itertools.permutations(pts):
        assert VOLUME(perm) == expected


# ---------------------------------------------------------------------------
# area_spectrum()
# ---------------------------------------------------------------------------

def test_area_spectrum_ones_small():
    assert AREA_SPECTRUM(np.ones((1, 1), dtype=np.uint8)) == [1]
    assert AREA_SPECTRUM(np.ones((2, 1), dtype=np.uint8)) == [8, 0]
    assert AREA_SPECTRUM(np.ones((1, 2), dtype=np.uint8)) == [8, 0]
    assert AREA_SPECTRUM(np.ones((2, 2), dtype=np.uint8)) == [40, 24, 0, 0]
    assert AREA_SPECTRUM(np.ones((3, 2), dtype=np.uint8)) == [108, 72, 36, 0, 0, 0]
    assert AREA_SPECTRUM(np.ones((2, 3), dtype=np.uint8)) == [108, 72, 36, 0, 0, 0]


def test_area_spectrum_ones_3x3():
    assert AREA_SPECTRUM(np.ones((3, 3), dtype=np.uint8)) == [
        273, 192, 192, 24, 48, 0, 0, 0, 0,
    ]


def test_area_spectrum_zero_image():
    assert AREA_SPECTRUM(np.zeros((4, 4), dtype=np.uint8)) == [0] * 16


def test_area_spectrum_mixed():
    assert AREA_SPECTRUM(np.array([[1, 0], [0, 1]], dtype=np.uint8)) == [8, 0, 0, 0]
    assert AREA_SPECTRUM(np.array([[1, 1], [0, 1]], dtype=np.uint8)) == [21, 6, 0, 0]


def test_area_spectrum_nonnegative():
    rng = np.random.default_rng(0)
    for shape in [(2, 2), (3, 2), (4, 4), (8, 8), (3, 3, 3)]:
        I = rng.integers(0, 5, size=shape).astype(np.uint8)
        assert all(x >= 0 for x in AREA_SPECTRUM(I))


def test_area_spectrum_symmetric_in_axis_swap():
    I = np.array([[1, 2, 4], [3, 1, 2]], dtype=np.uint8)
    assert AREA_SPECTRUM(I) == AREA_SPECTRUM(I.T)


def test_area_spectrum_mass_conservation():
    d = 3  # number of points (for d+1 = 3 vertices per triple)
    rng = np.random.default_rng(7)
    for shape in [(3, 3), (4, 4), (2, 2, 2)]:
        I = rng.integers(0, 4, size=shape).astype(np.int64)
        triples = sum(int(I[v]) for v in np.ndindex(shape))
        assert sum(AREA_SPECTRUM(I)) == triples ** (len(shape) + 1)


def test_area_spectrum_homogeneous_in_scaling():
    rng = np.random.default_rng(11)
    I = rng.integers(0, 4, size=(3, 3)).astype(np.int64)
    expon = len(I.shape) + 1
    A1 = AREA_SPECTRUM(I)
    A3 = AREA_SPECTRUM(3 * I)
    for k in range(I.size):
        assert A3[k] == (3 ** expon) * A1[k]


# ---------------------------------------------------------------------------
# Jacobian row squared norms
# ---------------------------------------------------------------------------

def _brute_force_row_norms(I):
    """Independent reference: exact derivative of each spectrum entry wrt
    each pixel, via the product rule, then summed squared per spectrum index."""
    shape = I.shape
    d = len(shape)
    coords = list(np.ndindex(shape))
    n_pix = np.prod(shape)
    J = [dict() for _ in range(I.size)]
    # A_k = sum over tuples of (d+1) points with volume k of prod of I at those points.
    # dA_k/dI[p] = sum over positions i in the tuple of prod of the other d entries.
    for S in itertools.product(coords, repeat=d + 1):
        k = VOLUME(S)
        if k >= I.size:
            continue
        vals = [int(I[v]) for v in S]
        for i in range(len(vals)):
            p = coords.index(S[i])
            prod_excl = math.prod(vals[:i] + vals[i + 1:])
            J[k][p] = J[k].get(p, 0) + prod_excl
    return [sum(x * x for x in row.values()) for row in J]


def test_row_norms_matches_brute_force():
    rng = np.random.default_rng(3)
    for shape in [(2, 2), (3, 2), (4, 4), (2, 2, 2)]:
        I = rng.integers(0, 4, size=shape).astype(np.int64)
        assert ROW_NORMS(I) == _brute_force_row_norms(I)


def test_row_norms_known_values():
    assert ROW_NORMS(np.ones((2, 2), dtype=np.uint8)) == [3600, 1296, 0, 0]
    assert ROW_NORMS(np.ones((3, 2), dtype=np.uint8)) == [17496, 8208, 2376, 0, 0, 0]


def test_row_norms_nonnegative():
    rng = np.random.default_rng(5)
    for shape in [(3, 3), (4, 4)]:
        I = rng.integers(0, 5, size=shape).astype(np.uint8)
        assert all(x >= 0 for x in ROW_NORMS(I))


# ---------------------------------------------------------------------------
# scaling factors
# ---------------------------------------------------------------------------

def test_scaling_factor_squared_matches_row_norms():
    I = np.ones((3, 2), dtype=np.uint8)
    norms = ROW_NORMS(I)
    for sigma in range(I.size):
        assert SCALING_FACTOR_SQUARED(I, sigma) == norms[sigma]


def test_scaling_factor_is_sqrt():
    I = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    for sigma in range(I.size):
        assert SCALING_FACTOR(I, sigma) == math.sqrt(
            SCALING_FACTOR_SQUARED(I, sigma)
        )


# ---------------------------------------------------------------------------
# target_function_squared_norm
# ---------------------------------------------------------------------------

def test_target_squared_norm_zero_at_target():
    rng = np.random.default_rng(9)
    for shape in [(2, 2), (3, 2)]:
        I = rng.integers(1, 6, size=shape).astype(np.uint8)
        assert TARGET_SQUARED_NORM(I, AREA_SPECTRUM(I)) == 0.0


def test_target_squared_norm_nonnegative():
    I = np.ones((3, 3), dtype=np.uint8)
    target = [10] * I.size
    assert TARGET_SQUARED_NORM(I, target) >= 0.0


def test_target_squared_norm_matches_definition():
    I = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    target = [5, 5, 5, 5]
    expected = 0.0
    for a, t, n in zip(AREA_SPECTRUM(I), target, ROW_NORMS(I)):
        if n != 0:
            expected += ((a - t) / math.sqrt(n)) ** 2
    assert TARGET_SQUARED_NORM(I, target) == expected


# ---------------------------------------------------------------------------
# area_spectrum_scaled_gradient
# ---------------------------------------------------------------------------

def _spectrum_float(I, p=None, value=None):
    """area_spectrum computed over float pixel values.

    Lets the finite-difference test perturb a single pixel continuously,
    which the integer-truncating area_spectrum() cannot do.
    """
    shape = I.shape
    M = np.array(I, dtype=float)
    if p is not None:
        M[p] = value
    result = [0.0] * I.size
    for S in itertools.product(np.ndindex(shape), repeat=len(shape) + 1):
        result[VOLUME(S)] += math.prod(M[v] for v in S)
    return result


def _brute_force_gradient(I, target):
    """Independent reference gradient.

    Treats row norms as a fixed preconditioner computed at I:
        F(x) = sum_k ((A_k(x) - target_k) / sqrt(N_k(I)))^2
    So dF/dI[p] = sum_k 2 (A_k - t_k) / N_k * dA_k/dI[p] with dA_k/dI[p]
    computed exactly from the product rule.
    """
    shape = I.shape
    d = len(shape)
    coords = list(np.ndindex(shape))
    A = AREA_SPECTRUM(I)
    norms = ROW_NORMS(I)
    directions = [
        2.0 * (a - t) / float(n) if n != 0 else 0.0
        for a, t, n in zip(A, target, norms)
    ]
    J = [dict() for _ in range(I.size)]
    for S in itertools.product(coords, repeat=d + 1):
        k = VOLUME(S)
        if k >= I.size:
            continue
        vals = [int(I[v]) for v in S]
        for i in range(len(vals)):
            p = coords.index(S[i])
            prod_excl = math.prod(vals[:i] + vals[i + 1:])
            J[k][p] = J[k].get(p, 0) + prod_excl
    grad = np.zeros(shape, dtype=float)
    for p in np.ndindex(shape):
        pi = coords.index(p)
        grad[p] = sum(J[k].get(pi, 0) * directions[k] for k in range(len(directions)))
    return grad


def test_scaled_gradient_zero_at_target():
    I = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    target = AREA_SPECTRUM(I)
    grad = SCALED_GRADIENT(I, target)
    assert np.allclose(grad, np.zeros_like(grad))


def test_scaled_gradient_matches_brute_force():
    rng = np.random.default_rng(13)
    for shape in [(2, 2), (3, 2), (4, 4)]:
        I = rng.integers(1, 5, size=shape).astype(np.uint8)
        target = rng.integers(1, 10, size=I.size)
        assert np.allclose(
            np.asarray(SCALED_GRADIENT(I, target)),
            _brute_force_gradient(I, target),
        )


def test_scaled_gradient_matches_finite_difference():
    I = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    target = [21, 5, 0, 0]
    norm = ROW_NORMS(I)
    h = 0.005

    def F(x):
        total = 0.0
        for a, t, n in zip(x, target, norm):
            if n != 0:
                total += ((a - t) / math.sqrt(n)) ** 2
        return total

    grad = np.asarray(SCALED_GRADIENT(I, target))
    for p in np.ndindex(I.shape):
        fd = (
            F(_spectrum_float(I, p, float(I[p]) + h))
            - F(_spectrum_float(I, p, float(I[p]) - h))
        ) / (2 * h)
        assert np.isclose(grad[p], fd, rtol=1e-5, atol=1e-5), (p, grad[p], fd)


def test_scaled_gradient_shape():
    I = np.ones((4, 4), dtype=np.uint8)
    grad = SCALED_GRADIENT(I, [1] * I.size)
    assert np.asarray(grad).shape == I.shape


def test_scaled_gradient_scaled_with_pixel():
    # For a 1x1 image, F(I) is a function of the single pixel only and the
    # gradient must vanish at the minimum (I^3 == target).
    I = np.array([[3]], dtype=np.uint8)      # A = [27]
    target = [27]                            # already at the target
    assert np.allclose(SCALED_GRADIENT(I, target), [[0.0]])


# ---------------------------------------------------------------------------
# main runner (also works without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    g = globals()
    failing = 0
    for name in sorted(g):
        if name.startswith("test_") and callable(g[name]):
            try:
                g[name]()
                print("PASS", name)
            except Exception as e:
                failing += 1
                print("FAIL", name, "->", e)
    if failing:
        raise SystemExit(1)
    print("all definition tests passed")