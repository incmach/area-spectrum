from fractions import Fraction
import itertools
import math

import numpy as np
import pytest

import definition

VOLUME = definition.volume
AREA_SPECTRUM = definition.area_spectrum
ROW_SUMS = definition.jacobian_row_sums
GRADIENT = definition.normalized_spectrum_gradient


# ---------------------------------------------------------------------------
# helpers: independent references
# ---------------------------------------------------------------------------

def _reference_rows(I):
    """Full Jacobian rows: rows[k][p] = d area_spectrum[k] / d I[p].

    Independent of definition.py: accumulates the product rule per pixel into
    a dict instead of using prefix/suffix products.
    """
    coords = list(np.ndindex(I.shape))
    d = len(I.shape)
    rows = [dict() for _ in range(definition.spectrum_length(I))]
    for S in itertools.product(coords, repeat=d + 1):
        k = VOLUME(S)
        vals = [int(I[v]) for v in S]
        m = len(vals)
        for i in range(m):
            contrib = math.prod(vals[:i] + vals[i + 1:])
            rows[k][S[i]] = rows[k].get(S[i], 0) + contrib
    return rows


def _reference_row_sums(I):
    return [sum(row.values()) for row in _reference_rows(I)]


def _spectrum_float(I):
    """Area spectrum over float pixel values, for finite differencing.

    area_spectrum casts with int(), which makes it a step function, so its
    finite differences vanish. This is the polynomial relaxation the gradient
    is actually the derivative of.
    """
    coords = list(np.ndindex(I.shape))
    d = len(I.shape)
    result = [0.0] * definition.spectrum_length(I)
    for S in itertools.product(coords, repeat=d + 1):
        result[VOLUME(S)] += math.prod(float(I[v]) for v in S)
    return result


def _normalized_loss(I, scale, target):
    """The objective normalized_spectrum_gradient descends, defined independently.

    Written against the float spectrum so it can be finite differenced.
    """
    A = _spectrum_float(I)
    return sum(
        ((a - t) / s) ** 2 for a, t, s in zip(A, target, scale) if s > 0
    )


# ---------------------------------------------------------------------------
# volume
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
# area_spectrum
# ---------------------------------------------------------------------------

def test_area_spectrum_ones_1x1():
    assert AREA_SPECTRUM(np.ones((1, 1), dtype=np.uint8)) == [1]


def test_area_spectrum_ones_2x1():
    assert AREA_SPECTRUM(np.ones((2, 1), dtype=np.uint8)) == [8, 0]
    assert AREA_SPECTRUM(np.ones((1, 2), dtype=np.uint8)) == [8, 0]


def test_area_spectrum_ones_2x2():
    # 4^3 = 64 ordered triples; 4 triangles x 6 orderings = 24 with area 1
    assert AREA_SPECTRUM(np.ones((2, 2), dtype=np.uint8)) == [40, 24, 0, 0]


def test_area_spectrum_ones_3x2():
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


def test_area_spectrum_length():
    for shape in [(2, 2), (3, 3), (4, 4), (6, 6), (3, 2)]:
        I = np.ones(shape, dtype=np.uint8)
        assert len(AREA_SPECTRUM(I)) == definition.spectrum_length(I)


def test_area_spectrum_length_is_image_size_in_2d():
    for shape in [(2, 2), (3, 3), (4, 4), (6, 6), (5, 4)]:
        I = np.ones(shape, dtype=np.uint8)
        assert len(AREA_SPECTRUM(I)) == I.size


def test_area_spectrum_nonnegative():
    rng = np.random.default_rng(0)
    for shape in [(2, 2), (3, 2), (4, 4), (5, 4), (2, 2, 2)]:
        I = rng.integers(0, 5, size=shape).astype(np.uint8)
        assert all(x >= 0 for x in AREA_SPECTRUM(I))


def test_area_spectrum_symmetric_in_axis_swap():
    I = np.array([[1, 2, 4], [3, 1, 2]], dtype=np.uint8)
    assert AREA_SPECTRUM(I) == AREA_SPECTRUM(I.T)


def test_area_spectrum_mass_conservation():
    rng = np.random.default_rng(7)
    for shape in [(3, 3), (4, 4), (2, 2, 2)]:
        I = rng.integers(0, 4, size=shape).astype(np.int64)
        total = sum(int(I[v]) for v in np.ndindex(shape))
        assert sum(AREA_SPECTRUM(I)) == total ** (len(shape) + 1)


def test_area_spectrum_homogeneous():
    rng = np.random.default_rng(11)
    for shape in [(2, 2), (3, 3), (4, 4)]:
        I = rng.integers(0, 4, size=shape).astype(np.int64)
        degree = len(I.shape) + 1
        A1 = AREA_SPECTRUM(I)
        A3 = AREA_SPECTRUM(3 * I)
        for k in range(I.size):
            assert A3[k] == (3 ** degree) * A1[k]


# ---------------------------------------------------------------------------
# jacobian_row_sums
# ---------------------------------------------------------------------------

def test_row_sums_known_values():
    assert ROW_SUMS(np.ones((1, 1), dtype=np.uint8)) == [3]
    assert ROW_SUMS(np.ones((2, 1), dtype=np.uint8)) == [24, 0]
    assert ROW_SUMS(np.ones((2, 2), dtype=np.uint8)) == [120, 72, 0, 0]
    assert ROW_SUMS(np.ones((3, 2), dtype=np.uint8)) == [324, 216, 108, 0, 0, 0]
    assert ROW_SUMS(np.ones((3, 3), dtype=np.uint8)) == [819, 576, 576, 72, 144, 0, 0, 0, 0]


def test_row_sums_zero_image():
    assert ROW_SUMS(np.zeros((4, 4), dtype=np.uint8)) == [0] * 16


def test_row_sums_matches_reference():
    rng = np.random.default_rng(3)
    for shape in [(2, 2), (3, 2), (4, 4), (2, 2, 2), (5, 1)]:
        I = rng.integers(0, 4, size=shape).astype(np.int64)
        assert ROW_SUMS(I) == _reference_row_sums(I)


def test_row_sums_nonnegative():
    rng = np.random.default_rng(5)
    for shape in [(3, 3), (4, 4), (6, 6)]:
        I = rng.integers(0, 5, size=shape).astype(np.uint8)
        assert all(x >= 0 for x in ROW_SUMS(I))


def test_row_sums_length():
    for shape in [(2, 2), (3, 3), (4, 4), (8, 8)]:
        I = np.ones(shape, dtype=np.uint8)
        assert len(ROW_SUMS(I)) == definition.spectrum_length(I)


def test_row_sums_length_is_image_size_in_2d():
    for shape in [(2, 2), (3, 3), (4, 4), (8, 8)]:
        I = np.ones(shape, dtype=np.uint8)
        assert len(ROW_SUMS(I)) == I.size


def test_row_sums_matches_spectrum_length():
    rng = np.random.default_rng(31)
    for shape in [(2, 2), (3, 2), (4, 4), (2, 2, 2), (5, 1)]:
        I = rng.integers(0, 4, size=shape).astype(np.int64)
        assert len(ROW_SUMS(I)) == len(AREA_SPECTRUM(I))


def test_row_sums_homogeneous():
    rng = np.random.default_rng(11)
    for shape in [(2, 2), (3, 3), (4, 4)]:
        I = rng.integers(0, 4, size=shape).astype(np.int64)
        degree = len(I.shape)
        R1 = ROW_SUMS(I)
        R3 = ROW_SUMS(3 * I)
        for k in range(I.size):
            assert R3[k] == (3 ** degree) * R1[k]


def _derivative_at_zero(samples):
    """Exact g'(0) for a degree-(len(samples)-1) polynomial, in integer arithmetic.

    g(t) = S[I + t*1] is a product of (d+1) pixel values, so it is a degree
    (d+1) polynomial in t. Newton's forward-difference identity gives

        g'(0) = sum_j (-1)^(j+1) * Delta^j g(0) / j

    exactly, needing d+2 samples. Works for any d, unlike a hardcoded formula.
    """
    diffs = list(samples)
    result = Fraction(0)
    sign = 1
    for j in range(1, len(samples)):
        diffs = [b - a for a, b in zip(diffs, diffs[1:])]
        result += sign * Fraction(diffs[0], j)
        sign = -sign
    return result


def test_row_sums_is_directional_derivative():
    """L1 row sums equal d/dt S[I + t*1] at t=0, computed exactly."""
    rng = np.random.default_rng(13)
    for shape in [(2, 2), (3, 2), (3, 3), (4, 4), (5, 1), (2, 2, 2), (2, 2, 3)]:
        I = rng.integers(1, 5, size=shape).astype(np.int64)
        R = ROW_SUMS(I)
        g = [AREA_SPECTRUM(I + t) for t in range(len(shape) + 2)]
        for k in range(len(R)):
            deriv = _derivative_at_zero([s[k] for s in g])
            assert deriv.denominator == 1, (shape, k, deriv)
            assert R[k] == deriv.numerator, (shape, k, R[k], deriv)


def test_reference_rows_match_finite_difference():
    """Per-pixel Jacobian rows agree with a 5-point central difference.

    A 2-point central difference is biased here: tuples repeating a pixel make
    the spectrum quadratic in that pixel, so the h^2 term does not vanish. The
    5-point stencil cancels it, leaving O(h^4).
    """
    rng = np.random.default_rng(23)
    for shape in [(2, 2), (3, 2), (3, 3), (4, 4)]:
        I = rng.integers(1, 5, size=shape).astype(np.int64)
        rows = _reference_rows(I)
        h = 0.5
        for p in np.ndindex(shape):
            shifted = []
            for offset in (-2, -1, 1, 2):
                M = I.astype(float)
                M[p] += offset * h
                shifted.append(_spectrum_float(M))
            for k in range(I.size):
                fd = (
                    shifted[0][k]
                    - 8 * shifted[1][k]
                    + 8 * shifted[2][k]
                    - shifted[3][k]
                ) / (12 * h)
                assert math.isclose(fd, rows[k].get(p, 0), rel_tol=1e-6, abs_tol=1e-6), (
                    shape, p, k, fd, rows[k].get(p, 0),
                )


# ---------------------------------------------------------------------------
# normalizations built on the row sums
# ---------------------------------------------------------------------------

def test_normalized_spectrum_is_scale_equivariant():
    """S has degree d+1 and the L1 row sums degree d, so S/L1 has degree 1."""
    rng = np.random.default_rng(17)
    for shape in [(3, 3), (4, 4)]:
        I = rng.integers(1, 5, size=shape).astype(np.int64)
        A = AREA_SPECTRUM(I)
        R = ROW_SUMS(I)
        for c in (2, 3, 5):
            A2 = AREA_SPECTRUM(c * I)
            R2 = ROW_SUMS(c * I)
            for k in range(I.size):
                if R[k] == 0:
                    assert R2[k] == 0
                    continue
                assert math.isclose(A2[k] / R2[k], c * A[k] / R[k], rel_tol=1e-12)


def test_dead_bins_above_natural_support():
    """Bins above (W-1)(H-1) are identically zero, as are their row sums."""
    rng = np.random.default_rng(19)
    for shape in [(2, 2), (3, 3), (4, 4), (8, 5)]:
        I = rng.integers(0, 5, size=shape).astype(np.uint8)
        natural = (shape[0] - 1) * (shape[1] - 1)
        A = AREA_SPECTRUM(I)
        R = ROW_SUMS(I)
        assert all(x == 0 for x in A[natural + 1:])
        assert all(x == 0 for x in R[natural + 1:])


def test_dimension_multiplier():
    assert definition.dimension_multiplier(1) == 1
    assert definition.dimension_multiplier(2) == 1
    assert definition.dimension_multiplier(3) == 2
    assert definition.dimension_multiplier(4) == 4


def test_dimension_multiplier_rejects_too_many_dims():
    for d in (definition.MAX_DIMENSION + 1, 6):
        with pytest.raises(ValueError):
            definition.dimension_multiplier(d)
    with pytest.raises(ValueError):
        definition.spectrum_length(np.ones((2,) * (definition.MAX_DIMENSION + 1)))


def test_spectrum_length_exceeds_max_volume_3d():
    """spectrum_length must cover the true max volume: 2*(n-1)^3 > n^3.

    Asserted on the length alone, without enumerating tuples -- a 5x5x5
    enumeration is 244M tuples, so the bound has to be checked arithmetically.
    """
    for n in range(2, 12):
        shape = (n, n, n)
        max_volume = 2 * (n - 1) ** 3
        assert definition.spectrum_length(np.empty(shape, dtype=np.uint8)) > max_volume, n
    # and the overflow really starts at n=5, not before
    assert 2 * 4 ** 3 > 5 ** 3
    assert not 2 * 3 ** 3 > 4 ** 3


def test_dimension_multiplier_covers_all_small_shapes():
    for shape in [(2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (2, 3, 7), (3, 4, 11)]:
        I = np.empty(shape, dtype=np.uint8)
        d = len(shape)
        max_volume = definition.dimension_multiplier(d)
        bound = max(1, math.prod(n - 1 for n in shape)) * max_volume
        assert definition.spectrum_length(I) > bound, shape


def test_no_bin_is_dropped_3d():
    """Mass conservation catches any out-of-range sigma being silently skipped.

    Kept to 2x2x2 and 2x2x3: cost grows as pixels^4.
    """
    rng = np.random.default_rng(37)
    for shape in [(2, 2, 2), (2, 2, 3)]:
        I = rng.integers(1, 3, size=shape).astype(np.int64)
        total = sum(int(I[v]) for v in np.ndindex(shape))
        n_pixels = I.size
        assert sum(AREA_SPECTRUM(I)) == total ** (len(shape) + 1)
        assert sum(ROW_SUMS(I)) == n_pixels * (len(shape) + 1) * total ** len(shape)


def test_all_one_image_has_no_dead_bins_below_support():
    """The all-one image populates every bin in 0..(W-1)(H-1)."""
    for shape in [(2, 2), (3, 3), (4, 4), (5, 4), (8, 5), (6, 6)]:
        A = AREA_SPECTRUM(np.ones(shape, dtype=np.uint8))
        natural = (shape[0] - 1) * (shape[1] - 1)
        assert all(x > 0 for x in A[:natural + 1]), shape
        R = ROW_SUMS(np.ones(shape, dtype=np.uint8))
        assert all(x > 0 for x in R[:natural + 1]), shape


# ---------------------------------------------------------------------------
# jacobian_transpose
# ---------------------------------------------------------------------------

def test_jacobian_transpose_sums_to_dot_with_row_sums():
    """The two orientations meet at the total: sum_p (J^T w)[p] = sum_k w[k] S[k]."""
    rng = np.random.default_rng(41)
    for shape in [(2, 2), (3, 3), (3, 2), (2, 2, 2)]:
        I = rng.integers(0, 4, size=shape).astype(np.int64)
        w = [float(x) for x in rng.integers(-4, 5, size=definition.spectrum_length(I))]
        got = definition.jacobian_transpose(I, w)
        assert math.isclose(sum(got), sum(a * b for a, b in zip(w, ROW_SUMS(I))))


def test_jacobian_transpose_matches_reference():
    """result[p] = sum_k J[k][p] w[k], checked against the reference rows."""
    rng = np.random.default_rng(43)
    for shape in [(2, 2), (3, 3), (3, 2), (2, 2, 2)]:
        I = rng.integers(0, 4, size=shape).astype(np.int64)
        rows = _reference_rows(I)
        w = [float(x) for x in rng.integers(-4, 5, size=len(rows))]
        got = definition.jacobian_transpose(I, w)
        for i, p in enumerate(np.ndindex(shape)):
            expected = sum(row.get(p, 0) * wk for row, wk in zip(rows, w))
            assert math.isclose(got[i], expected, rel_tol=1e-12, abs_tol=1e-9), (
                shape, p, got[i], expected,
            )


def test_jacobian_transpose_stays_exact_for_integral_weights():
    """Integer weights must not pass through a float, or huge rows lose precision."""
    I = np.full((6, 6), 4, dtype=np.int64)
    result = definition.jacobian_transpose(I, [3] * definition.spectrum_length(I))
    assert all(isinstance(x, int) for x in result), [type(x) for x in result]
    plain = definition.jacobian_transpose(I, [1] * definition.spectrum_length(I))
    assert all(x == 3 * y for x, y in zip(result, plain))


def test_jacobian_transpose_length():
    for shape in [(2, 2), (3, 3), (4, 4), (2, 2, 2)]:
        I = np.ones(shape, dtype=np.int64)
        got = definition.jacobian_transpose(I, [1] * definition.spectrum_length(I))
        assert len(got) == I.size


# ---------------------------------------------------------------------------
# normalized_spectrum_gradient
# ---------------------------------------------------------------------------

def _random_delta(rng, I, seed):
    return [float(x) for x in rng.normal(size=definition.spectrum_length(I))]


def test_gradient_matches_finite_difference():
    """Central differences of the float objective, in 2d and 3d.

    Differencing area_spectrum itself would be meaningless: its int() cast
    makes it a step function whose differences under a 1e-5 step vanish.
    """
    rng = np.random.default_rng(47)
    for shape in [(3, 3), (4, 3), (2, 2, 2)]:
        I = rng.integers(1, 5, size=shape).astype(np.int64)
        scale = ROW_SUMS(I)
        A = _spectrum_float(I)
        delta = _random_delta(rng, I, 47)
        target = [a + d * s for a, d, s in zip(A, delta, scale)]
        grad = GRADIENT(I, delta)
        h = 1e-5
        for i, p in enumerate(np.ndindex(shape)):
            up = I.astype(float)
            up[p] += h
            down = I.astype(float)
            down[p] -= h
            fd = (
                _normalized_loss(up, scale, target)
                - _normalized_loss(down, scale, target)
            ) / (2 * h)
            assert math.isclose(fd, grad[i], rel_tol=1e-6, abs_tol=1e-6), (
                shape, p, fd, grad[i],
            )


def test_gradient_is_zero_for_zero_delta():
    rng = np.random.default_rng(53)
    for shape in [(3, 3), (4, 4), (2, 2, 2)]:
        I = rng.integers(1, 5, size=shape).astype(np.int64)
        n = definition.spectrum_length(I)
        assert GRADIENT(I, [0.0] * n) == [0.0] * I.size
        assert GRADIENT(I, {}) == [0.0] * I.size


def test_gradient_on_zero_image_is_zero_not_nan():
    """Every bin has scale 0 here, so the naive division is 0/0."""
    for shape in [(3, 3), (4, 4), (2, 2, 2)]:
        I = np.zeros(shape, dtype=np.int64)
        n = definition.spectrum_length(I)
        assert ROW_SUMS(I) == [0] * n
        grad = GRADIENT(I, [1.0] * n)
        assert all(x == 0.0 for x in grad), shape
        assert not any(math.isnan(x) for x in grad), shape


def test_gradient_ignores_dead_bins():
    """Bins above the natural support are unreachable, so edits there do nothing."""
    I = np.ones((4, 4), dtype=np.int64)
    natural = (4 - 1) * (4 - 1)
    for k in range(natural + 1, definition.spectrum_length(I)):
        assert ROW_SUMS(I)[k] == 0
        assert GRADIENT(I, {k: 5.0}) == [0.0] * I.size


def test_gradient_dead_bin_edits_do_not_disturb_live_bins():
    I = np.ones((4, 4), dtype=np.int64)
    natural = (4 - 1) * (4 - 1)
    delta = {k: 1.0 for k in range(natural + 1)}
    mixed = dict(delta)
    mixed[definition.spectrum_length(I) - 1] = 100.0
    assert GRADIENT(I, mixed) == GRADIENT(I, delta)


def test_gradient_length_matches_image():
    for shape in [(2, 2), (3, 3), (4, 4), (2, 2, 2)]:
        I = np.ones(shape, dtype=np.int64)
        n = definition.spectrum_length(I)
        assert len(GRADIENT(I, [1.0] * n)) == I.size


def test_gradient_scale_invariant():
    """J has degree d, scale has degree d, so the 1/scale weight cancels it.

    This is the whole point of normalizing by the row sums: the direction
    should not depend on how bright the image is.
    """
    rng = np.random.default_rng(59)
    for shape in [(3, 3), (4, 4), (2, 2, 2)]:
        I = rng.integers(1, 5, size=shape).astype(np.int64)
        delta = _random_delta(rng, I, 59)
        base = GRADIENT(I, delta)
        for c in (2, 5):
            scaled = GRADIENT(c * I, delta)
            for a, b in zip(base, scaled):
                assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (shape, c, a, b)


def test_gradient_rejects_out_of_range_bin():
    I = np.ones((3, 3), dtype=np.int64)
    with pytest.raises(IndexError):
        GRADIENT(I, {definition.spectrum_length(I): 1.0})
    with pytest.raises(IndexError):
        GRADIENT(I, [(-1, 1.0)])


def test_gradient_sparse_forms_agree():
    rng = np.random.default_rng(61)
    I = rng.integers(1, 5, size=(3, 3)).astype(np.int64)
    n = definition.spectrum_length(I)
    delta = _random_delta(rng, I, 61)
    full = GRADIENT(I, delta)
    assert GRADIENT(I, {k: v for k, v in enumerate(delta) if v}) == full
    assert GRADIENT(I, list(enumerate(delta))) == full


def test_gradient_descent_decreases_loss():
    """Stepping against the gradient must lower the objective."""
    rng = np.random.default_rng(67)
    for shape in [(3, 3), (4, 4)]:
        I = rng.integers(1, 5, size=shape).astype(np.int64)
        scale = ROW_SUMS(I)
        A = _spectrum_float(I)
        delta = _random_delta(rng, I, 67)
        target = [a + d * s for a, d, s in zip(A, delta, scale)]
        grad = GRADIENT(I, delta)
        step = 0.01 / max(abs(g) for g in grad)
        assert _normalized_loss(I.astype(float), scale, target) > _normalized_loss(
            I.astype(float) - step * np.array(grad).reshape(shape), scale, target
        )


def test_gradient_matches_reference_jacobian():
    """Independent check: grad = -2 * J^T (delta / scale)."""
    rng = np.random.default_rng(71)
    for shape in [(3, 3), (3, 2), (2, 2, 2)]:
        I = rng.integers(1, 5, size=shape).astype(np.int64)
        scale = ROW_SUMS(I)
        delta = _random_delta(rng, I, 71)
        rows = _reference_rows(I)
        expected = [
            -2.0 * sum(
                row.get(p, 0) * d / s for row, d, s in zip(rows, delta, scale) if s > 0
            )
            for p in np.ndindex(shape)
        ]
        got = GRADIENT(I, delta)
        for a, b in zip(got, expected):
            assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9), (shape, a, b)
