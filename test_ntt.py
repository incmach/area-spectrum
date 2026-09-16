"""
Tests for ntt.py, cross-checking the NTT-based implementation against
the exact, non-optimized ones in definition.py.

Covers find_ntt_prime() / choose_ntt_prime() / choose_ntt_crt_primes(),
crt_reconstruct(), area_spectrum_ntt(), area_spectrum_ntt_crt(),
area_spectrum_jacobian_row_squared_norms(), scaled_objective_ntt() and
area_spectrum_scaled_gradient_ntt().

The single-prime variants are checked against definition.py; the CRT
variant (area_spectrum_ntt_crt) is checked against the already-tested
single-prime area_spectrum_ntt.

Runnable directly (python test_ntt.py) or via pytest.
All images are small (<= 8x8); the 3D case is expensive (about 30s)
because of the per-prefix NTT and the naive pattern enumeration.
The 3D check is factored into a single test so it runs only once.
"""

import math

import numpy as np

import definition
import ntt

AREA_SPECTRUM_NTT = ntt.area_spectrum_ntt
AREA_SPECTRUM_NTT_CRT = ntt.area_spectrum_ntt_crt
ROW_NORMS_NTT = ntt.area_spectrum_jacobian_row_squared_norms
SCALED_OBJECTIVE_NTT = ntt.scaled_objective_ntt
SCALED_GRADIENT_NTT = ntt.area_spectrum_scaled_gradient_ntt

AREA_SPECTRUM = definition.area_spectrum
ROW_NORMS = definition.area_spectrum_jacobian_row_squared_norms
TARGET_SQUARED_NORM = definition.target_function_squared_norm
SCALED_GRADIENT = definition.area_spectrum_scaled_gradient


# ---------------------------------------------------------------------------
# prime selection
# ---------------------------------------------------------------------------

def test_find_ntt_prime_properties():
    for min_value in [0, 1, 100, 66_000_000]:
        for sizes in [[4], [8, 4], [16, 16], [2, 2, 2]]:
            p = ntt.find_ntt_prime(min_value, sizes)
            assert p > min_value
            for s in sizes:
                assert (p - 1) % s == 0


def test_choose_ntt_prime_properties():
    for shape in [(2, 2), (3, 2), (3, 3), (2, 2, 2)]:
        I = np.ones(shape, dtype=np.uint8)
        p, ntt_shape = ntt.choose_ntt_prime(I)
        expected_shape = tuple(
            1 << max(0, 2 * n - 2).bit_length() for n in shape
        )
        assert ntt_shape == expected_shape
        assert p > I.size * (255 ** (len(shape) + 1))
        for s in ntt_shape:
            assert (p - 1) % s == 0


def test_choose_ntt_prime_uses_original_dtype():
    # The prime must be chosen from the original (uint8) dtype, not from an
    # int64 view that would blow the bound up to ~2**193.
    I = np.ones((2, 2), dtype=np.uint8)
    p8, shape = ntt.choose_ntt_prime(I)
    p64, _ = ntt.choose_ntt_prime(np.asarray(I, dtype=np.int64))
    assert p8 < 1e9
    assert p64 > 1e50


def test_choose_ntt_crt_primes_properties():
    for shape in [(2, 2), (3, 2), (3, 3)]:
        I = np.ones(shape, dtype=np.uint8)
        primes, ntt_shape = ntt.choose_ntt_crt_primes(I)
        expected_shape = tuple(
            1 << max(0, 2 * n - 2).bit_length() for n in shape
        )
        assert ntt_shape == expected_shape
        bound = I.size * (255 ** (len(shape) + 1))
        assert primes == sorted(set(primes))  # distinct, ascending
        assert math.prod(primes) > bound  # product covers the bound
        assert math.prod(primes[:-1]) <= bound  # smallest possible set
        for p in primes:
            assert p > 1
            for s in ntt_shape:
                assert (p - 1) % s == 0


def test_crt_reconstruct_recovers_small_integers():
    moduli = [5, 7, 11, 13]
    for x in range(100):
        residues = [np.array([x % m]) for m in moduli]
        assert int(ntt.crt_reconstruct(residues, moduli)[0]) == x


def test_crt_reconstruct_rejects_out_of_range():
    # A value >= product of the moduli cannot be recovered uniquely.
    residues = [np.array([1]), np.array([1])]
    assert int(ntt.crt_reconstruct(residues, [2, 3])[0]) == 1


# ---------------------------------------------------------------------------
# area_spectrum_ntt()
# ---------------------------------------------------------------------------

def test_area_spectrum_ntt_ones_known_values():
    assert AREA_SPECTRUM_NTT(np.ones((1, 1), dtype=np.uint8)) == [1]
    assert AREA_SPECTRUM_NTT(np.ones((2, 1), dtype=np.uint8)) == [8, 0]
    assert AREA_SPECTRUM_NTT(np.ones((1, 2), dtype=np.uint8)) == [8, 0]
    assert AREA_SPECTRUM_NTT(np.ones((2, 2), dtype=np.uint8)) == [40, 24, 0, 0]
    assert AREA_SPECTRUM_NTT(np.ones((3, 2), dtype=np.uint8)) == [
        108, 72, 36, 0, 0, 0,
    ]
    assert AREA_SPECTRUM_NTT(np.ones((3, 3), dtype=np.uint8)) == [
        273, 192, 192, 24, 48, 0, 0, 0, 0,
    ]


def test_area_spectrum_ntt_zero_image():
    I = np.zeros((3, 3), dtype=np.uint8)
    assert AREA_SPECTRUM_NTT(I) == AREA_SPECTRUM(I)


def test_area_spectrum_ntt_matches_definition_2d():
    rng = np.random.default_rng(0)
    for shape in [(2, 2), (3, 2), (2, 3), (3, 3), (4, 4)]:
        I = rng.integers(0, 5, size=shape).astype(np.uint8)
        assert AREA_SPECTRUM_NTT(I) == AREA_SPECTRUM(I), shape


def test_area_spectrum_ntt_matches_definition_1d():
    rng = np.random.default_rng(1)
    for shape in [(4,), (1, 5), (5, 1)]:
        I = rng.integers(0, 5, size=shape).astype(np.uint8)
        assert AREA_SPECTRUM_NTT(I) == AREA_SPECTRUM(I), shape


def test_area_spectrum_ntt_explicit_prime_matches_default():
    rng = np.random.default_rng(2)
    I = rng.integers(0, 5, size=(3, 3)).astype(np.uint8)
    p, _ = ntt.choose_ntt_prime(I)
    assert AREA_SPECTRUM_NTT(I, prime=p) == AREA_SPECTRUM_NTT(I)
    assert AREA_SPECTRUM_NTT(I, prime=p) == AREA_SPECTRUM(I)


def test_area_spectrum_ntt_crt_matches_single_prime():
    rng = np.random.default_rng(10)
    for shape in [(4,), (2, 2), (3, 2), (2, 3), (3, 3)]:
        I = rng.integers(0, 5, size=shape).astype(np.uint8)
        assert AREA_SPECTRUM_NTT_CRT(I) == AREA_SPECTRUM_NTT(I), shape


def test_area_spectrum_ntt_crt_ones_known_values():
    assert AREA_SPECTRUM_NTT_CRT(np.ones((2, 2), dtype=np.uint8)) == [40, 24, 0, 0]
    assert AREA_SPECTRUM_NTT_CRT(np.ones((3, 3), dtype=np.uint8)) == [
        273, 192, 192, 24, 48, 0, 0, 0, 0,
    ]


# ---------------------------------------------------------------------------
# Jacobian row squared norms
# ---------------------------------------------------------------------------

def test_row_norms_ntt_matches_definition():
    rng = np.random.default_rng(3)
    for shape in [(2, 2), (3, 2), (2, 3), (3, 3)]:
        I = rng.integers(0, 5, size=shape).astype(np.uint8)
        assert ROW_NORMS_NTT(I) == ROW_NORMS(I), shape


def test_row_norms_ntt_ones_known_values():
    assert ROW_NORMS_NTT(np.ones((2, 2), dtype=np.uint8)) == [3600, 1296, 0, 0]
    assert ROW_NORMS_NTT(np.ones((3, 2), dtype=np.uint8)) == [
        17496, 8208, 2376, 0, 0, 0,
    ]


# ---------------------------------------------------------------------------
# scaled objective
# ---------------------------------------------------------------------------

def test_scaled_objective_ntt_matches_definition():
    rng = np.random.default_rng(4)
    for shape in [(2, 2), (3, 2), (3, 3)]:
        I = rng.integers(0, 5, size=shape).astype(np.uint8)
        target = list(rng.integers(1, 10, size=I.size))
        assert math.isclose(
            SCALED_OBJECTIVE_NTT(I, target), TARGET_SQUARED_NORM(I, target)
        )


def test_scaled_objective_ntt_zero_at_target():
    rng = np.random.default_rng(5)
    I = rng.integers(1, 5, size=(3, 2)).astype(np.uint8)
    assert SCALED_OBJECTIVE_NTT(I, AREA_SPECTRUM(I)) == 0.0


# ---------------------------------------------------------------------------
# scaled gradient
# ---------------------------------------------------------------------------

def test_scaled_gradient_ntt_matches_definition():
    rng = np.random.default_rng(6)
    for shape in [(2, 2), (3, 2), (2, 3), (3, 3)]:
        I = rng.integers(0, 5, size=shape).astype(np.uint8)
        target = list(rng.integers(1, 10, size=I.size))
        got = np.asarray(SCALED_GRADIENT_NTT(I, target))
        expect = np.asarray(SCALED_GRADIENT(I, target))
        assert np.allclose(got, expect), shape


def test_scaled_gradient_ntt_zero_at_target():
    I = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    grad = SCALED_GRADIENT_NTT(I, AREA_SPECTRUM(I))
    assert np.allclose(grad, np.zeros_like(grad))


# ---------------------------------------------------------------------------
# 3D (expensive; single combined test)
# ---------------------------------------------------------------------------

def test_3d_matches_definition():
    rng = np.random.default_rng(7)
    I = rng.integers(0, 4, size=(2, 2, 2)).astype(np.uint8)
    target = list(rng.integers(1, 6, size=I.size))
    assert AREA_SPECTRUM_NTT(I) == AREA_SPECTRUM(I)
    assert ROW_NORMS_NTT(I) == ROW_NORMS(I)
    assert math.isclose(SCALED_OBJECTIVE_NTT(I, target), TARGET_SQUARED_NORM(I, target))
    assert np.allclose(
        np.asarray(SCALED_GRADIENT_NTT(I, target)),
        np.asarray(SCALED_GRADIENT(I, target)),
    )


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
    print("all ntt tests passed")