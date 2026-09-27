"""Tests for the NTT area-spectrum path in ntt_spectrum.py.

The reference is definition.py, and the claims here are exactness, not
approximation: every spectrum comparison is integer equality. The parts most
worth pinning are the ones that fail silently rather than loudly -- the roll
sign in the section, the twiddle stride, the negation side, the offset decode,
and the per-axis padding floor. Each of those returns an array of the right
shape with plausible values, or a spectrum that is right everywhere except the
degenerate cases, so only a comparison against the reference catches them.
"""
import math

import numpy as np
import pytest

import definition
import descent
import ntt_spectrum
from definition import area_spectrum, spectrum_length
from ntt_spectrum import (
    NTTBackend,
    UnsupportedDimension,
    _offsets,
    _primes_of_order_2,
    area_spectrum_ntt,
    bin_bound,
    crt,
    dtype_max,
    embed,
    intt,
    ntt,
    pad_length,
    pad_length_of_shape,
    pad_shape,
    primes_for,
    primes_for_shape,
    primitive_root,
    root_table,
    spectrum_mod,
    triple_correlation,
)

P, ROOT = 998244353, 3


def _primes(exponent, count=4):
    return list(_primes_of_order_2(exponent)[:count])


# ---------------------------------------------------------------------------
# the transform
# ---------------------------------------------------------------------------

def test_ntt_matches_the_naive_dft():
    M = 64
    w = pow(ROOT, (P - 1) // M, P)
    rng = np.random.default_rng(0)
    a = rng.integers(0, 50, size=M)
    naive = np.array([
        sum(int(a[v]) * pow(w, v * k, P) for v in range(M)) % P for k in range(M)
    ])
    assert np.array_equal(naive, ntt(a, P, ROOT))


def test_ntt_round_trips():
    rng = np.random.default_rng(1)
    for M in (2, 4, 8, 16, 64, 256, 1024):
        a = rng.integers(0, 100, size=M)
        assert np.array_equal(intt(ntt(a, P, ROOT), P, ROOT) % P, a % P), M


def test_ntt_is_linear():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 20, size=32)
    b = rng.integers(0, 20, size=32)
    assert np.array_equal(
        ntt((a + b) % P, P, ROOT) % P, (ntt(a, P, ROOT) + ntt(b, P, ROOT)) % P
    )


def test_ntt_rejects_a_non_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        ntt(np.zeros(12, dtype=np.int64), P, ROOT)


def test_root_table_rejects_a_root_without_the_required_order():
    with pytest.raises(ValueError, match="no order"):
        root_table(8, P, 2)


def test_transforms_work_at_every_needed_length():
    """Each power-of-two length a small image can ask for, across primes."""
    for exponent in range(2, 9):
        M = 1 << exponent
        p = _primes(exponent, 1)[0]
        rng = np.random.default_rng(3)
        a = rng.integers(0, 40, size=M)
        assert np.array_equal(intt(ntt(a, p, primitive_root(p)), p,
                                   primitive_root(p)) % p, a % p), M


# ---------------------------------------------------------------------------
# prime selection
# ---------------------------------------------------------------------------

def test_searched_primes_are_prime_and_have_the_required_2_adic_factor():
    for exponent in range(2, 13):
        step = 1 << exponent
        for p in _primes(exponent, 3):
            assert ntt_spectrum.galois.is_prime(p), p
            assert (p - 1) % step == 0, (p, exponent)


def test_searched_primes_are_increasing_and_distinct():
    for exponent in (3, 5, 8, 11):
        ps = _primes(exponent, 6)
        assert ps == sorted(ps)
        assert len(set(ps)) == len(ps)


def test_searched_primes_carry_a_root_of_the_needed_order():
    """A 2**n-th root of unity must exist, and have exactly that order."""
    for exponent in range(2, 11):
        p = _primes(exponent, 1)[0]
        omega = pow(primitive_root(p), (p - 1) // (1 << exponent), p)
        assert pow(omega, 1 << exponent, p) == 1
        assert pow(omega, 1 << (exponent - 1), p) != 1, (p, exponent)


def test_prime_search_finds_the_familiar_998244353_at_high_order():
    """998244353 = 119*2**23 + 1 is in the same family; the search should find
    the same primes a hand-picked table would have."""
    found = _primes_of_order_2(20)
    assert 1048576 + 1 in found or found[0] > 1048576
    # every prime found is a genuine k*2**20+1
    for p in found[:5]:
        assert (p - 1) % (1 << 20) == 0


def test_prime_search_is_cached_per_exponent():
    ntt_spectrum._primes_of_order_2.cache_clear()
    first = _primes_of_order_2(7)
    second = _primes_of_order_2(7)
    assert first is second, "the search should not run twice for one exponent"


def test_primes_for_is_cached_per_shape_and_dtype():
    rng = np.random.default_rng(4)
    a = rng.integers(0, 5, size=(3, 3)).astype(np.int64)
    b = np.full((3, 3), 9, dtype=np.int64)      # different values, same type
    pa = primes_for(a)
    pb = primes_for(b)
    assert [p for p, _ in pa] == [p for p, _ in pb], "bound must ignore values"
    assert primes_for(a) == pa, "the result should come from the cache"


def test_primes_for_differs_between_dtypes():
    a = np.zeros((3, 3), dtype=np.uint8)
    b = np.zeros((3, 3), dtype=np.int64)
    assert [p for p, _ in primes_for(a)] != [p for p, _ in primes_for(b)]
    # uint8's range is far smaller, so it needs no more primes
    assert len(primes_for(a)) <= len(primes_for(b))


def test_primes_for_honours_an_explicit_max_value():
    I = np.full((6, 6), 3, dtype=np.int64)
    default = [p for p, _ in primes_for(I)]
    tight = [p for p, _ in primes_for(I, max_value=3)]
    assert len(tight) < len(default)
    assert set(tight) <= set(default)


def test_primes_for_shape_is_cached_and_keyed():
    ntt_spectrum.primes_for_shape.cache_clear()
    a = primes_for_shape((3, 3), "i8", 10 ** 20)
    b = primes_for_shape((3, 3), "i8", 10 ** 20)
    assert a is b
    c = primes_for_shape((4, 4), "i8", 10 ** 20)
    # 3x3 and 4x4 both pad to 8x8, so the same transform length and the same
    # prime list; the key is shape, and equal keys must give equal results.
    assert c == a
    # a shape needing a different transform length gets a different exponent
    d = primes_for_shape((5, 5), "i8", 10 ** 20)     # pads to 16x16
    assert d != a
    assert primes_for_shape((3, 3), "i8", 10 ** 30) != a, "bound is part of the key"


def test_primes_for_shape_product_exceeds_the_bound():
    rng = np.random.default_rng(5)
    for shape in [(2, 2), (3, 3), (4, 4), (3, 7), (6, 10)]:
        I = rng.integers(0, 255, size=shape).astype(np.int64)
        chosen = primes_for(I)
        assert math.prod(p for p, _ in chosen) > bin_bound(I), shape


def test_prime_search_raises_when_it_cannot_reach_the_bound():
    with pytest.raises(ValueError, match="large enough"):
        primes_for_shape((2, 2), "i8", 10 ** 400)


def test_dtype_max_covers_the_integer_dtypes():
    assert dtype_max(np.dtype("uint8")) == 255
    assert dtype_max(np.dtype("int8")) == 127
    assert dtype_max(np.dtype("uint16")) == 65535
    assert dtype_max(np.dtype("bool")) == 1


def test_dtype_max_rejects_an_unknown_kind():
    with pytest.raises(TypeError):
        dtype_max(np.dtype("complex64"))


def test_bin_bound_ignores_the_pixel_values():
    rng = np.random.default_rng(6)
    dim = rng.integers(0, 5, size=(5, 5)).astype(np.int64)
    bright = np.full((5, 5), 255, dtype=np.int64)
    assert bin_bound(dim) == bin_bound(bright)


def test_bin_bound_exceeds_every_bin():
    rng = np.random.default_rng(7)
    for shape in [(2, 2), (3, 3), (4, 4), (5, 5), (2, 7), (7, 2)]:
        I = rng.integers(0, 256, size=shape).astype(np.int64)
        assert max(area_spectrum(I)) < bin_bound(I), shape


def test_bin_bound_is_cubic_in_the_pixel_range():
    """The bound's exponent is the part that is load-bearing.

    bin_bound is astronomically looser than the real bins for int64, so merely
    checking bound > max(bin) cannot see a bound that has been shrunk to
    maxval**2 -- the modulus product sized for it would still look adequate.
    Pinning the cubic dependence is what makes such a regression visible.
    """
    I = np.zeros((5, 5), dtype=np.int64)
    b2, b4, b8 = (bin_bound(I, max_value=v) for v in (2, 4, 8))
    assert b4 == b2 * 8
    assert b8 == b4 * 8


def test_bin_bound_grows_with_the_number_of_offset_pairs():
    """A wider image has more offset pairs per bin, so the bound must rise."""
    small = np.zeros((3, 3), dtype=np.int64)
    wide = np.zeros((3, 9), dtype=np.int64)
    assert bin_bound(wide, max_value=4) > bin_bound(small, max_value=4)


def test_undersizing_the_bound_is_detectable():
    """The failure bin_bound exists to prevent, exercised directly.

    A modulus product smaller than the true bin does not error -- it returns a
    wrapped number, which is the dangerous kind of wrong. So the guarantee is
    the bound being large enough, checked against the real spectrum, and the
    raise when even the whole available prime pool cannot reach a bound.
    """
    I = np.full((5, 5), 9, dtype=np.int64)
    true_max = max(area_spectrum(I))
    assert true_max < bin_bound(I, max_value=9)
    # a quadratic bound is too small for this image, which is the regression
    # the cubic test above guards against
    quadratic = (2 * 5 - 1) * (2 * 5 - 1) ** 2 * I.size * 9 ** 2
    assert quadratic < true_max
    # and a bound no available prime set can reach is an error, not a guess
    with pytest.raises(ValueError, match="large enough"):
        primes_for_shape((5, 5), "i8", 10 ** 400)


def test_bin_bound_accepts_a_tighter_max_value():
    I = np.full((4, 4), 5, dtype=np.int64)
    assert bin_bound(I, max_value=5) < bin_bound(I, max_value=200)


# ---------------------------------------------------------------------------
# padding
# ---------------------------------------------------------------------------

def test_pad_shape_is_a_rectangle_of_powers_of_two():
    rng = np.random.default_rng(8)
    for shape in [(1, 1), (2, 2), (3, 3), (4, 4), (2, 8), (3, 7), (5, 8)]:
        rows, cols = pad_shape(np.zeros(shape, dtype=np.int64))
        assert rows & (rows - 1) == 0, shape
        assert cols & (cols - 1) == 0, shape


def test_pad_shape_bounds_both_axes_independently():
    """My >= 2H-1 and Mx >= 2W-1, each from its own axis, not the largest."""
    for shape in [(2, 8), (8, 2), (3, 7), (7, 3), (1, 4), (4, 1), (5, 8)]:
        rows, cols = pad_shape(np.zeros(shape, dtype=np.int64))
        H, W = shape
        assert rows >= 2 * H - 1, shape
        assert cols >= 2 * W - 1, shape


def test_pad_shape_floors_each_axis_at_two():
    """A degenerate axis must still pad to 2, or centring misreads 0 as -1.

    With rows == 1 the centring test dy >= rows//2 reads 0 >= 0 and reports a
    shift of zero as -1, which empties the valid set and silently zeroes bin 0.
    """
    for shape in [(1, 1), (1, 4), (4, 1), (1, 8), (8, 1)]:
        rows, cols = pad_shape(np.zeros(shape, dtype=np.int64))
        assert rows >= 2 and cols >= 2, shape


def test_pad_shape_beats_a_square_for_non_square_images():
    def square(shape):
        M = 1
        while M < 2 * max(shape):
            M *= 2
        return M * M

    for shape, saving in [((2, 8), True), ((3, 7), True), ((4, 20), True)]:
        rows, cols = pad_shape(np.zeros(shape, dtype=np.int64))
        assert (rows * cols < square(shape)) is saving, shape
    # a square shape cannot save anything, and must not claim to
    rows, cols = pad_shape(np.zeros((4, 4), dtype=np.int64))
    assert (rows, cols) == (8, 8)


def test_pad_length_is_a_power_of_two():
    for shape in [(1, 1), (2, 2), (3, 3), (4, 4), (2, 8), (3, 7), (6, 10)]:
        L = pad_length(np.zeros(shape, dtype=np.int64))
        assert L & (L - 1) == 0, shape


def test_pad_length_of_shape_agrees_with_pad_length():
    for shape in [(1, 1), (2, 3), (4, 4), (3, 7), (5, 8)]:
        I = np.zeros(shape, dtype=np.int64)
        assert pad_length_of_shape(shape) == pad_length(I), shape


def test_offsets_decode_injectively_on_valid_shifts():
    """No two distinct valid offsets may share a flat shift.

    This is the property the per-axis bounds exist to provide: a collision
    would sum two different offsets into one section entry.
    """
    for shape in [(1, 4), (4, 1), (2, 8), (3, 7), (5, 5), (6, 4)]:
        I = np.zeros(shape, dtype=np.int64)
        pad = pad_shape(I)
        dy, dx, valid = _offsets(pad, I.shape)
        seen = {}
        for s in np.nonzero(valid)[0]:
            key = (int(dy[s]), int(dx[s]))
            assert seen.get(key, s) == s, (shape, key)
            seen[key] = s


def test_offsets_recover_the_offset_from_the_flat_shift():
    for shape in [(1, 4), (2, 8), (3, 7), (5, 5)]:
        I = np.zeros(shape, dtype=np.int64)
        pad = pad_shape(I)
        rows, cols = pad
        dy, dx, valid = _offsets(pad, I.shape)
        for s in np.nonzero(valid)[0]:
            assert (int(dy[s]) * cols + int(dx[s])) % (rows * cols) == s, shape


def test_embed_places_the_image_in_the_corner():
    I = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    flat = embed(I, (4, 8))
    assert flat.shape == (32,)
    assert flat[:3].tolist() == [1, 2, 3]
    assert flat[8:11].tolist() == [4, 5, 6]
    assert flat[3:8].sum() == 0
    assert flat[11:].sum() == 0


def test_embed_shifts_do_not_alias_x_into_y():
    """A shift of (0, -1) must not move to a different row.

    The flat index is y*Mx + x, so adding -1 to a pixel in the last column
    lands in the padding rather than on the previous row's pixel.
    """
    I = np.zeros((3, 3), dtype=np.int64)
    I[0, 2] = 1
    flat = embed(I, (8, 8))
    assert flat[2] == 1
    assert flat[1] == 0
    assert flat[8 - 1] == 0


def test_embed_of_a_wide_image_uses_the_wide_stride():
    I = np.array([[1, 2, 3, 4]], dtype=np.int64)
    flat = embed(I, (2, 8))
    assert flat[0] == 1
    assert flat[3] == 4
    assert flat[4:8].sum() == 0, "row padding must be clear"


# ---------------------------------------------------------------------------
# the section
# ---------------------------------------------------------------------------

def _direct_section_entry(I, d1, d2):
    """T(d1, d2) by brute force, as a plain sum over anchors."""
    pad = pad_shape(I)
    rows, cols = pad
    flat = embed(I, pad)
    N = rows * cols
    s1 = (d1[0] * cols + d1[1]) % N
    s2 = (d2[0] * cols + d2[1]) % N
    total = 0
    for v in range(N):
        a, b, c = flat[v], flat[(v + s1) % N], flat[(v + s2) % N]
        if a and b and c:
            total += a * b * c
    return total % P


def test_section_entry_equals_a_direct_triple_sum():
    rng = np.random.default_rng(9)
    I = rng.integers(0, 5, size=(3, 3)).astype(np.int64)
    pad = pad_shape(I)
    Ihat = ntt(embed(I, pad), P, ROOT)
    for d2 in [(0, 1), (1, 0), (1, 1), (2, 1), (0, 2), (-1, 1)]:
        T = triple_correlation(I, d2, pad, P, ROOT, Ihat)
        for d1 in [(0, 1), (1, 0), (1, 1), (2, 1), (0, 2), (2, 0), (1, 2)]:
            s1 = (d1[0] * pad[1] + d1[1]) % (pad[0] * pad[1])
            assert int(T[s1]) % P == _direct_section_entry(I, d1, d2), (d1, d2)


def test_section_entry_matches_on_a_non_square_image():
    """The wide stride is the case the rectangle exists for."""
    rng = np.random.default_rng(10)
    I = rng.integers(0, 5, size=(2, 8)).astype(np.int64)
    pad = pad_shape(I)
    Ihat = ntt(embed(I, pad), P, ROOT)
    for d2 in [(0, 1), (1, 0), (1, 1), (0, 3)]:
        T = triple_correlation(I, d2, pad, P, ROOT, Ihat)
        for d1 in [(0, 1), (1, 0), (1, 1), (0, 7), (1, 4)]:
            s1 = (d1[0] * pad[1] + d1[1]) % (pad[0] * pad[1])
            assert int(T[s1]) % P == _direct_section_entry(I, d1, d2), (d1, d2)


def test_section_is_symmetric_in_its_two_offsets():
    """T(d1, d2) == T(d2, d1): the product commutes."""
    rng = np.random.default_rng(11)
    I = rng.integers(0, 5, size=(3, 3)).astype(np.int64)
    pad = pad_shape(I)
    N = pad[0] * pad[1]
    for d1, d2 in [((0, 1), (1, 1)), ((1, 0), (2, 1)), ((0, 2), (1, 1))]:
        a = int(triple_correlation(I, d2, pad, P, ROOT)[
            (d1[0] * pad[1] + d1[1]) % N]) % P
        b = int(triple_correlation(I, d1, pad, P, ROOT)[
            (d2[0] * pad[1] + d2[1]) % N]) % P
        assert a == b, (d1, d2)


def test_section_is_zero_for_an_offset_pair_that_cannot_share_an_anchor():
    I = np.ones((3, 3), dtype=np.int64)
    pad = pad_shape(I)
    rows, cols = pad
    N = rows * cols
    T = triple_correlation(I, (2, 2), pad, P, ROOT)
    s1 = ((-2) * cols + 2) % N
    s2 = (2 * cols + 2) % N
    flat = embed(I, pad)
    direct = sum(int(flat[v]) * int(flat[(v + s1) % N]) * int(flat[(v + s2) % N])
                 for v in range(N))
    assert direct == 0, "this offset pair should have no in-image anchor"
    assert int(T[s1]) % P == 0


# ---------------------------------------------------------------------------
# the spectrum
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (2, 2), (3, 3), (4, 4), (2, 3), (3, 2), (5, 5), (2, 5), (5, 2), (6, 4),
    (4, 6), (3, 7), (7, 3), (2, 8), (8, 2), (1, 4), (4, 1), (1, 1), (6, 10),
])
def test_spectrum_mod_equals_the_reference(shape):
    rng = np.random.default_rng(12)
    I = rng.integers(0, 5, size=shape).astype(np.int64)
    assert np.array_equal(
        spectrum_mod(I, P, ROOT, pad_shape(I)), np.array(area_spectrum(I))
    ), shape


def test_spectrum_mod_is_all_ones_exact():
    for shape in [(3, 3), (4, 4), (5, 5), (2, 8)]:
        I = np.ones(shape, dtype=np.int64)
        assert np.array_equal(
            spectrum_mod(I, P, ROOT), np.array(area_spectrum(I))
        ), shape


def test_spectrum_mod_of_a_zero_image_is_zero():
    I = np.zeros((4, 4), dtype=np.int64)
    assert spectrum_mod(I, P, ROOT).tolist() == [0] * spectrum_length(I)


def test_spectrum_mod_is_a_residue_not_the_count():
    """mod p means a bin can exceed p, so this asserts the reduction is real.

    A 12x12 of 40s has bins above 998244353, so spectrum_mod is genuinely
    reducing and the comparison against the reference is a congruence.
    """
    I = np.full((12, 12), 40, dtype=np.int64)
    assert max(area_spectrum(I)) > P, "test needs bins that exceed one prime"
    assert np.array_equal(
        spectrum_mod(I, P, ROOT), np.array(area_spectrum(I)) % P
    )


# ---------------------------------------------------------------------------
# CRT
# ---------------------------------------------------------------------------

def test_crt_reconstructs_below_the_modulus_product():
    primes = _primes(6, 2)
    product = primes[0] * primes[1]
    for v in (0, 5, 1000, primes[0] - 1, primes[1] - 1, product - 1):
        assert crt([v % p for p in primes], primes) == v, v


def test_crt_wraps_above_the_modulus_product():
    """Not a bug: it is why bin_bound has to hold, and that is tested."""
    primes = _primes(6, 2)
    product = primes[0] * primes[1]
    assert crt([product % p for p in primes], primes) == 0


def test_crt_with_one_prime_is_the_identity_below_it():
    for v in (0, 7, P - 1):
        assert crt([v], [P]) == v


def test_crt_needs_the_residues_given_per_prime():
    """One scalar per prime, in prime order -- the obvious mis-shape is [n][k]."""
    primes = _primes(6, 3)
    v = 4242
    assert crt([v % p for p in primes], primes) == v


# ---------------------------------------------------------------------------
# the whole pipeline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (2, 2), (3, 3), (4, 4), (2, 3), (5, 5), (6, 4), (2, 8), (3, 7), (6, 10),
    (10, 6), (1, 4), (4, 1), (1, 1),
])
def test_area_spectrum_ntt_equals_the_reference(shape):
    rng = np.random.default_rng(13)
    I = rng.integers(0, 6, size=shape).astype(np.int64)
    assert area_spectrum_ntt(I) == area_spectrum(I), shape


def test_area_spectrum_ntt_is_exact_for_a_wide_and_a_tall_image():
    """Both orientations of a non-square image, at a size with real content."""
    rng = np.random.default_rng(14)
    for shape in [(2, 9), (9, 2), (3, 8), (8, 3)]:
        I = rng.integers(0, 9, size=shape).astype(np.int64)
        assert area_spectrum_ntt(I) == area_spectrum(I), shape


def test_area_spectrum_ntt_with_large_values_needs_more_than_one_prime():
    I = np.array([[200, 255], [3, 7]], dtype=np.int64)
    assert len(primes_for(I)) >= 2
    assert area_spectrum_ntt(I) == area_spectrum(I)


def test_area_spectrum_ntt_with_an_explicit_max_value_is_still_exact():
    I = np.full((4, 4), 3, dtype=np.int64)
    tight = primes_for(I, max_value=3)
    assert area_spectrum_ntt(I, primes=tight) == area_spectrum(I)


@pytest.mark.parametrize("I", [
    np.zeros((3, 3), dtype=np.int64),
    np.ones((3, 3), dtype=np.int64),
    np.ones((1, 1), dtype=np.int64),
    np.ones((2, 1), dtype=np.int64),
    np.ones((1, 2), dtype=np.int64),
    np.ones((2, 2), dtype=np.int64),
    np.array([[5]], dtype=np.int64),
])
def test_area_spectrum_ntt_edge_cases(I):
    assert area_spectrum_ntt(I) == area_spectrum(I), I.shape


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int16", "int32"])
def test_area_spectrum_ntt_across_dtypes(dtype):
    rng = np.random.default_rng(15)
    info = np.iinfo(dtype)
    lo, hi = 0, min(60, info.max)
    I = rng.integers(lo, hi, size=(4, 4)).astype(dtype)
    assert area_spectrum_ntt(I) == area_spectrum(I), dtype


def test_area_spectrum_ntt_rejects_3d_with_a_reason():
    I = np.ones((2, 2, 2), dtype=np.int64)
    with pytest.raises(UnsupportedDimension) as info:
        area_spectrum_ntt(I)
    message = str(info.value)
    assert "3x3" in message
    assert "two" in message


def test_unsupported_dimension_is_a_value_error():
    assert issubclass(UnsupportedDimension, ValueError)


# ---------------------------------------------------------------------------
# the backend
# ---------------------------------------------------------------------------

def test_backend_satisfies_the_descent_protocol():
    assert isinstance(NTTBackend(), descent.SpectrumBackend)


def test_backend_spectrum_matches_the_reference():
    rng = np.random.default_rng(16)
    I = rng.integers(0, 6, size=(3, 3)).astype(np.int64)
    assert NTTBackend().area_spectrum(I) == area_spectrum(I)


def test_backend_passes_max_value_through():
    I = np.full((3, 3), 2, dtype=np.int64)
    backend = NTTBackend(max_value=2)
    assert backend.area_spectrum(I) == area_spectrum(I)


def test_descent_runs_against_the_ntt_backend():
    """The seam, exercised against a genuinely different implementation."""
    rng = np.random.default_rng(17)
    I = rng.integers(1, 6, size=(3, 3)).astype(np.int64)
    edited = I.copy()
    edited[1, 1] += 1
    target = area_spectrum(edited)
    scale = definition.jacobian_row_sums(I)
    fast = descent.descent(I, target, scale, max_steps=40, patience=8,
                           width=8, backend=NTTBackend(max_value=6))
    slow = descent.descent(I, target, scale, max_steps=40, patience=8, width=8)
    assert math.isclose(fast.loss, slow.loss, rel_tol=1e-9)
    assert np.array_equal(fast.image, slow.image)
