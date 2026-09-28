"""Tests for the NTT area-spectrum path in ntt_spectrum.py.

The reference is definition.py, and the claims here are exactness, not
approximation: every spectrum comparison is integer equality. The parts most
worth pinning are the ones that fail silently rather than loudly -- the roll
sign in the section, the twiddle stride, the negation side, the offset decode,
and the per-axis padding floor. Each of those returns an array of the right
shape with plausible values, or a spectrum that is right everywhere except the
degenerate cases, so only a comparison against the reference catches them.
"""
import itertools
import math
from collections import Counter

import numpy as np
import pytest

import definition
import descent
import ntt_spectrum
from definition import area_spectrum, spectrum_length
from ntt_spectrum import (
    NTTBackend,
    UnsupportedDimension,
    _half_offsets,
    _masked_copy,
    _offsets,
    _primes_of_order_2,
    area_spectrum_ntt,
    bin_bound,
    crt,
    dtype_max,
    embed,
    intt,
    jacobian_row_sums_ntt,
    ntt,
    pad_length,
    pad_length_of_shape,
    pad_shape,
    primes_for,
    primes_for_row_sums,
    primes_for_shape,
    primitive_root,
    root_table,
    row_sum_bound,
    row_sums_mod,
    spectrum_gradient_ntt,
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


def test_selected_primes_are_under_the_int64_safe_cap():
    """p**2 must fit an int64 or the butterfly wraps and returns wrong numbers
    without raising, so every modulus the module can pick is checked against
    the cap that keeps the product inside int64."""
    rng = np.random.default_rng(11)
    for shape in [(2, 2), (3, 3), (5, 5), (8, 8)]:
        for mv in (3, 255, 2 ** 63 - 1):
            I = rng.integers(0, max(mv, 1) + 1, size=shape).astype(np.int64)
            for ps in (primes_for(I, max_value=mv),
                       primes_for_row_sums(I, max_value=mv)):
                for p, _ in ps:
                    assert p <= ntt_spectrum.PRIME_CAP, (p, shape)
                    assert p * p < 2 ** 63, ("p**2 would overflow int64", p)


def test_selection_minimises_the_number_of_primes():
    """Every prime is one pass over the sections, and a pass costs the same at
    any modulus, so the count is what matters. The minimum is the smallest k
    whose k largest eligible primes clear the bound, and dropping one prime
    must fall short -- otherwise the set is doing more work than asked."""
    rng = np.random.default_rng(12)
    for shape in [(3, 3), (6, 6), (8, 8)]:
        for mv in (3, 255, 2 ** 63 - 1):
            I = rng.integers(0, max(mv, 1) + 1, size=shape).astype(np.int64)
            bound = ntt_spectrum.bin_bound(I, mv)
            chosen = ntt_spectrum.select_primes(
                ntt_spectrum.pad_length_of_shape(shape).bit_length() - 1, bound
            )
            assert math.prod(chosen) > bound, "product must clear the bound"
            if len(chosen) > 1:
                one_fewer = chosen[:-1]
                assert math.prod(one_fewer) <= bound, (
                    "the last prime is not needed; the count is not minimal"
                )


def test_selected_primes_are_valid_moduli_for_the_transform():
    """A selected prime must actually carry a 2**exponent-th root of unity and
    be prime, or the transform has no twiddle table to build."""
    for exponent in range(2, 13):
        for p in ntt_spectrum.select_primes(exponent, 10 ** 30):
            assert ntt_spectrum.galois.is_prime(p), p
            assert (p - 1) % (1 << exponent) == 0, (p, exponent)
            omega = pow(ntt_spectrum.primitive_root(p), (p - 1) >> exponent, p)
            assert pow(omega, 1 << exponent, p) == 1
            assert pow(omega, 1 << (exponent - 1), p) != 1, (p, exponent)


def test_selection_is_ascending_distinct_and_cached():
    exponent = 8
    bound = 10 ** 20
    ntt_spectrum.select_primes.cache_clear()
    first = ntt_spectrum.select_primes(exponent, bound)
    second = ntt_spectrum.select_primes(exponent, bound)
    assert first is second, "the selection should not run twice for one key"
    assert list(first) == sorted(first)
    assert len(set(first)) == len(first)
    assert all(p <= ntt_spectrum.PRIME_CAP for p in first)


def test_selection_prefers_a_narrower_dtype_when_one_suffices():
    """Among equal-count sets the narrowest uint wins, so a small bound is
    served by uint8 or uint16 moduli rather than uint32 ones. At the 2**31 cap
    a large bound forces uint32 (there is no narrower way to reach it), but a
    small one does not, and the tie-break is what picks the small primes."""
    # a bound one uint8-clearing product wide: 193 * 257 is tiny, so the
    # selection has room to choose the narrow class
    tiny = ntt_spectrum.select_primes(6, 1)
    assert all(p < 2 ** 16 for p in tiny), tiny
    # a bound no uint8 or uint16 set can reach forces the wide class
    huge = ntt_spectrum.select_primes(6, 2 ** 200)
    assert max(huge) > 2 ** 16, huge


def test_largest_primes_scan_returns_ascending_largest_below_the_cap():
    found = ntt_spectrum._largest_primes_of_order_2(8, 1 << 20, 3)
    assert list(found) == sorted(found)
    assert len(found) == 3
    assert all(p < (1 << 20) for p in found)
    assert all((p - 1) % 256 == 0 for p in found)
    # asking for more than exist below the cap returns everything, not padding
    everything = ntt_spectrum._largest_primes_of_order_2(18, 1 << 20, 100)
    assert len(everything) < 100
    assert max(everything) < (1 << 20)


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
    # a bound past the product of every eligible prime under the cap is an
    # error, not a silently short set that would make CRT return residues.
    # 10**400 is now reachable (64 large primes clear it), so the bound has to
    # sit far past what a capped pool can supply.
    with pytest.raises(ValueError, match="reaches a product above"):
        primes_for_shape((2, 2), "i8", 1 << 40000)


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
    with pytest.raises(ValueError, match="reaches a product above"):
        primes_for_shape((5, 5), "i8", 1 << 40000)


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
# the halved offset loop
# ---------------------------------------------------------------------------

def _ordered_triple_oracle(I):
    """The spectrum by walking ordered triples, with no offsets involved.

    Independent of everything ntt_spectrum does: no padding, no flat shift, no
    correlation, no bin formula. Slow, so only for small images, but it is the
    statement of what the loop is supposed to compute, and the other tests in
    this file all reach the same quantity *through* those steps.
    """
    n = spectrum_length(I)
    H, W = I.shape
    out = np.zeros(n, dtype=np.int64)
    for a, b, c in itertools.permutations(
        [(y, x) for y in range(H) for x in range(W)], 3
    ):
        k = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if k < n:
            out[k] += int(I[a]) * int(I[b]) * int(I[c])
    return out


@pytest.mark.parametrize("shape", [
    (2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 7), (7, 2),
    (5, 5), (4, 6), (6, 4),
])
def test_spectrum_mod_equals_an_independent_ordered_triple_sum(shape):
    rng = np.random.default_rng(31)
    for hi in (2, 3, 7):
        I = rng.integers(0, hi, size=shape).astype(np.int64)
        assert np.array_equal(
            spectrum_mod(I, P, ROOT), _ordered_triple_oracle(I)
        ), (shape, hi)


@pytest.mark.parametrize("I", [
    np.zeros((3, 3), dtype=np.int64),            # no triangle at all
    np.ones((4, 4), dtype=np.int64),             # every triangle, all equal
    np.eye(5, dtype=np.int64),                   # every bin forced at once
    (np.arange(16).reshape(4, 4) % 2).astype(np.int64),
    np.full((3, 3), 5, dtype=np.int64),
])
def test_spectrum_mod_on_degenerate_images(I):
    """Zeros, all-ones, the identity: the cases a symmetry argument can break.

    Bin 0 in particular carries the collinear-but-distinct triangles, which the
    relabelling treats no differently from the rest, so a halving that is right
    in general can still be wrong there.
    """
    assert np.array_equal(spectrum_mod(I, P, ROOT), _ordered_triple_oracle(I))
    assert np.array_equal(spectrum_mod(I, P, ROOT), np.array(area_spectrum(I)))


@pytest.mark.parametrize("shape", [
    (2, 2), (3, 3), (4, 4), (2, 5), (5, 2), (3, 4), (4, 3), (6, 4), (7, 5),
    (8, 8), (2, 16), (16, 2),
])
def test_half_offsets_takes_exactly_one_of_d_and_minus_d(shape):
    """The transversal property the factor of 2 rests on.

    Any set with this property would do -- the weights are flat, so which member
    of each pair survives is irrelevant. Checking it directly is what pins the
    halving to a real symmetry rather than to a coincidence of the shapes
    tested above.
    """
    pad = pad_shape(np.zeros(shape, dtype=np.int64))
    dy, dx, valid = _offsets(pad, shape)
    half = _half_offsets(dy, dx, valid)
    rows, cols = pad
    for y, x in zip(dy, dx):
        s = (y * cols + x) % (rows * cols)
        if not valid[s] or s == 0:
            continue
        neg = (-y * cols - x) % (rows * cols)
        assert bool(half[s]) != bool(half[neg]), (shape, int(y), int(x))
    # exactly half, less the zero offset that neither half contains
    assert int(half.sum()) == (int(valid.sum()) - 1) // 2, shape


@pytest.mark.parametrize("shape", [
    (2, 2), (3, 3), (4, 4), (2, 5), (5, 2), (3, 4), (6, 4), (7, 5), (8, 8),
])
def test_spectrum_mod_equals_the_full_loop_with_one_weight(shape):
    """The halved loop, re-expanded by hand, against a loop over every offset.

    spectrum_mod now runs over half the offsets and multiplies by 2, which is
    only equal to the plain loop if the relabelling argument is right. This
    states the equality directly instead of resting on it, so a wrong factor
    (1, 3, 4) or a half that is not a transversal is caught as itself rather
    than only as a mismatch against definition.py.
    """
    rng = np.random.default_rng(32)
    I = rng.integers(0, 5, size=shape).astype(np.int64)
    pad = pad_shape(I)
    rows, cols = pad
    N = rows * cols
    n = spectrum_length(I)
    dy, dx, valid = _offsets(pad, I.shape)
    Ihat = ntt(embed(I, pad), P, ROOT)

    def loop(offsets, weight=1):
        A = np.zeros(n, dtype=np.int64)
        for s2 in offsets:
            s2 = int(s2)
            if s2 == 0:
                continue
            b2y, b2x = int(dy[s2]), int(dx[s2])
            bins = np.abs(dy * b2x - dx * b2y)
            keep = valid & (bins < n) & (np.arange(N) != s2)
            keep[0] = False
            T = triple_correlation(I, (b2y, b2x), pad, P, ROOT, Ihat)
            np.add.at(A, bins[keep], weight * T[keep])
            A %= P
        return A

    full = loop(np.nonzero(valid)[0])
    half_raw = loop(np.nonzero(_half_offsets(dy, dx, valid))[0])
    # the halved loop reproduces the full one only with the factor of 2 ...
    assert np.array_equal(spectrum_mod(I, P, ROOT, pad), full), shape
    assert np.array_equal(2 * half_raw % P, full), shape
    # ... and the factor is load-bearing rather than accidentally right, so a
    # dropped "2 *" or a doubled one is caught here as itself
    assert not np.array_equal(half_raw % P, full), shape


def test_the_relabelling_identity_the_halving_needs():
    """T(d1,d2) = T(d1-d2,-d2), the step that makes the weights flat.

    Re-anchoring a triangle at a different one of its three pixels has to leave
    the section alone, and that is what lets one flat factor cover all six
    relabellings. It is also what makes the orbits with fewer than six members
    harmless: if d1-d2 is out of range the right-hand side is a section at an
    out-of-range offset, and every such section is zero.

    Checked as a whole section rather than entry by entry, so it is the identity
    being pinned and not one offset pair that happens to agree.
    """
    rng = np.random.default_rng(33)
    I = rng.integers(1, 6, size=(5, 5)).astype(np.int64)
    pad = pad_shape(I)
    rows, cols = pad
    N = rows * cols
    dy, dx, valid = _offsets(pad, I.shape)
    idx = np.nonzero(valid)[0]
    for d2y, d2x in [(0, 1), (1, 0), (2, 1), (-1, 2), (0, -2), (2, -2)]:
        lhs = triple_correlation(I, (d2y, d2x), pad, P, ROOT)
        rhs = triple_correlation(I, (-d2y, -d2x), pad, P, ROOT)
        # the whole section: T(., d2) at every valid d1 against the
        # re-anchoring T(., -d2) at d1-d2, entry by entry
        s = (np.arange(N) - d2y * cols - d2x) % N
        assert np.array_equal(lhs[idx], rhs[s[idx]]), (d2y, d2x)


def test_the_row_sum_and_gradient_sections_are_not_relabelling_invariant():
    """Why the halving stops at the spectrum, as a test rather than a claim.

    The two neighbours share the loop shape but not the symmetry: the row-sum
    section is a correlation against a *masked* copy, so it depends on d1 and
    not on d2-d1 alone, and the gradient's on the difference alone. Both break
    the identity above, which is exactly why their loops stay whole. If a
    future change makes either of them invariant, this test fails and points at
    the halving as newly available.
    """
    rng = np.random.default_rng(34)
    I = rng.integers(1, 6, size=(5, 5)).astype(np.int64)
    pad = pad_shape(I)
    rows, cols = pad
    N = rows * cols
    neg = (-np.arange(N)) % N
    Ihat = ntt(embed(I, pad), P, ROOT)
    d1y, d1x, d2y, d2x = 1, 0, 0, 1
    # the spectrum section, which is invariant
    T = triple_correlation(I, (d2y, d2x), pad, P, ROOT, Ihat)
    # the row-sum section at d1 and at the re-anchoring -d1
    B = intt(ntt(_masked_copy(I, pad, d1y, d1x), P, ROOT)[neg] * Ihat % P,
             P, ROOT)
    B2 = intt(ntt(_masked_copy(I, pad, -d1y, -d1x), P, ROOT)[neg] * Ihat % P,
              P, ROOT)
    s1 = (d1y * cols + d1x) % N
    s2 = ((d1y - d2y) * cols + (d1x - d2x)) % N
    s2b = ((-d1y - d2y) * cols + (-d1x - d2x)) % N
    # T(., d2) at d1 against T(., -d2) at d1-d2, the re-anchoring
    T2 = triple_correlation(I, (-d2y, -d2x), pad, P, ROOT)
    assert int(T[s1]) == int(T2[s2]), "spectrum section lost relabelling symmetry"
    assert int(B[s2]) != int(B2[s2b]), \
        "row-sum section became relabelling-invariant"


def test_spectrum_mod_loops_over_half_as_many_offsets():
    """The point of the change, measured rather than argued."""
    import ntt_spectrum as ns

    calls = []
    real = ns.triple_correlation

    def counting(I, d2, pad, p, root, Ihat=None):
        calls.append(d2)
        return real(I, d2, pad, p, root, Ihat)

    for shape in [(4, 4), (6, 6), (3, 9)]:
        calls.clear()
        I = np.ones(shape, dtype=np.int64)
        pad = pad_shape(I)
        dy, dx, valid = _offsets(pad, shape)
        ns.triple_correlation = counting
        try:
            spectrum_mod(I, P, ROOT, pad)
        finally:
            ns.triple_correlation = real
        assert len(calls) == int(_half_offsets(dy, dx, valid).sum()), shape
        assert len(calls) == (int(valid.sum()) - 1) // 2, shape
        assert ns.spectrum_mod(I, P, ROOT, pad).tolist() == \
            np.array(area_spectrum(I)).tolist(), shape


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


# ---------------------------------------------------------------------------
# the row sums
# ---------------------------------------------------------------------------

def test_masked_copy_keeps_exactly_the_anchors_whose_partner_stays_inside():
    """The kernel behind the row sums: a window on the anchors, not a shift."""
    rng = np.random.default_rng(18)
    I = rng.integers(1, 9, size=(3, 4)).astype(np.int64)
    H, W = I.shape
    pad = pad_shape(I)
    for dy in range(-(H - 1), H):
        for dx in range(-(W - 1), W):
            got = _masked_copy(I, pad, dy, dx).reshape(pad)
            want = np.zeros(pad, dtype=np.int64)
            for y in range(H):
                for x in range(W):
                    y2, x2 = y + dy, x + dx
                    if 0 <= y2 < H and 0 <= x2 < W:
                        want[y, x] = I[y2, x2]
            assert np.array_equal(got, want), f"offset {(dy, dx)}"


def test_masked_copy_differs_from_the_product_that_would_be_the_spectrum():
    """I * shift(I, d) has three factors and is the triple correlation.

    Substituting it for the window returns the right shape and the wrong
    numbers, so the difference is worth pinning rather than assuming.
    """
    rng = np.random.default_rng(19)
    I = rng.integers(1, 9, size=(3, 3)).astype(np.int64)
    pad = pad_shape(I)
    flat = embed(I, pad)
    d = (1, 1)
    window = _masked_copy(I, pad, *d)
    product = flat * np.roll(flat, -(d[0] * pad[1] + d[1]))
    assert not np.array_equal(window, product)


def test_masked_copy_at_zero_is_the_embedded_image():
    I = np.arange(1, 10, dtype=np.int64).reshape(3, 3)
    pad = pad_shape(I)
    assert np.array_equal(_masked_copy(I, pad, 0, 0), embed(I, pad))


def test_masked_copy_on_a_wide_image_uses_the_wide_stride():
    """cols != W here, so a stride taken from the image would misalign."""
    I = np.arange(1, 15, dtype=np.int64).reshape(2, 7)
    pad = pad_shape(I)
    assert pad[1] > I.shape[1]
    got = _masked_copy(I, pad, 0, 1).reshape(pad)
    want = np.zeros(pad, dtype=np.int64)
    want[:2, :6] = I[:2, 1:7]
    assert np.array_equal(got, want)


@pytest.mark.parametrize("shape", [
    (2, 2), (3, 3), (4, 4), (2, 5), (5, 2), (3, 4), (4, 3), (6, 4),
])
def test_row_sums_mod_equals_the_reference_mod_p(shape):
    rng = np.random.default_rng(20)
    I = rng.integers(0, 7, size=shape).astype(np.int64)
    p, root = 998244353, 3
    assert np.array_equal(row_sums_mod(I, p, root),
                          np.array(definition.jacobian_row_sums(I)) % p)


@pytest.mark.parametrize("shape", [
    (2, 2), (3, 3), (4, 4), (2, 5), (5, 2), (3, 4), (4, 3), (6, 4),
])
def test_jacobian_row_sums_ntt_equals_the_reference_exactly(shape):
    """Integer equality: the row sums are counts, so a tolerance would let an
    off-by-three or a dropped pair pass."""
    rng = np.random.default_rng(21)
    I = rng.integers(0, 7, size=shape).astype(np.int64)
    assert jacobian_row_sums_ntt(I) == list(definition.jacobian_row_sums(I))


def test_row_sums_on_a_wide_and_a_tall_image():
    """Non-square, so the read-back stride and the mask both matter."""
    rng = np.random.default_rng(22)
    for shape in [(2, 7), (7, 2), (2, 16), (16, 2), (3, 9), (9, 3)]:
        I = rng.integers(0, 7, size=shape).astype(np.int64)
        assert jacobian_row_sums_ntt(I) == list(definition.jacobian_row_sums(I))


def test_row_sums_of_a_zero_image_are_zero():
    I = np.zeros((3, 3), dtype=np.int64)
    assert jacobian_row_sums_ntt(I) == list(definition.jacobian_row_sums(I))


def test_row_sums_of_a_single_lit_pixel_match_the_reference():
    I = np.zeros((3, 3), dtype=np.int64)
    I[1, 1] = 1
    assert jacobian_row_sums_ntt(I) == list(definition.jacobian_row_sums(I))


def test_row_sums_with_a_bounded_max_value_are_still_exact():
    I = np.full((3, 3), 7, dtype=np.int64)
    assert jacobian_row_sums_ntt(I, max_value=7) == list(
        definition.jacobian_row_sums(I))


def test_row_sums_need_more_than_one_prime_for_bright_pixels():
    """A single prime would return a residue rather than the count."""
    I = np.full((3, 3), 255, dtype=np.int64)
    assert len(primes_for_row_sums(I)) > 1


def test_row_sums_reject_3d_with_a_reason():
    with pytest.raises(UnsupportedDimension, match="2D"):
        jacobian_row_sums_ntt(np.ones((2, 2, 2), dtype=np.int64))


def test_row_sum_bound_exceeds_every_row_sum():
    rng = np.random.default_rng(23)
    for shape in [(2, 2), (3, 3), (4, 4), (2, 5), (3, 4)]:
        for I in (np.full(shape, 255, dtype=np.int64),
                  rng.integers(0, 255, size=shape).astype(np.int64)):
            assert max(definition.jacobian_row_sums(I)) <= row_sum_bound(I)


def test_row_sum_bound_counts_all_three_pairs_of_a_triple():
    """Three, not one: a row sum collects I[a]I[b] + I[a]I[c] + I[b]I[c]."""
    I = np.zeros((3, 3), dtype=np.int64)
    pairs = ((2 * 3 - 1) * (2 * 3 - 1)) ** 2
    assert row_sum_bound(I, max_value=1) == 3 * pairs * I.size


def test_row_sum_bound_is_quadratic_in_the_pixel_range():
    """Two factors, not three: the row sums drop the third power."""
    I = np.zeros((3, 3), dtype=np.int64)
    assert (row_sum_bound(I, max_value=100)
            == 100 * row_sum_bound(I, max_value=10))


def test_row_sum_bound_grows_with_the_number_of_offset_pairs():
    small = row_sum_bound(np.zeros((3, 3), dtype=np.int64), max_value=9)
    large = row_sum_bound(np.zeros((4, 4), dtype=np.int64), max_value=9)
    assert large > small


def test_row_sum_bound_ignores_the_pixel_values():
    I = np.zeros((3, 3), dtype=np.int64)
    assert row_sum_bound(I) == row_sum_bound(I, max_value=None)


def test_undersizing_the_row_sum_bound_is_detectable():
    """One prime short of the bound gives a plausible wrong answer, so the
    bound has to be an upper bound rather than an estimate.

    The check is against the modulus *product*, not the smallest prime: the
    selected moduli are now the largest under the cap, so a single one dwarfs
    the true entries and comparing against it would pass vacuously. What has to
    hold for exactness is that the product of the chosen primes exceeds the
    largest true row sum, and that dropping one prime would not.
    """
    I = np.full((3, 3), 4, dtype=np.int64)
    exact = jacobian_row_sums_ntt(I, max_value=4)
    ps = [p for p, _ in primes_for_row_sums(I, max_value=4)]
    product = math.prod(ps)
    assert max(exact) < product, "the product must cover the largest entry"
    # one prime fewer must fall short, or the bound is looser than it needs to be
    assert max(exact) > math.prod(ps[:-1]), "the last prime is load-bearing"


# ---------------------------------------------------------------------------
# the gradient
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (2, 2), (3, 3), (4, 4), (2, 5), (5, 2), (3, 4), (4, 3), (6, 4),
])
def test_spectrum_gradient_ntt_matches_the_reference(shape):
    """A tolerance, unlike the spectrum: the residual weights are floats."""
    rng = np.random.default_rng(24)
    I = rng.integers(1, 7, size=shape).astype(np.int64)
    target = area_spectrum(rng.integers(1, 7, size=shape).astype(np.int64))
    scale = definition.jacobian_row_sums(I)
    got = np.array(spectrum_gradient_ntt(I, target, scale))
    want = np.array(definition.spectrum_gradient(I, target, scale))
    assert got.shape == want.shape == (I.size,)
    assert np.allclose(got, want, rtol=1e-9, atol=1e-7)


def test_spectrum_gradient_on_a_wide_and_a_tall_image():
    """Non-square, so the read-back has to use the padded stride."""
    rng = np.random.default_rng(25)
    for shape in [(2, 7), (7, 2), (2, 16), (16, 2), (3, 9), (9, 3)]:
        I = rng.integers(1, 7, size=shape).astype(np.int64)
        target = area_spectrum(rng.integers(1, 7, size=shape).astype(np.int64))
        scale = definition.jacobian_row_sums(I)
        got = np.array(spectrum_gradient_ntt(I, target, scale))
        want = np.array(definition.spectrum_gradient(I, target, scale))
        assert np.allclose(got, want, rtol=1e-9, atol=1e-7)


def test_spectrum_gradient_vanishes_against_its_own_spectrum():
    rng = np.random.default_rng(26)
    I = rng.integers(1, 7, size=(3, 3)).astype(np.int64)
    spec = area_spectrum(I)
    scale = definition.jacobian_row_sums(I)
    got = np.array(spectrum_gradient_ntt(I, spec, scale))
    assert np.allclose(got, 0.0, atol=1e-9)


def test_spectrum_gradient_of_a_zero_image_is_zero():
    I = np.zeros((3, 3), dtype=np.int64)
    target = area_spectrum(np.ones((3, 3), dtype=np.int64))
    got = np.array(spectrum_gradient_ntt(I, target,
                                         definition.jacobian_row_sums(I)))
    assert np.allclose(got, 0.0, atol=1e-12)


def test_spectrum_gradient_is_exactly_the_adjoint_of_the_spectrum():
    """A single-pixel step is an exact linear functional of the gradient.

    The spectrum is multilinear -- every term is a product of distinct pixel
    values -- so it is linear in any one pixel, and definition.spectrum_gradient
    is an exact adjoint rather than a first-order one. The directional
    derivative of the loss along a step at a single pixel is therefore *exactly*
    sum_k w[k] * dA[k], with no epsilon and no truncation, and it must equal
    grad[p] * step.

    Two details make the obvious version of this test wrong. A float step is
    no use, because area_spectrum reads int(I[v]) and would truncate the
    perturbation away. A multi-pixel direction fails too: the spectrum is
    multilinear, not linear, so a step touching two pixels carries cross terms
    the gradient does not see. One pixel at a time is the only exact direction.

    This exercises the gradient as a linear functional of the spectrum, so a
    sign or factor error that cancelled entry-wise against the reference would
    not cancel here.
    """
    rng = np.random.default_rng(27)
    I = rng.integers(2, 6, size=(3, 3)).astype(np.int64)
    target = area_spectrum(rng.integers(1, 6, size=(3, 3)).astype(np.int64))
    scale = definition.jacobian_row_sums(I)
    base = np.array(area_spectrum(I), dtype=np.float64)
    w = np.array(definition.spectrum_residual_weights(base.tolist(), target,
                                                     scale))
    grad = np.array(spectrum_gradient_ntt(I, target, scale))
    for y, x, step in [(0, 0, 5), (1, 2, -3), (2, 2, 2), (0, 1, 4)]:
        moved = I.copy()
        moved[y, x] += step
        exact = float(w @ (np.array(area_spectrum(moved), dtype=np.float64)
                          - base))
        assert math.isclose(exact, float(grad[y * I.shape[1] + x]) * step,
                            rel_tol=1e-9, abs_tol=1e-12)


def test_spectrum_gradient_rejects_3d_with_a_reason():
    with pytest.raises(UnsupportedDimension, match="2D"):
        spectrum_gradient_ntt(np.ones((2, 2, 2), dtype=np.int64), [], [1, 2, 3])


# ---------------------------------------------------------------------------
# the backend, with all three quantities on the correlation path
# ---------------------------------------------------------------------------

def test_backend_row_sums_match_the_reference():
    rng = np.random.default_rng(28)
    I = rng.integers(0, 7, size=(3, 3)).astype(np.int64)
    assert NTTBackend().jacobian_row_sums(I) == list(
        definition.jacobian_row_sums(I))


def test_backend_gradient_matches_the_reference():
    rng = np.random.default_rng(29)
    I = rng.integers(1, 7, size=(3, 3)).astype(np.int64)
    target = area_spectrum(rng.integers(1, 7, size=(3, 3)).astype(np.int64))
    scale = definition.jacobian_row_sums(I)
    got = np.array(NTTBackend().spectrum_gradient(I, target, scale))
    want = np.array(definition.spectrum_gradient(I, target, scale))
    assert np.allclose(got, want, rtol=1e-9, atol=1e-7)


def test_backend_row_sums_pass_max_value_through():
    I = np.full((3, 3), 3, dtype=np.int64)
    assert NTTBackend(max_value=3).jacobian_row_sums(I) == list(
        definition.jacobian_row_sums(I))


def test_backend_does_not_delegate_the_row_sums_or_the_gradient():
    """The seam has to be a real seam, not a thin wrapper over definition."""
    rng = np.random.default_rng(30)
    I = rng.integers(1, 7, size=(3, 3)).astype(np.int64)
    target = area_spectrum(rng.integers(1, 7, size=(3, 3)).astype(np.int64))
    scale = definition.jacobian_row_sums(I)
    called = []
    originals = (definition.jacobian_row_sums, definition.spectrum_gradient)
    definition.jacobian_row_sums = lambda *a, **k: called.append("row_sums")
    definition.spectrum_gradient = lambda *a, **k: called.append("gradient")
    try:
        NTTBackend().jacobian_row_sums(I)
        NTTBackend().spectrum_gradient(I, target, scale)
    finally:
        (definition.jacobian_row_sums,
         definition.spectrum_gradient) = originals
    assert called == []


def test_bin_bound_counts_every_offset_pair():
    """The pair factor is ((2H-1)(2W-1))**2, not (2H-1)(2W-1)**2.

    The older formula counted pairs as if only the row coordinate were
    squared. It is a third of the total for a 3x3 and, more to the point,
    smaller than the worst *single* bin at 2x2, 3x3 and 4x4: 40 of 81 pairs
    land in one bin of a 2x2 against the 27 quoted. Nothing caught it because
    the maxval**3 slack in the bound is enormous for uint8; it is a bound that
    happens to hold, not a bound that is derived.
    """
    for shape, pairs in [((2, 2), 81), ((3, 3), 625), ((4, 4), 2401)]:
        I = np.zeros(shape, dtype=np.int64)
        H, W = shape
        assert bin_bound(I, max_value=1) == ((2 * H - 1) * (2 * W - 1)) ** 2 * I.size
        assert bin_bound(I, max_value=1) == pairs * I.size


def test_bin_bound_covers_the_worst_single_bin_of_a_small_image():
    """A direct count, so the pair factor cannot be weakened unnoticed."""
    for shape in [(2, 2), (3, 3), (4, 4)]:
        H, W = shape
        per_bin = Counter(
            abs(d1y * d2x - d1x * d2y)
            for d1y in range(-(H - 1), H) for d1x in range(-(W - 1), W)
            if (d1y, d1x) != (0, 0)
            for d2y in range(-(H - 1), H) for d2x in range(-(W - 1), W)
            if (d2y, d2x) != (0, 0) and (d2y, d2x) != (d1y, d1x))
        worst = max(per_bin.values())
        I = np.zeros(shape, dtype=np.int64)
        assert bin_bound(I, max_value=1) >= worst * I.size, shape


@pytest.mark.parametrize("fn", [area_spectrum_ntt, jacobian_row_sums_ntt])
def test_crt_paths_refuse_a_negative_pixel(fn):
    """CRT returns a value in [0, product), so a negative bin cannot appear.

    Without the guard the answer is an ordinary-looking integer of the wrong
    size rather than an error: a bin of -84 came back as 186253. Nothing
    downstream can tell, which is why this is a refusal and not a warning.
    """
    I = np.array([[-2, 3], [4, -5]], dtype=np.int64)
    with pytest.raises(ValueError, match="negative"):
        fn(I, max_value=5)


def test_a_negative_pixel_is_refused_rather_than_silently_wrapped():
    """The shape of the failure that motivated the guard."""
    I = np.array([[-2, 3], [4, -5]], dtype=np.int64)
    assert min(definition.area_spectrum(I)) < 0
    # and the reference is exact over the integers, negatives included
    assert definition.area_spectrum(I)[1] == -84


def test_the_gradient_still_handles_a_negative_image():
    """The float FFT carries the sign; only the CRT paths refuse negatives.

    The spectrum is passed in from definition.py, which is exact over the
    integers, since the default would go through the CRT path and refuse.
    """
    I = np.array([[-2, 3], [4, -5]], dtype=np.int64)
    target = area_spectrum(np.array([[1, 2], [3, 4]], dtype=np.int64))
    scale = definition.jacobian_row_sums(I)
    got = np.array(spectrum_gradient_ntt(I, target, scale,
                                         spectrum=definition.area_spectrum(I)))
    want = np.array(definition.spectrum_gradient(I, target, scale))
    assert np.allclose(got, want)


@pytest.mark.parametrize("fn", [area_spectrum_ntt, jacobian_row_sums_ntt])
def test_crt_paths_accept_a_zero_image(fn):
    """Zero is the boundary: min() == 0 must not trip the guard."""
    I = np.zeros((2, 2), dtype=np.int64)
    fn(I, max_value=1)


def test_the_gradient_never_falls_back_to_the_reference_spectrum():
    """It is the one place that could put the O(p**3) walk back.

    The weights need the spectrum of I, and reaching for definition there would
    defeat the whole function. The reference is stubbed to fail so a fallback
    is loud rather than merely slow.
    """
    rng = np.random.default_rng(31)
    I = rng.integers(1, 7, size=(3, 3)).astype(np.int64)
    target = area_spectrum(rng.integers(1, 7, size=(3, 3)).astype(np.int64))
    scale = definition.jacobian_row_sums(I)
    original = definition.area_spectrum
    definition.area_spectrum = lambda *a, **k: pytest.fail(
        "the gradient called the reference spectrum")
    try:
        spectrum_gradient_ntt(I, target, scale)
        NTTBackend().spectrum_gradient(I, target, scale)
    finally:
        definition.area_spectrum = original


def test_the_gradient_accepts_a_spectrum_from_the_caller():
    rng = np.random.default_rng(32)
    I = rng.integers(1, 7, size=(3, 3)).astype(np.int64)
    target = area_spectrum(rng.integers(1, 7, size=(3, 3)).astype(np.int64))
    scale = definition.jacobian_row_sums(I)
    want = np.array(definition.spectrum_gradient(I, target, scale))
    got = np.array(spectrum_gradient_ntt(I, target, scale,
                                         spectrum=area_spectrum(I)))
    assert np.allclose(got, want, rtol=1e-9, atol=1e-7)


def test_backend_reuses_the_spectrum_it_already_computed():
    """descent asks for the spectrum and then the gradient of the same image."""
    rng = np.random.default_rng(33)
    I = rng.integers(1, 7, size=(3, 3)).astype(np.int64)
    target = area_spectrum(rng.integers(1, 7, size=(3, 3)).astype(np.int64))
    scale = definition.jacobian_row_sums(I)
    backend = NTTBackend()
    backend.area_spectrum(I)                 # now cached
    calls = []
    real = ntt_spectrum.area_spectrum_ntt
    ntt_spectrum.area_spectrum_ntt = lambda *a, **k: calls.append(1) or real(*a, **k)
    try:
        got = np.array(backend.spectrum_gradient(I, target, scale))
    finally:
        ntt_spectrum.area_spectrum_ntt = real
    assert calls == [], "the cached spectrum was not used"
    want = np.array(definition.spectrum_gradient(I, target, scale))
    assert np.allclose(got, want, rtol=1e-9, atol=1e-7)


def test_backend_does_not_reuse_a_spectrum_from_another_image():
    """A stale cache entry would silently weight the wrong spectrum."""
    rng = np.random.default_rng(34)
    I = rng.integers(1, 7, size=(3, 3)).astype(np.int64)
    J = rng.integers(1, 7, size=(3, 3)).astype(np.int64)
    target = area_spectrum(rng.integers(1, 7, size=(3, 3)).astype(np.int64))
    scale = definition.jacobian_row_sums(I)
    backend = NTTBackend()
    backend.area_spectrum(I)                 # caches I, not J
    got = np.array(backend.spectrum_gradient(J, target, scale))
    want = np.array(definition.spectrum_gradient(J, target, scale))
    assert np.allclose(got, want, rtol=1e-9, atol=1e-7)
