"""Area spectrum by triple correlation, section extraction, and NTTs.

The fast path contemplated by T0.1-02, written to be checked against the
reference in definition.py rather than trusted.

The identity
------------
For an image I on Z^d, the three-point correlation is

    T(d1, d2) = sum_v I[v] I[v+d1] I[v+d2]

and the area spectrum is its diagonal sum by volume,

    A[k] = sum over offset pairs with |det| = k of T(d1, d2)

taken over offsets whose d+1 points are distinct, exactly as definition.py
requires. A nonzero determinant already forces affine independence and hence
distinctness, so the restriction only ever removes pairs from bin 0.

This is inherently a 2D method. A d-dimensional volume is the determinant of a
d x d matrix built from d offsets, so a three-point correlation -- two offsets --
determines the volume only when d = 2. For d >= 3 the same scheme needs a
(d+1)-point correlation and a d-offset section, which costs one section per
(d-1)-tuple of offsets rather than per offset; see the error in
area_spectrum_ntt.

Why padding, and why a power of two
----------------------------------
The correlation is cyclic, so it is computed on a torus. Zeros are padding, and
a cyclic sum over a padded array equals the linear sum over the original, so
the padding is what makes the cyclic answer the linear one. A radix-2 NTT needs
a power-of-two length, and the 2D problem is turned into a single 1D transform
by embedding I in the corner of a padded rectangle and flattening row-major, so
a shift (dy, dx) is the flat shift dy*Mx + dx.

The rectangle is sized per axis, not from the longest side, because the two
axes have genuinely different requirements and forcing them equal wastes work on
every non-square image. With a row stride Mx, two offsets collide only if
(dy1-dy2)*Mx = dx2-dx1, and since |dx| <= W-1 that needs Mx > 2(W-1). The same
bound comes from the borrow: when x + dx < 0 the flat index steps into the
previous row, and the column it lands on, Mx-(W-1), must be past the image, so
again Mx >= 2W-1. The row axis needs only My >= 2H-1 by the identical argument.
Both are rounded up to powers of two, which also makes the flat length My*Mx a
power of two as radix-2 requires.

Padding from the longest side instead -- an M x M square with M from
max(W, H) -- is correct but pays for the larger axis on both. On a 2x8 image the
rectangle is 4x16 where the square is 16x16, a 4x saving in transform length,
and it is the common case for photographic crops:

    3x3   8x8    = 64   (square:  64)    0% saved
    5x5  16x16   = 256   (square: 256)    0% saved
    3x7   8x16   = 128   (square: 256)   50% saved
    2x8   4x16   = 64   (square: 256)   75% saved

The padding requirement is the whole of it: the bin index is an exact integer
determinant of signed offsets and is never reduced mod p, so the modulus does
not have to resolve volumes.

Why several small primes and CRT
--------------------------------
Every value here is computed mod p, so a bin can come back as its residue
rather than as a nonnegative count. Choosing p large enough to hold a bin would
mean primes with thousands of bits, which no fast transform wants. Instead a
few small primes are used whose product provably exceeds an upper bound on any
bin, and the Chinese Remainder Theorem then reconstructs the exact integer. The
result is exact, not a tolerance, and the bound is asserted rather than assumed.

The primes are searched for rather than hardcoded, because the 2-adic exponent
they need is a property of the padded transform length, hence of the image
shape, and hardcoding a table would have to anticipate every shape. For a
required order 2**n the search takes the smallest primes of the form
k*2**n + 1, which is exactly the set with a root of unity of that order, and
checks each with galois.is_prime. This is the same family as the familiar
998244353 = 119*2**23 + 1, reached here by search rather than by memory.

The bound is deliberately independent of the pixel values, so the prime set is
a function of the shape and dtype alone. That makes it cacheable across images
of the same type -- the point of caching it -- and it removes a sharp edge
where an unusually bright image would silently need more primes than a dim one.
The cost is that the bound is set by the dtype's maximum rather than the
image's, so a dim image pays for a bright one's worst case. Using the dtype
maximum is still vastly tighter than any single prime, since the bound grows
only cubically in the value while the primes multiply.

Accumulation is done with np.add.at on int64, not np.bincount: bincount
accumulates in float64, and a bin of a 64x64 image can exceed 2**53, where
float64 addition silently stops being exact.

Halving the offset loop
-----------------------
A triangle is named once per way of naming its three points, and naming them
differently leaves both the correlation and the bin alone: T is a product of
the same three pixel values, and |det| is six times the same area whatever the
order. So the six relabellings

    (d1,d2)  (d2,d1)  (-d1,d2-d1)  (d1-d2,-d2)  (d2-d1,-d1)  (-d2,d1-d2)

all carry the same value into the same bin, and each one's *second* offset is
one of +-d1, +-d2, +-(d1-d2) -- one from each opposite pair. Any transversal of
the opposite pairs therefore meets exactly three of the six, whatever the
triangle, so a loop over such a half with a flat factor of 2 counts every
triangle as often as a loop over all of them. That is half the transforms, and
the weight is a constant rather than a clever per-triangle one because the case
analysis collapses: the only orbits with fewer than six members are those with
d1-d2 out of range, and those have a vanishing section, since
T(d1,d2) = T(d1-d2,-d2) by re-anchoring and T is zero for an out-of-range
offset.

This is a property of the three-point correlation, not of the offset loop as
such, and it does not extend to the two neighbours that share the loop shape.
The row sums weight a *masked* copy, so their section depends on d1 as well as
on d2-d1 -- the mask keeps only anchors that are themselves pixels with both
partners in the image, and re-anchoring moves that condition -- and the
gradient weights a two-point correlation, which sees only the difference
d1-d2. Neither is invariant under relabelling, so neither loop can be halved;
the swap symmetry alone is not enough, because a pair of offsets can sit
entirely in one half. spectrum_mod below is the only one of the three where
this applies.

Where this actually wins
------------------------
The reference is O(pixels**(d+1)) tuple enumerations; this is one
length-My*Mx transform per offset, i.e. about (2W)(2H) transforms of length
My*Mx, so O(pixels**2 log pixels) with a vectorised constant. The offset loop
runs over half the offsets, so the transform count is about (W)(2H) and the
measured gain is a flat 2x on this function. Against definition.area_spectrum,
exact and equal at every size, with max_value=5 (three passes) on a 0..5 image:

    4x4    pad  8x8     64    0.071s -> 0.011s    6.4x
    8x8    pad 16x16   256    0.237s -> 0.082s    2.9x
    12x12  pad 32x32  1024    2.897s -> 0.639s    4.5x
    16x16  pad 32x32  1024   16.708s -> 1.165s   14.3x
    6x10   pad 16x32   512    0.217s -> 0.137s    1.6x
    4x20   pad  8x64   512    0.490s -> 0.193s    2.5x

The value-independent bound is the price of the dtype default, and with it
this method is *slower* than the reference over most of the range. The same
four shapes with no max_value, so int64's range and 15-20 passes instead of 3:

    4x4    0.071s -> 0.066s    1.1x        8x8   0.237s -> 0.511s   0.5x
    12x12  2.897s -> 3.188s    0.9x       16x16 16.708s -> 5.766s   2.9x

Halving roughly doubles each of those ratios, since the loop is the whole of the
pass and the reference is unaffected by it. Before the change the same
measurement was 0.134s, 0.996s, 6.353s and 11.934s -- 0.5x, 0.2x, 0.5x and 1.4x
-- so 4x4 and 16x16 now beat the reference and 8x8 and 12x12 still do not. The
dtype-default crossover moves from 16x16 down to 4x4 and the max_value=5
crossover from 8x8 to 4x4. Pass max_value when the caller knows the range, and
the numbers in the first table are the ones to plan against.

Two things to read off the first table. The search yields the *smallest* primes
of the required 2-adic order -- 257, 769, 3329 for order 2**8 -- where the
hand-written table it replaced started at 998244353. A short modulus makes each
pass cheaper but forces more of them, since the product has to clear the same
bound, and that is the whole of why these numbers are worse than the ones this
module quoted before the search landed; the rectangle leaves a square's pad
unchanged, so it costs nothing on the square rows.

The non-square rows show the same effect from the other side: 6x10 and 4x20 sit
near the crossover because a handful of pixels makes the reference's
enumeration nearly free while this still pays for a full pass per prime, and
their long axis does not stop the rectangle from saving the padding. The
crossover is about area, not aspect ratio. This is a numpy implementation, so
the absolute numbers are far off what a compiled implementation would give; the
shape of the curve is the point, not the factor.
"""
import functools
import math

import galois
import numpy as np

import definition


def pad_shape(I):
    """Padded (rows, cols) for I, each axis a power of two.

    My >= 2H-1 and Mx >= 2W-1 are what the flat encoding requires: with a row
    stride Mx, two offsets collide only if (dy1-dy2)*Mx = dx2-dx1, and
    |dx| <= W-1 rules that out once Mx > 2(W-1). The same bound falls out of
    the borrow, since a negative x lands on column Mx-(W-1) of the previous row
    and that must be past the image. Rows need only My >= 2H-1 by the same
    argument, so the axes are sized independently and a non-square image is not
    charged for its longer side twice.

    Both axes are floored at 2, which a degenerate axis would otherwise miss: a
    1-row image would otherwise get rows == 1, and the centring test
    dy >= rows//2 would then read 0 >= 0 and report a shift of 0 as -1. That
    empties the valid set and silently zeroes bin 0.
    """
    H, W = I.shape
    rows = 2
    while rows < 2 * H - 1:
        rows *= 2
    cols = 2
    while cols < 2 * W - 1:
        cols *= 2
    return rows, cols


def pad_length(I):
    """The flat transform length, My*Mx, a power of two."""
    rows, cols = pad_shape(I)
    return rows * cols


def pad_length_of_shape(shape):
    """pad_length for a bare shape, so the prime cache needs no array."""
    rows, cols = 2, 2
    while rows < 2 * shape[0] - 1:
        rows *= 2
    while cols < 2 * shape[1] - 1:
        cols *= 2
    return rows * cols


@functools.lru_cache(maxsize=None)
def _primes_of_order_2(exponent):
    """The smallest primes p = k*2**exponent + 1, in increasing order.

    Exactly the primes carrying a 2**exponent-th root of unity, which is what a
    radix-2 transform of that length needs. A generator of the multiplicative
    group raised to (p-1)/2**exponent gives the root, so no separate search for
    a root is needed.

    Searched rather than hardcoded because the exponent follows from the padded
    length, hence from the image shape; a fixed table would have to anticipate
    every shape. Cached per exponent since many shapes share one.
    """
    step = 1 << exponent
    out = []
    candidate = step + 1
    while len(out) < 64:                     # far more than any image needs
        prime = int(galois.next_prime(candidate - 1))
        if (prime - 1) % step == 0:
            out.append(prime)
        # resume just past this prime, on the next 1 mod step
        candidate = prime + 1
        candidate += (step - candidate % step) % step
    return tuple(out)


_INT64_MAX = int(np.iinfo(np.int64).max)


def dtype_max(dtype):
    """The largest value the dtype can hold, for a value-independent bound."""
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return 1
    if dtype.kind in "iu":
        return int(np.iinfo(dtype).max)
    if dtype.kind == "f":
        # a float image is scaled down to an integer grid first; 255 is the
        # 8-bit working range, and the spectrum is only defined on integers
        return 255
    raise TypeError(f"no value bound known for dtype {dtype}")


def bin_bound(I, max_value=None):
    """A provable upper bound on any single bin, from the dtype not the values.

    A bin sums T(d1, d2) over the offset pairs landing in it. T is nonzero only
    for offsets that fit inside the image, so it has at most I.size terms, each
    at most maxval**3; and there are at most ((2H-1)*(2W-1))**2 offset pairs in
    all. The product of the chosen primes must exceed this, or CRT reconstructs
    a residue rather than the count.

    The pair count is the total, not the largest bin, because the largest bin is
    not known without enumerating the determinants. (2H-1)*(2W-1)**2, which
    counts pairs as if only the row coordinate were squared, is a third of the
    total for a 3x3 and undercounts the worst single bin outright at 2x2, 3x3
    and 4x4: 40 pairs share one bin of a 2x2 against a 27 quoted. It happens
    not to bite for uint8, whose maxval**3 slack is large, but it is not a bound.

    Deliberately blind to the actual pixels: that is what lets the prime set be
    cached per (shape, dtype), and it costs only that a dim image is charged for
    a bright one's worst case. For int64 that is not free -- the dtype's range
    is 2**63, so a 6x6 image wants 16 primes where uint8 wants 4, each one a
    full pass over the sections. Pass max_value to charge for the range the
    image actually uses when the caller knows it; the cache key includes it, so
    the tightened set is cached just as well.
    """
    H, W = I.shape
    if max_value is None:
        max_value = dtype_max(I.dtype)
    pairs = ((2 * H - 1) * (2 * W - 1)) ** 2
    return pairs * I.size * max(int(max_value), 1) ** 3


def row_sum_bound(I, max_value=None):
    """A provable upper bound on any single row-sum entry.

    A row-sum entry is a sum of the same terms as a spectrum bin but with two
    factors instead of three, so the count is the same and only the power of
    maxval drops, from 3 to 2. The factor 3 belongs to it as well: a row sum
    collects all three pairs of each triple, not one.
    """
    H, W = I.shape
    if max_value is None:
        max_value = dtype_max(I.dtype)
    pairs = ((2 * H - 1) * (2 * W - 1)) ** 2
    return 3 * pairs * I.size * max(int(max_value), 1) ** 2


@functools.lru_cache(maxsize=None)
def primes_for_shape(shape, dtype, bound):
    """Smallest prefix of 2**n-friendly primes whose product exceeds bound.

    Cached on (shape, dtype, bound). Shape fixes the transform length and so
    the required 2-adic exponent; dtype and shape together fix the bound; the
    bound fixes how many primes are needed. Two images of the same shape and
    dtype therefore share one prime set and one search.
    """
    n = pad_length_of_shape(shape)
    exponent = n.bit_length() - 1
    available = _primes_of_order_2(exponent)
    chosen = []
    product = 1
    for p in available:
        chosen.append(p)
        product *= p
        if product > bound:
            break
    if product <= bound:
        raise ValueError(
            f"no available prime set is large enough: {len(chosen)} primes of "
            f"order 2**{exponent} reach {product}, need more than {bound}"
        )
    return tuple(chosen)


def primitive_root(p):
    """A generator of the multiplicative group modulo p, via galois."""
    return int(galois.primitive_root(p))


class UnsupportedDimension(ValueError):
    """Raised when the triple-correlation method cannot reach the dimension."""


@functools.lru_cache(maxsize=None)
def root_table(M, p, root):
    """tab[i] = omega**i mod p for i in 0..M-1, omega = root**((p-1)//M).

    Built by repeated modular multiplication. The vectorised-looking
    np.power(omega, np.arange(M), p) is wrong twice over: numpy's power takes
    no modulus, so the values overflow int64 long before the % p, and a single
    butterfly stage needs only the first M/2 of them anyway. One Python pass per
    (M, p) is negligible beside the O(M log M) transform it serves, and is
    cached across every transform at that size.
    """
    omega = pow(root, (p - 1) // M, p)
    if pow(omega, M, p) != 1 or pow(omega, M // 2, p) == 1:
        raise ValueError(f"root {root} has no order {M} modulo {p}")
    tab = np.empty(M, dtype=np.int64)
    tab[0] = 1
    for i in range(1, M):
        tab[i] = tab[i - 1] * omega % p
    return tab


def ntt(a, p, root):
    """Forward transform fhat[k] = sum_v a[v] omega**(v*k), omega of order len(a).

    Iterative Cooley-Tukey, radix 2. The stage twiddles are
    omega**(j*M/length) for j < length/2, which is the root table sliced with
    stride M/length -- the even exponents, not the leading half of the table.
    """
    a = np.array(a, dtype=np.int64, copy=True)
    M = len(a)
    if M & (M - 1):
        raise ValueError(f"transform length {M} is not a power of two")
    if M == 1:
        return a
    j = 0
    for i in range(1, M):
        bit = M >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    tab = root_table(M, p, root)
    length = 2
    while length <= M:
        half = length >> 1
        tw = tab[0:M:(M // length)][:half]
        blocks = a.reshape(-1, 2, half)
        u = blocks[:, 0, :].copy()
        v = blocks[:, 1, :] * tw % p
        blocks[:, 0, :] = (u + v) % p
        blocks[:, 1, :] = (u - v) % p
        length <<= 1
    return a


def intt(a, p, root):
    """Inverse of ntt, including the 1/M normalisation."""
    M = len(a)
    if M == 1:
        return np.array(a, dtype=np.int64, copy=True)
    inv_root = pow(root, p - 2, p)
    return ntt(a, p, inv_root) * pow(M, p - 2, p) % p


def embed(I, pad):
    """I padded into the corner of a pad-shaped array, flattened row-major.

    Row-major means the flat index of (y, x) is y*Mx + x, so a 2D shift
    (dy, dx) is the flat shift dy*Mx + dx. That is the whole embedding trick:
    one length-My*Mx transform serves the 2D problem.
    """
    rows, cols = pad
    out = np.zeros((rows, cols), dtype=np.int64)
    H, W = I.shape
    out[:H, :W] = I
    return out.ravel()


def _offsets(pad, shape):
    """Per-axis signed coordinates and the valid mask for every flat shift.

    Coordinates are signed because a displacement of -1 and one of cols-1 are
    the same shift on the torus, while the determinant has to be formed on the
    signed representative to match definition.volume. Valid means the offset
    could arise between two points of the image; T is zero for the rest, so
    they are skipped rather than transformed.
    """
    rows, cols = pad
    N = rows * cols
    idx = np.arange(N, dtype=np.int64)
    # x first: dx is fixed by s mod cols because dy*cols vanishes mod cols.
    # Centring dx afterwards borrows from the row, so dy has to be recovered
    # from the already-centred dx rather than from s // cols independently --
    # centring both separately mislabels every shift whose dx is negative as a
    # diagonal one.
    dx = idx % cols
    dx = np.where(dx >= cols // 2, dx - cols, dx)
    dy = ((idx - dx) // cols) % rows
    dy = np.where(dy >= rows // 2, dy - rows, dy)
    H, W = shape
    valid = (np.abs(dy) <= H - 1) & (np.abs(dx) <= W - 1)
    return dy, dx, valid


def _half_offsets(dy, dx, valid):
    """One offset from each opposite pair (d, -d): a half for the d2 loop.

    The halving itself is a symmetry of the three-point correlation rather than
    of the offsets, and is argued in the module docstring; what this picks is
    just a set of representatives. dy > 0 with the dy == 0 tie broken by
    dx > 0 takes exactly one of d and -d for every nonzero offset, so the count
    is (valid.sum() - 1) // 2, the one being the zero offset that both halves
    leave out.

    Any such transversal works -- the weights are flat, so which member of each
    pair survives is irrelevant. This one is the cheapest to state: it needs
    only the coordinate arrays already in hand and no extra index arithmetic.
    """
    return valid & ((dy > 0) | ((dy == 0) & (dx > 0)))


def triple_correlation(I, d2, pad, p, root, Ihat=None):
    """One section T(., d2) of the three-point correlation, mod p.

    Fixing d2 turns the three-point correlation into a two-point one: with
    g = I * shift(I, d2) the section is the cyclic correlation of g with I, and
    in the frequency domain

        That[k] = ghat[-k] * Ihat[k]

    with the negation on the *shifted* transform, not on I. Mirroring it the
    other way still returns a plausible-looking array of the right shape, just
    the wrong numbers, so the tests compare single entries against a direct
    triple sum.

    Returns the whole section, indexed by the flat shift of d1.
    """
    rows, cols = pad
    N = rows * cols
    shift = int(d2[0] * cols + d2[1]) % N
    flat = embed(I, pad)
    # np.roll(a, k)[v] is a[(v - k) % N], so the rolled operand is -shift: we
    # want flat[(v + shift) % N]. The sign here is easy to get backwards, and
    # getting it wrong yields a section of the right shape with plausible
    # values, so the tests compare entries against a direct triple sum.
    g = flat * np.roll(flat, -shift) % p
    if Ihat is None:
        Ihat = ntt(flat, p, root)
    prod = ntt(g, p, root)[(-np.arange(N)) % N] * Ihat % p
    return intt(prod, p, root)


def spectrum_mod(I, p, root, pad=None):
    """The area spectrum of I, computed mod p.

    Streams the sections one at a time, adding each into the bins and
    forgetting it. The full section is ((2H-1)(2W-1))^2 entries -- for a 32x32
    image about 2.8 million, and for 64x64 about 2.5 * 10**8 -- and its sum by
    volume is all the spectrum needs, so materialising it would buy nothing.

    The loop runs over half the offsets and carries a factor of 2 to match; see
    the module docstring. A section's second offset has to be one specific
    member of its opposite pair for the count to be right, which is why the
    factor is applied once at the end rather than folded into the per-iteration
    factor or into the section itself.
    """
    H, W = I.shape
    if pad is None:
        pad = pad_shape(I)
    rows, cols = pad
    N = rows * cols
    n_spectrum = definition.spectrum_length(I)
    dy, dx, valid = _offsets(pad, I.shape)
    A = np.zeros(n_spectrum, dtype=np.int64)
    Ihat = ntt(embed(I, pad), p, root)
    for s2 in np.nonzero(_half_offsets(dy, dx, valid))[0]:
        s2 = int(s2)
        b2y, b2x = int(dy[s2]), int(dx[s2])
        # bin for every d1, and the pairs that keep the three points distinct
        bins = np.abs(dy * b2x - dx * b2y)
        keep = valid & (bins < n_spectrum) & (np.arange(N) != s2)
        keep[0] = False                    # d1 == 0 repeats v0
        T = triple_correlation(I, (b2y, b2x), pad, p, root, Ihat)
        np.add.at(A, bins[keep], T[keep])
        A %= p
    # s2 == 0 is not in the half, so the zero offset needs no test above, and
    # neither does d1 == d2: relabelling never turns a distinct pair into a
    # repeated point, so the exclusion is uniform across an orbit.
    return 2 * A % p


def _masked_copy(I, pad, dy, dx):
    """I with the anchors whose partner a+d stays in the image, flattened.

    The rectangle of surviving anchors is a slice of the padded array, so this
    is a couple of numpy assignments rather than a per-pixel test. It is
    *not* a shift: the entries where the partner leaves the image read zero
    instead of a wrapped neighbour, which is what makes the row sums below
    correct.
    """
    rows, cols = pad
    H, W = I.shape
    out = np.zeros((rows, cols), dtype=np.int64)
    # The surviving anchors a (those with a+d still in the image) are the
    # rectangle [max(0,-dy), H-max(0,dy)) x [max(0,-dx), W-max(0,dx)), which
    # sits at the same offsets in the padded output. The source values are the
    # same rectangle read at a+d, i.e. shifted by +d.
    y0, x0 = max(0, -dy), max(0, -dx)
    y1, x1 = min(H, H - dy), min(W, W - dx)
    out[y0:y1, x0:x1] = I[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return out.ravel()


def row_sums_mod(I, p, root, pad=None):
    """The Jacobian row sums of I, mod p, as spectrum_mod is the spectrum.

    A row sum is the gradient of a bin with respect to a pixel, so it is that
    bin's triples with two of the three factors kept instead of three. The
    reference walks triples; here the anchor is summed out by a correlation
    instead.

    Two things differ from the spectrum and both are load-bearing:

    The kernel is a *masked* copy of I, not I * shift(I, d1). The latter is
    the three-point correlation -- it has three factors, so it is the spectrum
    and not the row sums. What is wanted is the two-point sum over v and v+d2
    restricted to anchors where v+d1 is still in the image, and the restriction
    is a mask. Using the unmasked product gives the right shape and the wrong
    numbers, silently.

    The loop runs over offset *pairs* (d1, d2) rather than over the two factors
    of one, and the factor of 3 is applied at the end. Each triple contributes
    I[a]I[b] + I[a]I[c] + I[b]I[c] to its bin, which cyclic symmetry of the
    three offsets makes exactly three times the I[a]I[c] term the loop already
    counted -- so multiplying once at the end is right, and multiplying per
    iteration would be wrong. The same 3 belongs to the gradient, for the
    unrelated reason that a triple moves all three of its pixels.

    The loop is over *all* the offsets, unlike spectrum_mod's half. The halving
    there is a symmetry of the three-point correlation, and the section here is
    not that: the mask keeps only anchors that are themselves pixels with both
    partners in the image, so B depends on d1 and not on d2-d1 alone, and
    re-anchoring moves that condition rather than preserving it.
    """
    if pad is None:
        pad = pad_shape(I)
    rows, cols = pad
    N = rows * cols
    n_spectrum = definition.spectrum_length(I)
    dy, dx, valid = _offsets(pad, I.shape)
    R = np.zeros(n_spectrum, dtype=np.int64)
    Ihat = ntt(embed(I, pad), p, root)
    flat_idx = np.arange(N)
    neg = (-flat_idx) % N
    for s1 in np.nonzero(valid)[0]:
        s1 = int(s1)
        if s1 == 0:
            continue                       # d1 == 0 repeats a point
        b1y, b1x = int(dy[s1]), int(dx[s1])
        # the second point of each triple; the third is the d2 being binned
        bins = np.abs(dy * b1x - dx * b1y)
        keep = valid & (bins < n_spectrum) & (flat_idx != s1)
        keep[0] = False                    # d2 == 0 repeats v0
        # B[d2] = sum_a K[a] I[a+d2] with K the masked copy: one correlation
        K = ntt(_masked_copy(I, pad, b1y, b1x), p, root)
        B = intt(K[neg] * Ihat % p, p, root)
        np.add.at(R, bins[keep], B[keep])
        # R is int64 and accumulates one residue per offset, so on a large
        # image the running sum can exceed int64 before the final reduction.
        # Cheap here -- n_spectrum entries against a transform of length N.
        R %= p
    return 3 * R % p


def spectrum_gradient_ntt(I, target, scale, pad=None, spectrum=None):
    """The spectrum gradient of I against target, via per-offset correlations.

    definition.spectrum_gradient walks every triple and differentiates it
    three ways. Written in offsets that is, for each ordered pair of distinct
    non-zero offsets (d1, d2) with k = |det(d1, d2)| in range, a contribution

        3 * w[k] * I[a + d1] * I[a + d2]      to every anchor a

    where w is the residual weight. Holding d1 fixed, the weight depends on d2
    only through a single scalar per d2, so the whole inner sum is a
    correlation against the kernel h[d2] = w[|det(d1, d2)|]:

        S_d1 = sum_{d2} h[d2] I[a + d2]        one transform, not one per pair
        grad += 3 * shift(I, d1) * S_d1

    With O(p) offsets and one length-O(p) transform each, that is
    O(p**2 log p) against the reference's O(p**3). The weights are floats, so
    this path uses numpy's float FFT rather than the NTT and returns a float
    vector; it is exact only up to rounding.

    The loop is over *all* the offsets, unlike spectrum_mod's half. The halving
    there needs a summand that relabelling the triangle leaves alone, and
    w[|det|] times a two-point correlation does not have one: the correlation
    sees only the difference d1-d2, so the three relabellings of a triangle that
    share an outer offset in one half can carry three different values. The
    swap alone is not enough either, since a pair of offsets can both land in
    the same half and so be counted twice, or in the other and be missed.

    The weights need the spectrum of I, which costs another O(p**2 log p).
    spectrum passes one in for a caller that has just computed it; the default
    is the fast spectrum, never definition.area_spectrum -- reaching for the
    reference there would put the O(p**3) enumeration back inside the one
    function that is supposed to have removed it.
    """
    I = np.asarray(I)
    if I.ndim != 2:
        raise UnsupportedDimension(
            f"the gradient follows the triple correlation and so reaches 2D "
            f"only, not {I.ndim}D"
        )
    if pad is None:
        pad = pad_shape(I)
    rows, cols = pad
    H, W = I.shape
    n_spectrum = definition.spectrum_length(I)
    if spectrum is None:
        spectrum = area_spectrum_ntt(I, pad)
    w = np.asarray(definition.spectrum_residual_weights(spectrum, target, scale),
                   dtype=np.float64)
    dy, dx, valid = _offsets(pad, I.shape)
    flat = embed(I, pad).astype(np.float64)
    Ifft = np.fft.fft(flat)
    idx = np.arange(rows * cols)
    grad = np.zeros(rows * cols, dtype=np.float64)
    for s1 in np.nonzero(valid)[0]:
        s1 = int(s1)
        if s1 == 0:
            continue
        b1y, b1x = int(dy[s1]), int(dx[s1])
        bins = np.abs(dy * b1x - dx * b1y)
        keep = valid & (bins < n_spectrum) & (idx != s1)
        keep[0] = False
        if not keep.any():
            continue
        # kernel over d2: one float weight per valid offset, zero elsewhere
        h = np.zeros(rows * cols, dtype=np.float64)
        h[keep] = w[bins[keep]]
        # sum_d2 h[d2] I[a + d2]; the negation is on h, as in triple_correlation
        S = np.fft.ifft(np.fft.fft(h)[(-idx) % (rows * cols)] * Ifft).real
        # np.roll by -shift gives flat[v + shift]; correct on the image for any
        # offset in range, and the padding it gets wrong is never read back
        grad += 3.0 * np.roll(flat, -s1) * S
    # read back only the image, which is the top-left H x W corner of the
    # padded frame -- not the first H*W flat entries, which would be the image
    # plus a slice of the first padding row whenever cols > W
    image = np.arange(rows * cols).reshape(rows, cols)[:H, :W]
    return grad[image.ravel()]


def crt(residues, primes):
    """Exact reconstruction from residues mod pairwise coprime primes.

    Incremental: after each step x is the unique value in [0, product) matching
    the residues seen so far. Valid only while the true value is below the
    product, which bin_bound and area_spectrum_ntt together enforce.
    """
    x, m = int(residues[0]), primes[0]
    for r, p in zip(residues[1:], primes[1:]):
        t = (int(r) - x) * pow(m % p, p - 2, p) % p
        x += m * t
        m *= p
    return x


def primes_for(I, max_value=None):
    """The primes needed for I, as (prime, primitive root) pairs.

    Derived from the shape and dtype alone, so it is cached: two images of the
    same type share one search. Each prime carries a root of unity of order
    2**exponent, exponent being log2 of the padded transform length.

    max_value tightens the bound below the dtype's range when the caller knows
    the image is dim; see bin_bound for why the default is value-independent
    and what that costs.
    """
    I = np.asarray(I)
    bound = bin_bound(I, max_value)
    key = I.dtype.str if max_value is None else (I.dtype.str, int(max_value))
    chosen = primes_for_shape(tuple(I.shape), key, bound)
    return [(p, primitive_root(p)) for p in chosen]


def _check_nonneg(I):
    """Reject a negative pixel, which the CRT path cannot represent.

    crt returns the unique value in [0, product) matching the residues, so a
    bin whose true value is negative comes back as its residue minus the
    modulus product instead. Nothing downstream can tell: the answer is an
    ordinary-looking integer of the wrong size, which is worse than a refusal.
    The float paths (the gradient) have no such trouble and do not call this.

    A signed reconstruction would recover these, but the products then have to
    cover a range on both sides of zero and the bounds below would need
    doubling; the images this module is aimed at are counts, so the guard is
    the honest cheap answer for now.
    """
    if I.size and int(I.min()) < 0:
        raise ValueError(
            f"the NTT path reconstructs bins by CRT, which returns a value in "
            f"[0, product) and so cannot represent a negative one; this image "
            f"has a pixel of {int(I.min())}. Shift it nonneg, or use "
            f"definition.py, which is exact over the integers."
        )


def area_spectrum_ntt(I, pad=None, primes=None, max_value=None):
    """The exact area spectrum of I, via sections and CRT.

    Equals definition.area_spectrum(I) exactly, in integers. max_value is
    forwarded to primes_for; see bin_bound for what it buys.
    """
    I = np.asarray(I)
    if I.ndim != 2:
        raise UnsupportedDimension(
            f"triple correlation reaches 2D only: a {I.ndim}D volume is a "
            f"{I.ndim}x{I.ndim} determinant needing {I.ndim} offsets, but a "
            f"three-point correlation supplies two. A {I.ndim}D version needs a "
            f"{I.ndim + 1}-point correlation and one section per "
            f"({I.ndim - 1})-tuple of offsets, not per offset."
        )
    if primes is None:
        primes = primes_for(I, max_value)
    _check_nonneg(I)
    residues = [spectrum_mod(I, p, root, pad) for p, root in primes]
    ps = [p for p, _ in primes]
    return [crt([r[k] for r in residues], ps) for k in
            range(definition.spectrum_length(I))]


def primes_for_row_sums(I, max_value=None):
    """The primes the row sums of I need, as (prime, primitive root) pairs.

    Separate from primes_for because the bound is a different power of the
    value -- quadratic rather than cubic -- so the two quantities can need
    different prime counts, and picking the wrong one leaves CRT reconstructing
    a residue instead of a count.

    The "row-sums" tag on the cache key is belt and braces rather than the
    thing that keeps them apart: the bound is already part of the key, so a
    row-sum search cannot collide with a spectrum search unless the two bounds
    agree. The tag is there to stop that happening by accident if a bound is
    ever loosened to match, and to keep the two entries from evicting each
    other out of the cache.
    """
    I = np.asarray(I)
    bound = row_sum_bound(I, max_value)
    key = (I.dtype.str, int(max_value)) if max_value is not None else I.dtype.str
    chosen = primes_for_shape(tuple(I.shape), ("row-sums", key), bound)
    return [(p, primitive_root(p)) for p in chosen]


def jacobian_row_sums_ntt(I, pad=None, primes=None, max_value=None):
    """The exact Jacobian row sums of I, via correlations and CRT.

    Equals definition.jacobian_row_sums(I) exactly, in integers.
    """
    I = np.asarray(I)
    if I.ndim != 2:
        raise UnsupportedDimension(
            f"row sums follow the triple correlation and so reach 2D only, "
            f"not {I.ndim}D"
        )
    _check_nonneg(I)
    if primes is None:
        primes = primes_for_row_sums(I, max_value)
    residues = [row_sums_mod(I, p, root, pad) for p, root in primes]
    ps = [p for p, _ in primes]
    return [crt([r[k] for r in residues], ps) for k in
            range(definition.spectrum_length(I))]


def jacobian_column_ntt(I, pixel, max_value=None):
    """The exact Jacobian column of I at `pixel`, in integers.

    Equals definition.jacobian_column(I, pixel) exactly. Reaches 2D only, like
    the other correlation paths.
    """
    I = np.asarray(I)
    if I.ndim != 2:
        raise UnsupportedDimension(
            f"a column is a single-pixel relabelling of the offset sums and so "
            f"reaches 2D only, not {I.ndim}D"
        )
    _check_nonneg(I)
    H, W = I.shape
    py, px = pixel
    if not (0 <= py < H and 0 <= px < W):
        raise ValueError(f"pixel {tuple(pixel)} is not a coordinate of an image of shape {I.shape}")
    n_spectrum = definition.spectrum_length(I)
    # The column is 3 * sum over ordered offset pairs of I[p+d1] * I[p+d2], so
    # it is bounded by 3 * (sum of the pixels)**2 -- two factors of the pixel
    # value rather than three, and no dependence on the pixel count beyond that
    # sum. Well inside int64 for any real image, but not for an int64 one
    # holding values near 2**63, so the bound is checked rather than assumed and
    # the arithmetic is never silently wrapped. The sum is taken in Python
    # integers for the same reason: an int64 total would itself wrap.
    total = sum(int(v) for v in I.ravel())
    bound = 3 * total * total
    if bound > _INT64_MAX:
        raise ValueError(
            f"a Jacobian column of this image would exceed int64 "
            f"(3 * sum**2 = {bound}); reduce the pixel range"
        )
    # The offsets that keep p + d inside the image: a rectangle, which is the
    # same shape as the image itself and much smaller at a corner than the
    # full (2H-1) x (2W-1) offset grid the spectrum loops over.
    dy = np.arange(-py, H - py)
    dx = np.arange(-px, W - px)
    grid_y, grid_x = np.meshgrid(dy, dx, indexing="ij")
    dy = grid_y.ravel()
    dx = grid_x.ravel()
    vals = I[py + dy, px + dx].astype(np.int64)
    det = np.abs(dy[:, None] * dx[None, :] - dx[:, None] * dy[None, :])
    # d1 and d2 must be distinct non-zero offsets. Distinct because the tuple
    # must not name a point twice, non-zero for the same reason; either one
    # gives det = 0, so a pair with either is in bin 0 and has to be removed
    # from there rather than by its determinant.
    keep = (
        (det < n_spectrum)
        & ((dy[:, None] != 0) | (dx[:, None] != 0))
        & ((dy[None, :] != 0) | (dx[None, :] != 0))
        & ~np.eye(len(dy), dtype=bool)
    )
    weights = (3 * vals[:, None] * vals[None, :])[keep]
    column = np.bincount(det[keep], weights=weights, minlength=n_spectrum)
    return [int(v) for v in column[:n_spectrum]]


class NTTBackend:

    """A descent backend with every quantity on the correlation path.

    Satisfies descent.SpectrumBackend. The spectrum, the row sums and the
    columns are the exact integers of definition.py, reached by correlation, CRT
    and direct offset sums; the gradient is a float vector from numpy's FFT,
    equal to definition.spectrum_gradient up to rounding. The gradient is the
    one part that cannot be exact, because the weights themselves are floats.

    jacobian_column is optional in the protocol but supplied here, which is what
    lets descent step a pixel without recomputing the spectrum: a column is
    O(pixels**2) of small integer arithmetic against the spectrum's O(pixels**2
    log pixels) per CRT pass, and it is exact either way, by trilinearity.
    """

    def __init__(self, pad=None, primes=None, max_value=None):
        self.pad = pad
        self.primes = primes
        self.max_value = max_value
        # descent keeps the current spectrum in hand as it walks -- by
        # trilinearity it is the old spectrum plus the column of the step taken
        # -- so it has no reason to ask for it again here. Kept for callers
        # that do: ask for a spectrum, then ask for the gradient at the same
        # image, and the gradient gets the weights for free. Keyed by id, so it
        # holds one image's worth and nothing grows with the number of steps.
        self._spectrum = None
        self._spectrum_for = None

    def area_spectrum(self, I, max_value=None):
        if max_value is None:
            max_value = self.max_value
        out = area_spectrum_ntt(I, self.pad, self.primes, max_value)
        self._spectrum, self._spectrum_for = out, id(I)
        return out

    def jacobian_row_sums(self, I, max_value=None):
        if max_value is None:
            max_value = self.max_value
        return jacobian_row_sums_ntt(I, self.pad, self.primes, max_value)

    def spectrum_gradient(self, I, target, scale, spectrum=None, max_value=None):
        if spectrum is None and self._spectrum_for == id(I):
            spectrum = self._spectrum
        return spectrum_gradient_ntt(I, target, scale, self.pad, spectrum)

    def jacobian_column(self, I, pixel, max_value=None):
        return jacobian_column_ntt(I, pixel, max_value if max_value is not None
                                   else self.max_value)

