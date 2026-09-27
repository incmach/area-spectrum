"""Tests for the descent search in descent.py.

The spectrum, the Jacobian and the gradient are covered in test_definition.py;
this file covers the objective they feed, the candidate ranking, and the
accept/reject loop. The backend tests at the end are the reason the search lives
in its own module: they check it against an implementation that shares no code
with definition.py.
"""
import itertools
import math

import numpy as np
import pytest

import definition
import descent
from definition import area_spectrum, jacobian_row_sums, spectrum_gradient
from descent import (
    SpectrumBackend,
    REFERENCE,
    ReferenceBackend,
    descent as run_descent,
    integer_gradient_direction,
    ranked_gradient_steps,
    spectrum_loss,
)

VOLUME = definition.volume
AREA_SPECTRUM = area_spectrum
ROW_SUMS = jacobian_row_sums
GRADIENT = definition.normalized_spectrum_gradient
SPEC_GRADIENT = spectrum_gradient
LOSS = spectrum_loss
DIRECTION = integer_gradient_direction
DESCENT = run_descent


def _reference_rows(I):
    """Full Jacobian rows: rows[k][p] = d area_spectrum[k] / d I[p].

    Independent of definition.py: accumulates the product rule per pixel into
    a dict instead of using prefix/suffix products. Like area_spectrum, the
    tuple's points are distinct.
    """
    coords = list(np.ndindex(I.shape))
    d = len(I.shape)
    rows = [dict() for _ in range(definition.spectrum_length(I))]
    for S in itertools.permutations(coords, d + 1):
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
    is the derivative of. Like area_spectrum, the tuple's points are distinct.
    """
    coords = list(np.ndindex(I.shape))
    d = len(I.shape)
    result = [0.0] * definition.spectrum_length(I)
    for S in itertools.permutations(coords, d + 1):
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
# spectrum_loss / spectrum_gradient
# ---------------------------------------------------------------------------

def test_spectrum_loss_is_zero_at_its_target():
    rng = np.random.default_rng(73)
    for shape in [(3, 3), (4, 4), (2, 2, 2)]:
        I = rng.integers(1, 5, size=shape).astype(np.int64)
        scale = ROW_SUMS(I)
        assert LOSS(I, AREA_SPECTRUM(I), scale) == 0.0, shape


def test_spectrum_loss_ignores_unreachable_bins():
    """scale[k] == 0 means the bin cannot move, so its target is meaningless."""
    I = np.ones((4, 4), dtype=np.int64)
    scale = ROW_SUMS(I)
    natural = (4 - 1) * (4 - 1)
    n = definition.spectrum_length(I)
    reachable = AREA_SPECTRUM(I)
    unreachable = list(reachable)
    unreachable[natural + 1] = 10 ** 6
    assert LOSS(I, unreachable, scale) == LOSS(I, reachable, scale)


def test_spectrum_loss_equals_squared_normalized_residual():
    rng = np.random.default_rng(79)
    I = rng.integers(1, 5, size=(3, 3)).astype(np.int64)
    scale = ROW_SUMS(I)
    target = AREA_SPECTRUM(rng.integers(1, 5, size=(3, 3)).astype(np.int64))
    expected = sum(
        ((a - t) / s) ** 2 for a, t, s in zip(AREA_SPECTRUM(I), target, scale) if s > 0
    )
    assert math.isclose(LOSS(I, target, scale), expected, rel_tol=1e-12)


def test_spectrum_gradient_matches_finite_difference():
    """Central differences of the float relaxation, in 2d and 3d."""
    rng = np.random.default_rng(83)
    for shape in [(3, 3), (2, 2, 2)]:
        I = rng.integers(1, 5, size=shape).astype(np.int64)
        scale = ROW_SUMS(I)
        target = [float(x) for x in rng.integers(0, 40000, size=len(scale))]
        grad = SPEC_GRADIENT(I, target, scale)
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
            assert math.isclose(fd, grad[i], rel_tol=1e-6, abs_tol=1e-6), (shape, p, fd, grad[i])


def test_spectrum_gradient_reduces_to_normalized_spectrum_gradient():
    """The two are the same function on a target of the delta form.

    normalized_spectrum_gradient exists only for target = A + delta*scale, where
    the residual is pinned to -delta. With a general target the weight is
    2*(A - target)/scale**2 instead.
    """
    rng = np.random.default_rng(89)
    for shape in [(3, 3), (2, 2, 2)]:
        I = rng.integers(1, 5, size=shape).astype(np.int64)
        scale = ROW_SUMS(I)
        A = AREA_SPECTRUM(I)
        delta = [float(x) for x in rng.normal(size=len(A))]
        target = [a + d * s for a, d, s in zip(A, delta, scale)]
        general = SPEC_GRADIENT(I, target, scale)
        special = GRADIENT(I, delta)
        for a, b in zip(general, special):
            assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (shape, a, b)


def test_spectrum_gradient_on_zero_image_is_zero():
    I = np.zeros((3, 3), dtype=np.int64)
    scale = ROW_SUMS(I)
    target = AREA_SPECTRUM(np.ones((3, 3), dtype=np.int64))
    assert SPEC_GRADIENT(I, target, scale) == [0.0] * I.size


# ---------------------------------------------------------------------------
# integer_gradient_direction
# ---------------------------------------------------------------------------

def test_direction_is_ranked_by_effect_on_the_spectrum():
    """The head is at worst a shade off the best step.

    The spectrum is trilinear, so the gradient gives a step's effect on it
    exactly. The loss is quadratic in the spectrum, so ranking by residual
    change and ranking by loss drop can disagree; when they do the head is
    still within a few percent. It is not guaranteed to improve at all, so
    descent measures candidates rather than trusting the order.
    """
    rng = np.random.default_rng(97)
    for shape in [(3, 3), (4, 4)]:
        for _ in range(3):
            I = rng.integers(0, 4, size=shape).astype(np.int64)
            scale = ROW_SUMS(I)
            target = AREA_SPECTRUM(rng.integers(0, 4, size=shape).astype(np.int64))
            ranked = ranked_gradient_steps(I, target, scale)
            if len(ranked) < 4:
                continue
            base = LOSS(I, target, scale)

            def decrease(pixel, step):
                trial = I.copy()
                trial[pixel] += step
                return base - LOSS(trial, target, scale)

            best = max(decrease(*m) for m in ranked)
            head = decrease(*ranked[0])
            assert head >= 0.9 * best, (shape, head, best)


def test_head_is_almost_always_the_best_step():
    """Trilinearity makes the gradient's head the best step nearly always.

    The gradient is the exact change in the residual and the loss is quadratic
    in the spectrum, so the head is exactly best whenever ranking by residual
    change agrees with ranking by loss drop. Measured over many cases it
    agrees the large majority of the time; the rare disagreement is small.
    What is *not* safe is trusting it: the head can come out a net increase
    when the best step only gains a little, which is why descent measures each
    candidate against the loss rather than accepting the head.
    """
    rng = np.random.default_rng(0)
    exact = 0
    total = 0
    for shape in [(3, 3), (4, 4), (5, 4)]:
        for _ in range(20):
            I = rng.integers(0, 4, size=shape).astype(np.int64)
            scale = ROW_SUMS(I)
            target = AREA_SPECTRUM(rng.integers(0, 4, size=shape).astype(np.int64))
            ranked = ranked_gradient_steps(I, target, scale)
            if len(ranked) < 4:
                continue
            base = LOSS(I, target, scale)

            def decrease(pixel, step):
                trial = I.copy()
                trial[pixel] += step
                return base - LOSS(trial, target, scale)

            best = max(decrease(*m) for m in ranked)
            if best <= 0:
                continue
            head = decrease(*ranked[0])
            total += 1
            exact += math.isclose(head, best, rel_tol=1e-9)
    assert total >= 40, total
    # agreeing the large majority of the time, and never wildly off
    assert exact > 0.75 * total, (exact, total)


def test_direction_step_respects_bounds():
    """A pixel already at a bound must be stepped inward or not at all."""
    # 255 is not fully pinned, so it can step down; 0 is, and a target wanting
    # more light than a black image can give leaves nothing feasible at all.
    for value, has_move in ((0, False), (255, True)):
        I = np.full((3, 3), value, dtype=np.int64)
        scale = ROW_SUMS(I)
        target = AREA_SPECTRUM(np.full((3, 3), 255 - value, dtype=np.int64))
        proposal = DIRECTION(I, target, scale)
        assert (proposal is not None) is has_move, value
        if proposal is None:
            continue
        pixel, step = proposal
        assert 0 <= int(I[pixel]) + step <= 255, (value, pixel, step)


def test_direction_is_none_when_saturated():
    """Every pixel pinned at 0 with a target that wants less gives no move."""
    I = np.zeros((3, 3), dtype=np.int64)
    scale = ROW_SUMS(I)
    assert DIRECTION(I, AREA_SPECTRUM(np.zeros((3, 3), dtype=np.int64)), scale) is None


def test_direction_is_none_at_its_target():
    rng = np.random.default_rng(101)
    I = rng.integers(1, 5, size=(3, 3)).astype(np.int64)
    scale = ROW_SUMS(I)
    assert DIRECTION(I, AREA_SPECTRUM(I), scale) is None


def test_direction_is_a_unit_step():
    rng = np.random.default_rng(103)
    for shape in [(3, 3), (4, 4)]:
        I = rng.integers(1, 5, size=shape).astype(np.int64)
        scale = ROW_SUMS(I)
        target = AREA_SPECTRUM(rng.integers(0, 5, size=shape).astype(np.int64))
        pixel, step = DIRECTION(I, target, scale)
        assert step in (-1, 1)
        assert I[pixel] + step != I[pixel]


# ---------------------------------------------------------------------------
# descent
# ---------------------------------------------------------------------------

def test_descent_is_monotone_and_improving():
    rng = np.random.default_rng(107)
    for shape in [(3, 3), (4, 4)]:
        I = rng.integers(0, 6, size=shape).astype(np.int64)
        scale = ROW_SUMS(I)
        target = AREA_SPECTRUM(rng.integers(0, 6, size=shape).astype(np.int64))
        r = DESCENT(I, target, scale, max_steps=300, patience=25)
        assert all(b <= a for a, b in zip(r.history, r.history[1:])), shape
        assert r.loss <= r.history[0]
        assert len(r.history) == r.accepted_steps + 1, shape


def test_descent_recovers_a_single_pixel_edit():
    """A +-1 edit is one gradient step away, so descent should find it.

    Held to a unit edit deliberately. A larger edit is a longer path and can
    sit in a basin the unit-step neighbourhood cannot leave: on the old
    definition a +3 edit on a 4x4 was recoverable, and under the distinct
    points spectrum it is not, at any width. That is the local-minimum
    behaviour patience is meant to report, not a regression.
    """
    rng = np.random.default_rng(109)
    I = rng.integers(1, 6, size=(4, 4)).astype(np.int64)
    edited = I.copy()
    edited[2, 1] += 1
    r = DESCENT(I, AREA_SPECTRUM(edited), ROW_SUMS(I), max_steps=300, patience=25, width=8)
    assert r.loss == 0.0
    assert (r.image == edited).all()


def test_descent_recovers_unit_edits_across_seeds():
    """The same round trip, over enough seeds to catch a systematic failure."""
    recovered = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        I = rng.integers(1, 6, size=(4, 4)).astype(np.int64)
        edited = I.copy()
        edited[int(rng.integers(0, 4)), int(rng.integers(0, 4))] += 1
        r = DESCENT(I, AREA_SPECTRUM(edited), ROW_SUMS(I), max_steps=200,
                    patience=25, width=8)
        recovered += r.loss == 0.0
    assert recovered >= 18, recovered


def test_descent_reports_a_local_minimum_it_cannot_escape():
    """A larger edit can be unreachable; descent must say so, not claim success.

    Guards the honesty of stopped_early: the target is attainable (its loss is
    0) yet the unit-step neighbourhood of the stalled image has no improving
    move, so the run must end above 0 and report the stall.
    """
    rng = np.random.default_rng(109)
    I = rng.integers(0, 6, size=(4, 4)).astype(np.int64)
    edited = I.copy()
    edited[2, 1] += 3
    target = AREA_SPECTRUM(edited)
    scale = ROW_SUMS(I)
    assert LOSS(edited, target, scale) == 0.0, "target must be attainable"
    r = DESCENT(I, target, scale, max_steps=300, patience=25, width=8)
    assert r.loss > 0.0
    assert r.stopped_early


def test_descent_respects_bounds():
    """Pinned pixels must not be pushed past the clamp, even under pressure."""
    for value in (0, 255):
        I = np.full((3, 3), value, dtype=np.int64)
        scale = ROW_SUMS(I)
        target = AREA_SPECTRUM(np.full((3, 3), 255 - value, dtype=np.int64))
        r = DESCENT(I, target, scale, max_steps=200, patience=25)
        assert r.image.min() >= 0, value
        assert r.image.max() <= 255, value


def test_descent_honours_explicit_bounds():
    I = np.full((3, 3), 10, dtype=np.int64)
    scale = ROW_SUMS(I)
    target = AREA_SPECTRUM(np.full((3, 3), 4, dtype=np.int64))
    r = DESCENT(I, target, scale, max_steps=200, patience=25, min_value=0, max_value=12)
    assert r.image.max() <= 12
    assert r.image.min() >= 0


def test_descent_does_nothing_at_target():
    rng = np.random.default_rng(113)
    I = rng.integers(1, 5, size=(3, 3)).astype(np.int64)
    r = DESCENT(I, AREA_SPECTRUM(I), ROW_SUMS(I), max_steps=50)
    assert r.accepted_steps == 0
    assert r.loss == 0.0
    assert (r.image == I).all()
    assert r.history == [0.0]


def test_descent_respects_max_steps():
    rng = np.random.default_rng(127)
    I = rng.integers(0, 6, size=(4, 4)).astype(np.int64)
    scale = ROW_SUMS(I)
    target = AREA_SPECTRUM(rng.integers(0, 6, size=(4, 4)).astype(np.int64))
    r = DESCENT(I, target, scale, max_steps=3, patience=10 ** 6)
    assert r.accepted_steps <= 3
    assert len(r.history) == r.accepted_steps + 1


def test_descent_reports_early_stop():
    """stopped_early distinguishes a local minimum from a completed run."""
    rng = np.random.default_rng(131)
    I = rng.integers(0, 6, size=(3, 3)).astype(np.int64)
    scale = ROW_SUMS(I)
    target = AREA_SPECTRUM(rng.integers(0, 6, size=(3, 3)).astype(np.int64))
    tight = DESCENT(I, target, scale, max_steps=1000, patience=1)
    assert tight.stopped_early


def test_descent_preserves_shape_and_dtype():
    rng = np.random.default_rng(137)
    I = rng.integers(0, 6, size=(3, 3, 2)).astype(np.int64)
    scale = ROW_SUMS(I)
    target = AREA_SPECTRUM(rng.integers(0, 6, size=(3, 3, 2)).astype(np.int64))
    r = DESCENT(I, target, scale, max_steps=50, patience=10)
    assert r.image.shape == I.shape
    assert r.image.dtype == np.int64


def test_descent_does_not_mutate_input():
    rng = np.random.default_rng(139)
    I = rng.integers(0, 6, size=(3, 3)).astype(np.int64)
    before = I.copy()
    DESCENT(I, AREA_SPECTRUM(np.ones((3, 3), dtype=np.int64)), ROW_SUMS(I), max_steps=20)
    assert (I == before).all()


def test_descent_default_scale_is_row_sums():
    rng = np.random.default_rng(149)
    I = rng.integers(0, 6, size=(3, 3)).astype(np.int64)
    target = AREA_SPECTRUM(rng.integers(0, 6, size=(3, 3)).astype(np.int64))
    assert DESCENT(I, target).image.tolist() == DESCENT(I, target, ROW_SUMS(I)).image.tolist()


# ---------------------------------------------------------------------------
# the backend seam
# ---------------------------------------------------------------------------

class DetBackend:
    """A second implementation, sharing no code with definition.py.

    Built to be obviously correct rather than fast: the spectrum comes from
    explicit 2x2 determinants over distinct pixel quadruples, the row sums from
    finite differences of that, and the gradient by central differences of the
    spectrum itself. Nothing here calls into definition.py, so a disagreement
    with the reference is a real disagreement, not a shared bug.

    The gradient being a finite difference is the point: it is a different
    algorithm that must land on the same answer, which is what an NTT
    implementation will also have to do.
    """

    def __init__(self, h=1):
        self.h = h

    def area_spectrum(self, I):
        d = len(I.shape)
        n = definition.spectrum_length(I)
        out = [0] * n
        points = list(np.ndindex(I.shape))
        for combo in itertools.combinations(points, d + 1):
            for S in itertools.permutations(combo):
                m = np.array([[S[i + 1][j] - S[0][j] for j in range(d)]
                              for i in range(d)], dtype=object)
                det = int(round(np.linalg.det(np.array(
                    [[int(x) for x in row] for row in m], dtype=float))))
                out[abs(det)] += math.prod(int(I[v]) for v in S)
        return out

    def jacobian_row_sums(self, I):
        h = self.h
        base = self.area_spectrum(I)
        total = [0] * len(base)
        for p in np.ndindex(I.shape):
            up = I.copy()
            up[p] += h
            a = self.area_spectrum(up)
            for k in range(len(base)):
                total[k] += (a[k] - base[k]) / h
        return total

    def spectrum_gradient(self, I, target, scale):
        h = 1e-4
        grad = []
        for p in np.ndindex(I.shape):
            up = I.astype(float)
            up[p] += h
            grad.append(
                (self._loss_float(up, target, scale) - self._loss_float(I, target, scale))
                / h
            )
        return grad

    def _loss_float(self, I, target, scale):
        """The objective over float pixels, binning before applying residuals.

        The loss is sum_k ((A[k] - target[k]) / scale[k])**2 over *bins*, with
        A[k] the bin's total. Adding the residual once per tuple instead is a
        different functional and a much larger one, so binning has to come
        first. A float spectrum is required here: the gradient is a finite
        difference, and the int() cast in area_spectrum would make it flat.
        """
        d = len(I.shape)
        bins = [0.0] * definition.spectrum_length(I)
        for combo in itertools.combinations(list(np.ndindex(I.shape)), d + 1):
            for S in itertools.permutations(combo):
                m = np.array([[S[i + 1][j] - S[0][j] for j in range(d)]
                              for i in range(d)], dtype=float)
                k = abs(int(round(np.linalg.det(m))))
                prod = 1.0
                for v in S:
                    prod *= float(I[v])
                bins[k] += prod
        return sum(
            ((value - target[k]) / scale[k]) ** 2
            for k, value in enumerate(bins)
            if k < len(scale) and scale[k] > 0
        )

    def _spectrum_float(self, I):
        d = len(I.shape)
        out = [0.0] * definition.spectrum_length(I)
        for combo in itertools.combinations(list(np.ndindex(I.shape)), d + 1):
            for S in itertools.permutations(combo):
                m = np.array([[S[i + 1][j] - S[0][j] for j in range(d)]
                              for i in range(d)], dtype=float)
                k = abs(int(round(np.linalg.det(m))))
                prod = 1.0
                for v in S:
                    prod *= float(I[v])
                out[k] += prod
        return out


def test_det_backend_agrees_with_the_reference_spectrum():
    """The second implementation must produce the reference's numbers."""
    rng = np.random.default_rng(211)
    det = DetBackend()
    for shape in [(3, 3), (3, 2), (4, 4)]:
        I = rng.integers(1, 5, size=shape).astype(np.int64)
        assert det.area_spectrum(I) == AREA_SPECTRUM(I), shape


def test_det_backend_agrees_with_the_reference_row_sums():
    """Finite-differenced row sums must land on the analytic ones."""
    rng = np.random.default_rng(223)
    det = DetBackend()
    for shape in [(3, 3), (3, 2)]:
        I = rng.integers(1, 5, size=shape).astype(np.int64)
        got = det.jacobian_row_sums(I)
        expected = ROW_SUMS(I)
        for a, b in zip(got, expected):
            assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6), (shape, a, b)


def test_det_backend_gradient_agrees_with_the_reference():
    """A finite-difference gradient must land on the analytic one."""
    rng = np.random.default_rng(227)
    det = DetBackend()
    I = rng.integers(1, 5, size=(3, 3)).astype(np.int64)
    scale = ROW_SUMS(I)
    target = AREA_SPECTRUM(rng.integers(1, 5, size=(3, 3)).astype(np.int64))
    got = det.spectrum_gradient(I, target, scale)
    expected = SPEC_GRADIENT(I, target, scale)
    for a, b in zip(got, expected):
        assert math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-4), (a, b)


def test_descent_runs_against_an_alternative_backend():
    """The search must be a drop-in over a backend that is not the reference.

    Same result as the reference backend, which is the whole claim: the search
    depends on the protocol and not on how the numbers were produced.
    """
    rng = np.random.default_rng(229)
    I = rng.integers(1, 6, size=(3, 3)).astype(np.int64)
    edited = I.copy()
    edited[1, 1] += 1
    target = AREA_SPECTRUM(edited)
    scale = ROW_SUMS(I)
    reference = DESCENT(I, target, scale, max_steps=50, patience=10, width=8)
    other = DESCENT(I, target, scale, max_steps=50, patience=10, width=8,
                    backend=DetBackend())
    assert reference.loss == 0.0
    assert math.isclose(other.loss, reference.loss, rel_tol=1e-6)
    assert (other.image == reference.image).all()


def test_default_backend_is_the_reference():
    rng = np.random.default_rng(233)
    I = rng.integers(1, 5, size=(3, 3)).astype(np.int64)
    scale = ROW_SUMS(I)
    target = AREA_SPECTRUM(rng.integers(1, 5, size=(3, 3)).astype(np.int64))
    assert DESCENT(I, target, scale, max_steps=20, patience=5).image.tolist() == \
        DESCENT(I, target, scale, max_steps=20, patience=5, backend=REFERENCE).image.tolist()


def test_reference_backend_matches_the_module_functions():
    """The adapter must forward, not reimplement."""
    rng = np.random.default_rng(239)
    I = rng.integers(1, 5, size=(3, 3)).astype(np.int64)
    scale = ROW_SUMS(I)
    target = AREA_SPECTRUM(rng.integers(1, 5, size=(3, 3)).astype(np.int64))
    assert REFERENCE.area_spectrum(I) == definition.area_spectrum(I)
    assert REFERENCE.jacobian_row_sums(I) == definition.jacobian_row_sums(I)
    assert REFERENCE.spectrum_gradient(I, target, scale) == \
        definition.spectrum_gradient(I, target, scale)


def test_backends_satisfy_the_protocol():
    assert isinstance(REFERENCE, SpectrumBackend)
    assert isinstance(DetBackend(), SpectrumBackend)


def test_an_incomplete_backend_is_rejected_with_a_clear_error():
    class MissingGradient:
        def area_spectrum(self, I):
            return AREA_SPECTRUM(I)

        def jacobian_row_sums(self, I):
            return ROW_SUMS(I)

    I = np.ones((3, 3), dtype=np.int64)
    with pytest.raises(TypeError, match="spectrum_gradient"):
        LOSS(I, AREA_SPECTRUM(I), ROW_SUMS(I), backend=MissingGradient())
    with pytest.raises(TypeError, match="spectrum_gradient"):
        DESCENT(I, AREA_SPECTRUM(I), ROW_SUMS(I), max_steps=1,
                backend=MissingGradient())


def test_descent_does_not_touch_definition_internals():
    """descent.py must not reach past the backend into the reference.

    Enforced by wrapping the reference in a backend whose methods are the only
    thing available: if the search called area_spectrum directly, replacing the
    backend would change nothing and the two results would diverge.
    """
    class Counting:
        def __init__(self, inner):
            self.inner = inner
            self.calls = {"spectrum": 0, "row_sums": 0, "gradient": 0}

        def area_spectrum(self, I):
            self.calls["spectrum"] += 1
            return self.inner.area_spectrum(I)

        def jacobian_row_sums(self, I):
            self.calls["row_sums"] += 1
            return self.inner.jacobian_row_sums(I)

        def spectrum_gradient(self, I, target, scale):
            self.calls["gradient"] += 1
            return self.inner.spectrum_gradient(I, target, scale)

    rng = np.random.default_rng(241)
    I = rng.integers(1, 6, size=(3, 3)).astype(np.int64)
    target = AREA_SPECTRUM(rng.integers(1, 6, size=(3, 3)).astype(np.int64))
    counting = Counting(REFERENCE)
    DESCENT(I, target, max_steps=10, patience=5, backend=counting)
    assert counting.calls["spectrum"] > 0
    assert counting.calls["row_sums"] == 1, counting.calls
    assert counting.calls["gradient"] > 0
