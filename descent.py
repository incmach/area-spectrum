"""Steepest descent on a fixed area-spectrum target, over a pluggable backend.

The search lives here, separately from the spectrum and the gradient it is
driven by. Nothing in this module knows how either is computed: it asks a
backend for a spectrum, a scale, and a gradient, and picks integer steps. That
is the seam a faster implementation drops into without touching the search.

The gradient deliberately stays in definition.py alongside the spectrum and the
Jacobian. It is not a separate kind of computation: J^T w reuses the same
offset enumeration and the same per-bin weights, only weighted instead of
summed, and under a triple-correlation or NTT formulation it is the adjoint of
the very section-selection operator the forward pass uses. Anything that can
speed up the spectrum can therefore speed up the gradient by the same route, so
the two belong together and are optimised together. What is genuinely separate is
the *search* -- the objective, the candidate ordering, and the accept/reject
loop -- none of which cares where the numbers came from.

A backend is any object with three methods:

    area_spectrum(I)                  -> the spectrum, length spectrum_length(I)
    jacobian_row_sums(I)              -> per-bin L1 scale, same length
    spectrum_gradient(I, target, scale) -> flat per-pixel gradient, length I.size

The reference backend wraps the brute-force implementation in definition.py and
is the default, so descent works out of the box against it. A faster
implementation is then a drop-in replacement:

    descent(I, target, backend=FastSpectrum())
"""
from collections import namedtuple
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

import definition
from definition import coordinates


@runtime_checkable
class SpectrumBackend(Protocol):
    """What a spectrum implementation must provide for descent to use it.

    Named after the definition.py functions on purpose, so a faster
    implementation can be validated against this interface and swapped in
    without renaming anything. Kept in terms of operations rather than
    internals, so a backend is free to cache, batch, or run in Rust underneath.
    """

    def area_spectrum(self, I) -> Sequence[int]:
        """The exact area spectrum of I, indexed by bin."""

    def jacobian_row_sums(self, I) -> Sequence[int]:
        """Per-bin L1 row sums of the spectrum's Jacobian."""

    def spectrum_gradient(self, I, target, scale) -> Sequence[float]:
        """Gradient of the normalized residual at I, ordered like I.ravel()."""


@runtime_checkable
class ColumnBackend(Protocol):
    """The optional fast path: exact Jacobian columns, one pixel at a time.

    Split out from SpectrumBackend because it is genuinely optional, and a
    Protocol that isinstance-checks cannot express "has this if you have it".
    A backend satisfying only SpectrumBackend is complete and correct; adding
    this one makes descent an order of magnitude faster without changing a
    single result.

    Given the exact column at a pixel, a unit step's spectrum is a sum rather
    than a fresh spectrum, by trilinearity. See _step_spectrum.
    """

    def jacobian_column(self, I, pixel) -> Sequence[int]:
        """Exact Jacobian column at one pixel, indexed by bin."""


class ReferenceBackend:
    """The brute-force reference in definition.py, as a descent backend.

    Deliberately thin: the spectrum, the row sums and the gradient all stay in
    definition.py, where an NTT-based implementation would replace them. Only
    the descent-facing signatures live here.
    """

    def area_spectrum(self, I, max_value=None):
        return definition.area_spectrum(I)

    def jacobian_row_sums(self, I, max_value=None):
        return definition.jacobian_row_sums(I)

    def spectrum_gradient(self, I, target, scale, spectrum=None):
        return definition.spectrum_gradient(I, target, scale, spectrum)

    def jacobian_column(self, I, pixel, max_value=None):
        return definition.jacobian_column(I, pixel)


REFERENCE = ReferenceBackend()

_BACKEND_METHODS = ("area_spectrum", "jacobian_row_sums", "spectrum_gradient")


def _resolve(backend):
    """Return the backend to use, defaulting to the reference.

    Checked on entry so a backend missing a method fails with a clear message
    here, rather than as an AttributeError deep inside a descent loop.
    """
    if backend is None:
        return REFERENCE
    missing = [m for m in _BACKEND_METHODS if not callable(getattr(backend, m, None))]
    if missing:
        raise TypeError(
            f"backend {type(backend).__name__} is missing "
            f"{', '.join(missing)}; a backend needs "
            f"{', '.join(_BACKEND_METHODS)}"
        )
    return backend


def _with_max_value(fn, backend, args, max_value):
    """Call a backend method, offering max_value if it will take one.

    descent already knows the value range it will confine the search to -- it is
    the max_value the steps are clamped to -- and a backend that reconstructs
    exact integers by CRT has to pick its primes from some bound on the pixels.
    Left to guess from the dtype, an int64 image is charged for 2**63 and gets
    fifteen to twenty passes where five would do. Passing the bound down is
    exact, not an approximation: the true values are smaller than the bound, so
    fewer primes still reconstruct them.

    Optional on purpose. A backend that does not declare a max_value is asked
    without one, so existing implementations keep working; the TypeError comes
    from the call itself, and only for the keyword this adds.
    """
    if max_value is None:
        return fn(*args)
    try:
        return fn(*args, max_value=max_value)
    except TypeError:
        # Only the keyword is retried without; a TypeError from inside the call
        # would be swallowed here, which is the one cost of staying compatible
        # with backends written against the three-method protocol. The bound is
        # an optimization, so a backend that rejects it must not fail the run.
        return fn(*args)


def _step_spectrum(spectrum, I, pixel, step, backend, max_value=None):
    """The spectrum of I after one unit step, by trilinearity.

        area_spectrum(I + step * e_p) == area_spectrum(I) + step * jacobian_column(I, p)

    This is the identity, not a linearisation. Every term of the spectrum is a
    product of *distinct* pixel values, so no term contains a square of the pixel
    being moved and the spectrum is exactly linear in it; a unit step therefore
    moves bin k by exactly the column, and the sum is the spectrum of the trial
    image as an integer vector. The result is fed to the same
    spectrum_residual_loss as a measured spectrum, so a candidate is scored by
    its exact loss either way.

    Falls back to measuring the trial image when the backend has no column, so a
    backend written before this optimization still works -- it just pays for the
    spectrum it could have added.
    """
    column = getattr(backend, "jacobian_column", None)
    if column is None:
        trial = np.array(I, copy=True)
        trial[pixel] += step
        return _with_max_value(backend.area_spectrum, backend, (trial,), max_value)
    c = _with_max_value(column, backend, (I, pixel), max_value)
    return [a + step * v for a, v in zip(spectrum, c)]


def spectrum_residual_loss(spectrum, target, scale):
    """Sum of squared normalized residuals, over the bins the image can reach.

    L(I) = sum_{scale[k] > 0} ((area_spectrum(I)[k] - target[k]) / scale[k])**2

    Bins the image cannot reach are skipped rather than divided by, so a target
    asking for one of them is a no-op instead of a division by zero. The zero
    test here is the same one definition.spectrum_residual_weights applies, so
    the loss and the gradient cannot drift apart about which bins count.

    This is the value descent minimises. It takes a spectrum rather than an
    image so that a measured spectrum and one predicted by trilinearity are
    reduced by identical arithmetic -- comparing two numbers that went through
    the same expression is what makes the comparison meaningful. Both are
    exact integers before the division, so a predicted spectrum gives the
    candidate's exact loss, not an estimate of it.
    """
    return sum(
        ((a - t) / s) ** 2
        for a, t, s in zip(spectrum, target, scale)
        if s > 0
    )


def spectrum_loss(I, target, scale, backend=None, spectrum=None, max_value=None):
    """The loss of an image, from its spectrum.

    spectrum is the spectrum of I if the caller already has it; supplying it
    saves the backend a spectrum pass. max_value is the bound the image's
    pixels are known to respect, offered to the backend so an exact-integer
    one can size its arithmetic from the real range instead of the dtype. It is
    the value descent minimises.
    """
    backend = _resolve(backend)
    if spectrum is None:
        spectrum = _with_max_value(backend.area_spectrum, backend, (I,), max_value)
    return spectrum_residual_loss(spectrum, target, scale)


def ranked_gradient_steps(I, target, scale, min_value=0, max_value=255, backend=None,
                          spectrum=None):
    """Every feasible (pixel, step) the gradient would allow, best first.

    For a step s in {-1, +1} at pixel p the loss changes by s * dL/dI[p], so the
    ordering key is -s * gradient[p]. Returned best first, which is what
    integer_gradient_direction takes the head of.

    Pixels pinned at a bound in the direction they want are omitted rather than
    clamped, since stepping the other way raises the loss.

    Pass the spectrum of I when the caller has it; the gradient needs it for the
    residual weights and it is the expensive half of that call.
    """
    backend = _resolve(backend)
    # Two optional keywords, both of which a backend written against the
    # three-method protocol will not accept: the spectrum, so the gradient does
    # not re-derive one descent already holds, and the value bound, so it does
    # not size its arithmetic from the dtype. Each is dropped independently
    # rather than together, so a backend that takes one still gets it.
    try:
        gradient = backend.spectrum_gradient(I, target, scale, spectrum, max_value)
    except TypeError:
        try:
            gradient = backend.spectrum_gradient(I, target, scale, spectrum)
        except TypeError:
            gradient = backend.spectrum_gradient(I, target, scale)
    candidates = []
    for p, g in zip(coordinates(I), gradient):
        if g == 0:
            continue
        value = int(I[p])
        for step in (-1, 1):
            if min_value <= value + step <= max_value:
                candidates.append((-step * g, p, step))
    candidates.sort(key=lambda c: -c[0])
    return [(p, step) for _, p, step in candidates]


def integer_gradient_direction(I, target, scale, min_value=0, max_value=255,
                              backend=None):
    """The single pixel and +-1 step that the gradient most wants to take.

    The head of ranked_gradient_steps, or None when no pixel has room to move in
    a direction the gradient wants. This ranks by the exact change the step
    makes to the spectrum, but the loss is quadratic in the spectrum, so the
    best step by loss is not always the best by gradient. Descent confirms a
    step against the real spectrum before keeping it.
    """
    candidates = ranked_gradient_steps(I, target, scale, min_value, max_value, backend)
    return candidates[0] if candidates else None


Descent = namedtuple("Descent", "image loss accepted_steps history stopped_early")


def iter_descent(I, target, scale=None, max_steps=1000, patience=25,
                 min_value=0, max_value=255, width=1, backend=None):
    """descent, yielding a snapshot of the run after every step.

    Yields a Descent once before the first step and once after every accepted
    step, so a caller that is drawing progress can read the running loss, the
    accepted-step count and the history as the run proceeds rather than waiting
    for a result it only sees at the end.     The last value yielded is the result,
    and the generator then returns.

    Yields a snapshot per accepted step, plus one more if patience runs out.
    That last one repeats the previous state rather than advancing, because
    patience costs no step: it is the same image reported a second way, saying
    why the run stopped. A caller counting yields therefore gets
    accepted_steps + 1, or + 2 if the run ended on patience.

    Each yielded image is a copy of the working image at that point, so a caller
    may hold on to a snapshot; the search keeps mutating its own.

    Cancelling a run is done by not asking for the next value. Nothing in the
    search needs to know, and the last snapshot is the state at the moment of
    cancellation. That is deliberately *not* the same thing as stopped_early,
    which means patience ran out and is reported by the generator itself -- a
    target can be attainable and still missed this way, which is a different
    fact from a user losing interest, and a caller showing a summary needs to
    keep them apart.

    descent() is this generator drained to its last value, so the search has
    one implementation rather than two that can drift apart.
    """
    backend = _resolve(backend)
    # The pixel range the search will stay inside, offered to the backend as a
    # value bound. A backend that picks CRT primes from the dtype alone is
    # charged for int64's 2**63 and pays for it in passes; the bound is exact
    # because the true values are inside it, and the steps are clamped to it
    # anyway, so the assumption is enforced by construction and not merely
    # expected. None disables it, leaving the backend to its own default.
    value_bound = max_value if max_value is not None else None
    if scale is None:
        scale = _with_max_value(backend.jacobian_row_sums, backend, (I,), value_bound)
    current = np.array(I, dtype=np.int64, copy=True)
    # The spectrum of the current image, kept in hand for the whole run. The
    # spectrum is trilinear, so a one-pixel step moves it by exactly the
    # Jacobian column at that pixel: the next spectrum is this one plus the
    # step's column, an exact integer either way. Carrying it forward is what
    # turns one spectrum per candidate into one spectrum per run, and it stays
    # exact over an arbitrary number of steps because every step adds integers.
    spectrum = backend.area_spectrum(current)
    loss = spectrum_residual_loss(spectrum, target, scale)
    history = [loss]
    accepted = 0
    stalled = 0
    yield Descent(current.copy(), loss, accepted, list(history), False)
    for _ in range(max_steps):
        candidates = ranked_gradient_steps(
            current, target, scale, min_value, max_value, backend, spectrum
        )
        best = None
        best_move = None
        for pixel, step in candidates[:width]:
            trial_spectrum = _step_spectrum(spectrum, current, pixel, step, backend)
            trial_loss = spectrum_residual_loss(trial_spectrum, target, scale)
            if trial_loss < loss and (best is None or trial_loss < best[0]):
                best = (trial_loss, trial_spectrum)
                best_move = (pixel, step)
        if best is None:
            stalled += 1
            if stalled >= patience:
                yield Descent(current.copy(), loss, accepted, list(history), True)
                return
            continue
        loss, spectrum = best
        # Replay the accepted move on the image itself. The spectrum carried
        # forward is the same one the trial was scored from, so the two cannot
        # disagree about which pixel moved -- but the image is the thing
        # returned, and it is derived here rather than carried alongside.
        current[best_move[0]] += best_move[1]
        accepted += 1
        stalled = 0
        history.append(loss)
        yield Descent(current.copy(), loss, accepted, list(history), False)


def descent(I, target, scale=None, max_steps=1000, patience=25,
            min_value=0, max_value=255, width=1, backend=None):
    """Reduce the discrepancy between the image's spectrum and a fixed target,
    moving one pixel at a time and keeping only moves that actually help.

    Each iteration takes the width best-ranked steps from ranked_gradient_steps
    and keeps the one that most reduces the loss. The spectrum is trilinear in
    the pixels, so the gradient gives a step's effect on the spectrum exactly,
    and its top-ranked step is the single best move available: measured on a 4x4,
    the head pick and the best available step both decrease the loss by 0.1535.
    The repeated-point spectrum this replaced was not so lucky -- its top pick
    gave 0.061 where 0.153 was available.

    Checking width candidates and keeping the best is therefore insurance
    against a mis-ranked step rather than a correction for nonlinearity, and
    width=1 follows the gradient exactly. Raising width costs that many column
    evaluations per accepted step and typically reaches a lower loss in fewer
    steps; a width covering every feasible step makes each move exactly the best
    available one.

    A candidate is scored by its exact loss, not an estimate: the spectrum of a
    one-pixel step is the current spectrum plus the step's Jacobian column, which
    is the identity rather than a first-order approximation, so scoring it costs
    a column and not a spectrum. The spectrum is therefore computed once for
    the run and carried forward a column at a time, instead of once per candidate.

    What that changes is where the exactness of a step's score comes from. It
    used to come from measuring each trial image's spectrum with an independent
    call; it now comes from the column being exact, and the gradient that ranked
    the step is derived from the same trilinearity. Both routes agree -- the
    tests assert descent produces the identical image, loss and history either
    way -- but a backend whose columns disagree with its spectrum would now be
    able to talk itself into a losing step, where the measured route would have
    caught it. Hide jacobian_column to get the measured route back.

    Because the winner is chosen by exact loss, the returned history is
    non-increasing by construction.

    Two stopping conditions, per T0.1-06: max_steps, and patience consecutive
    iterations in which no candidate improved. The second is a local-minimum
    test, and it is reported in stopped_early so the caller can widen the search
    or raise patience rather than assume optimality. A target can be attainable
    and still unreachable this way; a larger edit is a longer path than a unit
    step can always cover.

    scale defaults to the backend's row sums at the starting image and is held
    fixed for the whole run, which is what keeps the target's meaning stable
    across steps. backend defaults to the reference implementation in
    definition.py. Returns a Descent with the final image, its loss, the number
    of steps accepted, the loss history, and whether patience ran out.

    This blocks until the run is over. A caller that wants to watch a run, or to
    cancel one, should drive iter_descent instead, which yields the same
    snapshots and stops when the caller stops asking.
    """
    final = None
    for final in iter_descent(I, target, scale, max_steps, patience,
                              min_value, max_value, width, backend):
        pass
    return final

