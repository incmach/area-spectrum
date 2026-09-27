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


class ReferenceBackend:
    """The brute-force reference in definition.py, as a descent backend.

    Deliberately thin: the spectrum, the row sums and the gradient all stay in
    definition.py, where an NTT-based implementation would replace them. Only
    the descent-facing signatures live here.
    """

    def area_spectrum(self, I):
        return definition.area_spectrum(I)

    def jacobian_row_sums(self, I):
        return definition.jacobian_row_sums(I)

    def spectrum_gradient(self, I, target, scale):
        return definition.spectrum_gradient(I, target, scale)


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


def spectrum_loss(I, target, scale, backend=None):
    """Sum of squared normalized residuals, over the bins the image can reach.

    L(I) = sum_{scale[k] > 0} ((area_spectrum(I)[k] - target[k]) / scale[k])**2

    Bins the image cannot reach are skipped rather than divided by, so a target
    asking for one of them is a no-op instead of a division by zero. The zero
    test here is the same one definition.spectrum_residual_weights applies, so
    the loss and the gradient cannot drift apart about which bins count.

    This is the value descent minimises, and it calls the backend once per
    evaluation -- the hot path for the whole search. A backend that caches
    between calls makes descent proportionally faster.
    """
    backend = _resolve(backend)
    return sum(
        ((a - t) / s) ** 2
        for a, t, s in zip(backend.area_spectrum(I), target, scale)
        if s > 0
    )


def ranked_gradient_steps(I, target, scale, min_value=0, max_value=255, backend=None):
    """Every feasible (pixel, step) the gradient would allow, best first.

    For a step s in {-1, +1} at pixel p the loss changes by s * dL/dI[p], so the
    ordering key is -s * gradient[p]. Returned best first, which is what
    integer_gradient_direction takes the head of.

    Pixels pinned at a bound in the direction they want are omitted rather than
    clamped, since stepping the other way raises the loss.
    """
    backend = _resolve(backend)
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


def descent(I, target, scale=None, max_steps=1000, patience=25,
            min_value=0, max_value=255, width=1, backend=None):
    """Reduce the discrepancy between the image's spectrum and a fixed target,
    moving one pixel at a time and keeping only moves that actually help.

    Each iteration takes the width best-ranked steps from ranked_gradient_steps
    and keeps the one that most reduces the true integer-spectrum loss. The
    spectrum is trilinear in the pixels, so the gradient gives a step's effect
    on the spectrum exactly, and its top-ranked step is the single best move
    available: measured on a 4x4, the head pick and the best available step
    both decrease the loss by 0.1535. The repeated-point spectrum this replaced
    was not so lucky -- its top pick gave 0.061 where 0.153 was available.

    Checking width candidates and keeping the best is therefore insurance
    against a mis-ranked step rather than a correction for nonlinearity, and
    width=1 follows the gradient exactly. Raising width costs that many spectrum
    evaluations per accepted step and typically reaches a lower loss in fewer
    steps; a width covering every feasible step makes each move exactly the best
    available one.

    Because the winner is chosen by measured loss, the returned history is
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
    """
    backend = _resolve(backend)
    if scale is None:
        scale = backend.jacobian_row_sums(I)
    current = np.array(I, dtype=np.int64, copy=True)
    loss = spectrum_loss(current, target, scale, backend)
    history = [loss]
    accepted = 0
    stalled = 0
    for _ in range(max_steps):
        candidates = ranked_gradient_steps(
            current, target, scale, min_value, max_value, backend
        )
        best = None
        for pixel, step in candidates[:width]:
            trial = current.copy()
            trial[pixel] += step
            trial_loss = spectrum_loss(trial, target, scale, backend)
            if trial_loss < loss and (best is None or trial_loss < best[0]):
                best = (trial_loss, trial)
        if best is None:
            stalled += 1
            if stalled >= patience:
                return Descent(current, loss, accepted, history, True)
            continue
        loss, current = best
        accepted += 1
        stalled = 0
        history.append(loss)
    return Descent(current, loss, accepted, history, False)
