"""
Tests for per_coordinate.py.

Covers scaled_objective() and per_coordinate_descent().

Runnable directly (python test_per_coordinate.py) or via pytest.
All images are small (<= 8x8); the algorithms are intentionally naive.
"""

import math

import numpy as np

import definition
import per_coordinate

SCALED_OBJECTIVE = per_coordinate.scaled_objective
DESCENT = per_coordinate.per_coordinate_descent
AREA_SPECTRUM = definition.area_spectrum


# ---------------------------------------------------------------------------
# scaled_objective
# ---------------------------------------------------------------------------

def test_scaled_objective_zero_at_target():
    rng = np.random.default_rng(1)
    for shape in [(2, 2), (3, 2)]:
        I = rng.integers(1, 6, size=shape).astype(np.uint8)
        assert SCALED_OBJECTIVE(I, AREA_SPECTRUM(I)) == 0.0


def test_scaled_objective_nonnegative():
    I = np.ones((3, 3), dtype=np.uint8)
    target = [1] * I.size
    assert SCALED_OBJECTIVE(I, target) >= 0.0


def test_scaled_objective_strictly_positive_away_from_target():
    I = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    target = [0] * I.size
    assert SCALED_OBJECTIVE(I, target) > 0.0


def test_scaled_objective_matches_definition():
    I = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    target = [5, 5, 5, 5]
    assert SCALED_OBJECTIVE(I, target) == definition.target_function_squared_norm(
        I, target
    )


# ---------------------------------------------------------------------------
# per_coordinate_descent
# ---------------------------------------------------------------------------

def test_descent_terminal_objective_not_increasing():
    rng = np.random.default_rng(2)
    for _ in range(5):
        I = rng.integers(1, 5, size=(3, 3)).astype(np.uint8)
        target = rng.integers(1, 10, size=I.size)
        final = DESCENT(I, target, max_iter=20)
        assert SCALED_OBJECTIVE(final, target) <= SCALED_OBJECTIVE(I, target)


def test_descent_history_has_expected_fields():
    I = np.ones((3, 3), dtype=np.uint8)
    target = [1] * I.size
    final, history = DESCENT(I, target, max_iter=5, record=True)
    assert isinstance(history, list)
    for entry in history:
        assert set(entry) == {"iter", "F", "coord", "old", "new", "grad"}
        assert entry["iter"] >= 0
        assert entry["F"] >= 0.0
        assert tuple(int(c) for c in entry["coord"]) in np.ndindex(I.shape)
        assert abs(entry["new"] - entry["old"]) == 1


def test_descent_history_objective_strictly_decreasing():
    I = np.ones((3, 3), dtype=np.uint8)
    target = [1] * I.size
    final, history = DESCENT(I, target, max_iter=50, record=True)
    Fs = [entry["F"] for entry in history]
    assert all(b < a for a, b in zip(Fs, Fs[1:]))


def test_descent_no_more_than_max_iter():
    I = np.ones((4, 4), dtype=np.uint8)
    target = [1] * I.size
    for max_iter in (0, 1, 3):
        final, history = DESCENT(I, target, max_iter=max_iter, record=True)
        assert len(history) <= max_iter
    # zero max_iter -> nothing happens
    final, history = DESCENT(I, target, max_iter=0, record=True)
    assert np.array_equal(final, I)
    assert history == []


def test_descent_at_target_is_fixed_point():
    rng = np.random.default_rng(4)
    I = rng.integers(1, 5, size=(3, 3)).astype(np.uint8)
    target = AREA_SPECTRUM(I)
    final, history = DESCENT(I, target, max_iter=10, record=True)
    assert np.array_equal(final, I)
    assert history == []


def test_descent_moves_one_coordinate_at_a_time():
    I = np.ones((3, 3), dtype=np.uint8)
    target = [1] * I.size
    final, history = DESCENT(I, target, max_iter=4, record=True)
    if history:
        # Replay the recorded steps and verify each differs by exactly 1 unit.
        replay = I.astype(int).copy()
        for entry in history:
            c = tuple(int(x) for x in entry["coord"])
            diff = replay.copy()
            diff[c] = entry["new"]
            changed = diff - replay
            assert np.count_nonzero(changed) == 1
            assert abs(int(changed[c])) == 1
            replay = diff
        assert np.array_equal(replay, final)


def test_descent_respects_lower_bound():
    I = np.ones((3, 3), dtype=np.uint8)
    target = [0] * I.size
    final = DESCENT(I, target, max_iter=20, lower_bound=0, upper_bound=1)
    assert np.all(final >= 0)
    assert np.all(final <= 1)


def test_descent_respects_upper_bound():
    I = np.ones((3, 3), dtype=np.uint8)
    target = [1000] * I.size
    final = DESCENT(I, target, max_iter=20, lower_bound=0, upper_bound=1)
    assert np.all(final >= 0)
    assert np.all(final <= 1)


def test_descent_check_decrease_false_keeps_going():
    # With check_decrease=False every step is accepted, so the objective
    # may temporarily rise; descent still runs to max_iter unless a
    # coordinate hits a zero gradient / bound.
    I = np.ones((3, 3), dtype=np.uint8)
    target = [0] * I.size
    final, history = DESCENT(
        I, target, max_iter=8, check_decrease=False, record=True
    )
    assert len(history) == 8          # all accepted
    # single-unit moves only
    for entry in history:
        assert abs(entry["new"] - entry["old"]) == 1


def test_descent_return_type():
    I = np.ones((2, 2), dtype=np.uint8)
    target = [1] * I.size
    final = DESCENT(I, target, max_iter=3)
    assert isinstance(final, np.ndarray)
    assert final.shape == I.shape
    assert np.issubdtype(final.dtype, np.integer)


def test_descent_record_flag_toggles():
    I = np.ones((2, 2), dtype=np.uint8)
    target = [1] * I.size
    no_hist = DESCENT(I, target, max_iter=3, record=False)
    assert isinstance(no_hist, np.ndarray)
    with_hist = DESCENT(I, target, max_iter=3, record=True)
    assert isinstance(with_hist, tuple)
    assert len(with_hist) == 2


def test_descent_handles_list_input():
    I = [[1, 1], [1, 1]]
    target = [1, 1, 1, 1]
    final = DESCENT(I, target, max_iter=3)
    assert isinstance(final, np.ndarray)


def test_invariant_1x1_and_target_constant():
    # Single pixel: both spectrum and scaling factor are determined by
    # I[0,0]^3, so the scaled objective has a unique minimizer.
    I = np.array([[4]], dtype=np.uint8)
    target = [9]
    # at the minimum the gradient is zero, so descent is a fixed point
    final, history = DESCENT(I, target, max_iter=1, record=True)
    # I[0,0]**3 == target at the minimizer; check objective monotonically falls
    assert SCALED_OBJECTIVE(final, target) <= SCALED_OBJECTIVE(I, target)


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
    print("all per_coordinate tests passed")