"""Tests for the editable spectrum and the window that draws it.

The split under test is the one the design rests on: spectra_ui holds the
decisions (what a delta is worth, where a bin is, what target a delta asks for,
what a panel looks like) and view holds the loop and the input mapping. The
first is pure and tested against definition.py's exact arithmetic. The second is
tested by driving Viewer with synthetic mouse and key events, without a display,
because the mapping from a click to a delta is where a bug would be invisible in
a screenshot and obvious in a number.
"""
import cv2
import numpy as np
import pytest

import definition
import descent
import spectra_ui as ui
import view
from definition import area_spectrum, jacobian_row_sums

NEUTRAL = ui.NEUTRAL


@pytest.fixture
def image():
    """A small image with a few reachable bins and a few unreachable ones."""
    return np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]], dtype=np.int64)


@pytest.fixture
def editor(image):
    return ui.Editor(image, window=1.0, backend=descent.ReferenceBackend())


# -- load_image ------------------------------------------------------------


def test_load_image_round_trips_exact_values(tmp_path):
    path = str(tmp_path / "grey.png")
    original = np.array([[0, 1, 127], [128, 254, 255]], dtype=np.uint8)
    assert cv2.imwrite(path, original)
    I = ui.load_image(path)
    assert I.dtype == np.int64
    assert I.shape == original.shape
    # int64, not uint8: a uint8 here would overflow the moment descent added to
    # a pixel, and the round trip would hide it.
    assert np.array_equal(I, original.astype(np.int64))


def test_load_image_reads_colour_as_one_scalar_field(tmp_path):
    path = str(tmp_path / "colour.png")
    colour = np.zeros((2, 2, 3), dtype=np.uint8)
    colour[..., 1] = 200
    assert cv2.imwrite(path, colour)
    I = ui.load_image(path)
    # One scalar field, not three: the spectrum is defined for a single array
    # of pixel values, and a colour image has no one set of them. Grayscale is a
    # luminance weighting, not a channel pick, so what comes out is the same
    # conversion cv2 itself would do.
    assert I.ndim == 2
    assert np.array_equal(I, cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY).astype(np.int64))


def test_load_image_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such image"):
        ui.load_image(str(tmp_path / "absent.png"))


def test_load_image_rejects_a_file_that_is_not_an_image(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("this is not an image")
    with pytest.raises(ValueError, match="could not decode"):
        ui.load_image(str(path))


def test_load_image_max_side_reduces_the_longest_side(tmp_path):
    path = str(tmp_path / "big.png")
    assert cv2.imwrite(path, np.full((40, 80), 100, dtype=np.uint8))
    I = ui.load_image(path, max_side=20)
    assert max(I.shape) == 20
    assert sorted(I.shape) == [10, 20]


def test_load_image_max_side_averages_rather_than_samples(tmp_path):
    # INTER_AREA, so a halving of a constant stays that constant and a halving
    # of a ramp stays a ramp. Nearest-neighbour would alias the ramp into
    # stripes and the user would be editing an image descent never sees.
    path = str(tmp_path / "ramp.png")
    ramp = np.tile(np.arange(8, dtype=np.uint8), (8, 1))
    assert cv2.imwrite(path, ramp)
    I = ui.load_image(path, max_side=4)
    assert I.shape == (4, 4)
    assert np.all(np.diff(I[0]) > 0)


def test_load_image_keeps_the_size_when_under_the_limit(tmp_path):
    path = str(tmp_path / "small.png")
    assert cv2.imwrite(path, np.full((3, 3), 7, dtype=np.uint8))
    assert ui.load_image(path, max_side=64).shape == (3, 3)


# -- the byte the delta draws as -------------------------------------------


def test_zero_delta_is_the_neutral_byte():
    assert ui.delta_bytes([0.0], 1.0)[0] == NEUTRAL


def test_the_window_is_the_swing_in_either_direction():
    # 127 counts below neutral and 128 above, because that is how many byte
    # values there are. The window is the width of the 127-count side, so the
    # bottom of the scale is reached exactly and the top is one count beyond it.
    window = 3.0
    assert ui.delta_bytes(-window, window)[0] == 0
    assert ui.delta_bytes(window, window)[0] == 254
    assert ui.delta_bytes(128 / 127 * window, window)[0] == 255


def test_the_map_is_signed_about_neutral():
    # The same delta draws differently either side of neutral, which is what
    # makes the drawing show the sign of the residual rather than its size.
    assert ui.delta_bytes(-0.5, 1.0)[0] < NEUTRAL < ui.delta_bytes(0.5, 1.0)[0]


def test_the_map_clips_rather_than_wrapping():
    # Clip, so a saturated bin reads as "off the scale" instead of aliasing to
    # the other end and looking like a bin of the opposite sign.
    assert ui.delta_bytes([1e9], 1.0)[0] == 255
    assert ui.delta_bytes([-1e9], 1.0)[0] == 0


def test_doubling_the_window_doubles_the_byte():
    # The window is a scale on the drawing and nothing else, so this is the
    # whole contract of having it as a parameter.
    assert ui.delta_bytes([0.25], 1.0)[0] == ui.delta_bytes([0.5], 2.0)[0]


def test_a_whole_pixel_drag_moves_the_byte_by_exactly_one():
    for pixels in range(-126, 127):
        delta = pixels * ui.delta_per_pixel(1.0)
        assert int(ui.delta_bytes([delta], 1.0)[0]) == NEUTRAL + pixels


def test_delta_for_bytes_inverts_delta_bytes_on_the_pixel_grid():
    for pixels in range(-126, 127):
        drawn = int(ui.delta_bytes([pixels * ui.delta_per_pixel(0.7)], 0.7)[0])
        back = float(ui.delta_for_bytes(drawn - NEUTRAL, 0.7)[0])
        assert int(ui.delta_bytes([back], 0.7)[0]) == drawn


def test_delta_for_bytes_is_exact_on_the_pixel_grid():
    # Not merely close: a drag is a count of pixels, and a delta that drifted
    # within a pixel of where it was put would make the pointer and the drawn
    # byte disagree about what was asked for.
    for pixels in (-100, -1, 0, 1, 7, 100):
        delta = float(ui.delta_for_bytes(pixels, 1.0)[0])
        assert int(ui.delta_bytes(delta, 1.0)[0]) == NEUTRAL + pixels


# -- the target a delta asks for -------------------------------------------


def test_a_zero_delta_asks_for_the_current_spectrum(editor):
    assert np.array_equal(editor.target(), np.asarray(editor.spectrum))


def test_a_delta_asks_for_a_bin_moved_by_delta_times_scale(editor):
    k = 0
    editor.set_delta(k, 1.5)
    assert editor.target()[k] == editor.spectrum[k] + round(1.5 * editor.scale[k])


def test_the_target_is_rounded_to_whole_counts(editor):
    k = 0
    scale = editor.scale[k]
    editor.set_delta(k, 0.5 / scale)
    # 0.5 counts rounds to 0, not to 1: a fraction of a count is not a request
    # the integer spectrum can carry.
    assert editor.target()[k] == editor.spectrum[k]


def test_a_negative_delta_is_not_clamped(editor, image):
    # The user asked for bin k to go below zero. No non-negative image has such
    # a bin, and the honest outcome is a target that says so and a loss that
    # reports the residual, not a target quietly raised to zero.
    k = 0
    editor.set_delta(k, -1e6)
    assert editor.target()[k] < 0


def test_an_unreachable_bin_is_left_alone_in_the_target(editor):
    dead = int(np.flatnonzero(~editor.reachable)[0])
    editor.deltas[dead] = 12.0            # even bypassing the refusal
    assert editor.target()[dead] == editor.spectrum[dead]


def test_the_loss_is_zero_with_no_edits(editor):
    assert editor.loss() == 0.0


def test_the_loss_is_the_residual_of_the_edited_bins(editor):
    k = 0
    editor.set_delta(k, 2.0)
    expected = ((editor.spectrum[k] - editor.target()[k]) / editor.scale[k]) ** 2
    assert editor.loss() == pytest.approx(expected)


def test_the_loss_skips_unreachable_bins(editor):
    # Asking for a bin no pixel can move must not divide by zero, and must not
    # add a term either: the objective is the one descent minimises.
    dead = int(np.flatnonzero(~editor.reachable)[0])
    editor.set_delta(0, 1.0)
    editor.deltas[dead] = 5.0
    assert np.isfinite(editor.loss())


# -- which bins the run is trying to hit ------------------------------------


def test_the_objective_is_masked_to_the_bins_that_were_asked_for(editor):
    # The mechanism that makes a run move at all: a bin nobody asked for is
    # dropped from the objective by zeroing its scale, which is the rule
    # spectrum_residual_loss and spectrum_gradient already use to skip a bin.
    k = int(np.flatnonzero(editor.reachable)[0])
    editor.set_delta(k, 1.0)
    scale = editor.objective_scale
    assert scale[k] == editor.scale[k]
    assert sum(1 for s in scale if s > 0) == 1


def test_nothing_is_asked_for_until_a_bin_is_dragged(editor):
    assert not editor.edited.any()
    assert not editor.objective_scale.any()


def test_dragging_a_bin_back_to_neutral_stops_asking_for_it(editor):
    k = int(np.flatnonzero(editor.reachable)[0])
    editor.set_delta(k, 1.0)
    assert editor.edited[k]
    editor.set_delta(k, 0.0)
    assert not editor.edited[k]
    assert not editor.objective_scale.any()


def test_the_loss_counts_only_the_asked_for_bins(editor):
    # A bin the user did not touch contributes its residual, which is zero, and
    # must contribute nothing: a zero scale removes the term rather than
    # multiplying it by a small weight that still lets it dominate.
    k = int(np.flatnonzero(editor.reachable)[0])
    editor.set_delta(k, 2.0)
    expected = ((editor.spectrum[k] - editor.target()[k]) / editor.scale[k]) ** 2
    assert editor.loss() == pytest.approx(expected)


def test_the_loss_is_zero_when_the_edited_bins_already_match(editor):
    # The target is built from the deltas, so an edit of exactly zero is the
    # current spectrum and the objective is empty rather than 0 times a sum.
    assert editor.loss() == 0.0


def test_a_run_hits_the_bin_that_was_asked_for(image):
    # The regression this masking exists for. One bin asked for, everything else
    # free to move: a single pixel step perturbs the bins nobody asked about far
    # more than it improves the one that was, so without the mask the objective
    # sees pure damage, accepts nothing, and every run ends on step zero.
    editor = ui.Editor(image, window=0.1, backend=descent.ReferenceBackend())
    k = int(np.flatnonzero(editor.reachable)[len(np.flatnonzero(editor.reachable)) // 2])
    editor.set_delta(k, 1.0)
    before = editor.loss()
    result = descent.descent(editor.image, editor.target().tolist(),
                             scale=[int(s) for s in editor.objective_scale],
                             max_steps=200, patience=25,
                             backend=descent.ReferenceBackend())
    after = descent.spectrum_loss(result.image, editor.target().tolist(),
                                  editor.objective_scale,
                                  backend=descent.ReferenceBackend())
    assert after < before / 100
    assert result.accepted_steps > 0


def test_a_run_is_free_to_disturb_the_bins_it_was_not_asked_about(image):
    editor = ui.Editor(image, window=0.1, backend=descent.ReferenceBackend())
    k = int(np.flatnonzero(editor.reachable)[0])
    editor.set_delta(k, 1.0)
    result = descent.descent(editor.image, editor.target().tolist(),
                             scale=[int(s) for s in editor.objective_scale],
                             max_steps=200, patience=25,
                             backend=descent.ReferenceBackend())
    other = int(np.flatnonzero(np.arange(editor.n_bins) != k)[0])
    assert definition.area_spectrum(result.image)[other] != editor.spectrum[other]


# -- where a bin is on screen ----------------------------------------------


def test_a_bin_is_a_four_by_four_cell():
    assert ui.CELL == 4
    x, y = ui.cell_rect(3)
    assert (x, y) == (12, 0)


def test_the_row_wraps_at_1024_pixels_of_cells():
    assert ui.GRID_WIDTH == 1024
    assert ui.BINS_PER_ROW == 256


def test_the_row_wraps_to_the_next_line_after_256_bins():
    assert ui.cell_rect(255) == (255 * ui.CELL, 0)
    assert ui.cell_rect(256) == (0, ui.CELL)


def test_every_bin_is_where_hit_testing_says_it_is(editor):
    for k in range(editor.n_bins):
        x, y = ui.cell_rect(k)
        assert ui.bin_at(x, y, editor.n_bins) == k
        # Anywhere inside the cell, not just its corner.
        assert ui.bin_at(x + ui.CELL - 1, y + ui.CELL - 1, editor.n_bins) == k


def test_a_click_between_cells_lands_on_one_of_them(editor):
    # Forgiving about the last pixel of a cell: a click that does nothing at all
    # is indistinguishable from a broken one.
    assert ui.bin_at(4, 0, editor.n_bins) == 1


def test_a_click_past_the_end_of_the_spectrum_hits_nothing(editor):
    assert ui.bin_at(0, ui.CELL, editor.n_bins) is None
    assert ui.bin_at(-1, 0, editor.n_bins) is None
    assert ui.bin_at(0, -1, editor.n_bins) is None
    assert ui.bin_at(ui.GRID_WIDTH, 0, editor.n_bins) is None


def test_a_64_by_64_image_wraps_to_16_rows():
    # 64*64 = 4096 bins, not the 7938 a doubled count would give: the spectrum
    # has one bin per pixel.
    I = np.ones((64, 64), dtype=np.int64)
    assert definition.spectrum_length(I) == 4096
    assert ui.grid_rows(4096) == 16


def test_the_window_is_1024_wide_and_tall_enough_for_the_grid():
    # 4096 bins wrap to 16 rows of 4, which is shorter than the progress panel
    # the same space has to hold, so the panel sets the height.
    assert ui.window_size(4096) == (1024, ui.IMAGE_PANEL_H + ui.EDIT_PANEL_H)


def test_the_window_grows_when_the_grid_outgrows_the_progress_panel():
    # Up to 8192 bins the two fit the same space; past that the grid is the
    # taller of the two and the window has to follow, or the last rows of the
    # spectrum would be off the bottom of the screen.
    assert ui.window_size(8192)[1] == ui.IMAGE_PANEL_H + 32 * ui.CELL
    assert ui.window_size(8320)[1] == ui.IMAGE_PANEL_H + 33 * ui.CELL
    assert ui.window_size(8320)[1] > ui.window_size(8192)[1]


# -- drawing ---------------------------------------------------------------


def test_the_grid_is_a_uint8_bgr_image_of_the_right_size(editor):
    grid = editor.render_grid()
    assert grid.dtype == np.uint8 and grid.ndim == 3 and grid.shape[2] == 3
    assert grid.shape == (ui.grid_rows(editor.n_bins) * ui.CELL, ui.GRID_WIDTH, 3)


def test_an_untouched_grid_is_all_neutral_where_it_can_be_edited(editor):
    grid = editor.render_grid()
    for k in range(editor.n_bins):
        if editor.reachable[k]:
            x, y = ui.cell_rect(k)
            assert np.array_equal(grid[y, x], ui.diverging_lut()[NEUTRAL])


def test_unreachable_bins_are_drawn_differently_from_reachable_ones(editor):
    # Hatched dark, not a colour on the scale: a cell the user cannot move
    # should not look like one they can.
    grid = editor.render_grid()
    live, dead = int(np.flatnonzero(editor.reachable)[0]), int(np.flatnonzero(~editor.reachable)[0])
    lx, ly = ui.cell_rect(live)
    dx, dy = ui.cell_rect(dead)
    assert not np.array_equal(grid[ly, lx], grid[dy, dx])


def test_each_bin_is_drawn_in_its_own_cell(editor):
    # Every bin, given its own distinct delta, must show that delta in its own
    # cell. Asserting the grid looks plausible is not enough: a block expansion
    # that puts the wrong colour in each cell produces a full grid of correct
    # shape and wrong values, which is exactly the bug this pins down.
    editor.deltas = np.linspace(-0.4, 0.4, editor.n_bins)
    grid = editor.render_grid()
    lut = ui.diverging_lut()
    for k in range(editor.n_bins):
        x, y = ui.cell_rect(k)
        if not editor.reachable[k]:
            continue
        want = tuple(int(c) for c in lut[int(ui.delta_bytes([editor.deltas[k]],
                                                            editor.window)[0])])
        got = tuple(int(c) for c in grid[y + 1, x + 1])
        assert got == want, f"bin {k} drawn as {got}, wanted {want}"


def test_a_drawn_cell_is_uniform_apart_from_the_outline(editor):
    # With nothing selected or alerted, a live cell is one colour all the way
    # across. Unreachable cells are deliberately two-tone, which is the hatch.
    editor.deltas = np.linspace(-0.4, 0.4, editor.n_bins)
    grid = editor.render_grid()
    for k in range(editor.n_bins):
        if not editor.reachable[k]:
            continue
        x, y = ui.cell_rect(k)
        cell = grid[y:y + ui.CELL, x:x + ui.CELL].reshape(-1, 3)
        assert len(np.unique(cell, axis=0)) == 1


def test_the_selection_is_outlined_without_hiding_the_byte(editor):
    k = int(np.flatnonzero(editor.reachable)[0])
    x, y = ui.cell_rect(k)
    plain = editor.render_grid()
    editor.selected = k
    marked = editor.render_grid()
    assert not np.array_equal(plain[y, x], marked[y, x])
    # The middle of the cell is the byte itself, and it is still there.
    assert np.array_equal(plain[y + 1, x + 1], marked[y + 1, x + 1])


def test_a_negative_delta_draws_on_the_other_side_of_neutral(editor):
    k = int(np.flatnonzero(editor.reachable)[0])
    editor.set_delta(k, 0.2)
    up = editor.render_grid()
    editor.set_delta(k, -0.2)
    down = editor.render_grid()
    lut = ui.diverging_lut()
    x, y = ui.cell_rect(k)
    assert np.array_equal(up[y, x], lut[int(ui.delta_bytes([0.2], 1.0)[0])])
    assert not np.array_equal(up[y, x], down[y, x])


def test_the_image_panel_is_a_uint8_image_of_the_requested_box(image):
    panel = ui.render_image(image, box=(64, 32))
    assert panel.shape == (32, 64, 3) and panel.dtype == np.uint8


def test_the_image_is_drawn_nearest_neighbour(image):
    # 3x3 in a 30x30 box, so ten pixels per side: each source pixel becomes a
    # 10x10 block of its own value. What is drawn is the image's values, not a
    # resampled version of them.
    panel = ui.render_image(image, box=(30, 30))
    for (r, c), value in np.ndenumerate(image):
        # Inset by one, so the frame around the image is not counted as part of
        # the outermost blocks.
        block = panel[r * 10 + 1:r * 10 + 9, c * 10 + 1:c * 10 + 9]
        assert len(np.unique(block)) == 1
        assert block[0, 0][0] == value


def test_a_large_image_is_scaled_down_to_fit(image):
    panel = ui.render_image(np.arange(4096).reshape(64, 64) % 256, box=(32, 32))
    assert panel.shape == (32, 32, 3)


def test_the_progress_panel_is_the_size_it_is_asked_for():
    assert ui.render_progress([1.0], 0, 10, box=(64, 16)).shape == (16, 64, 3)


def test_the_progress_bar_reflects_the_steps_taken():
    half = ui.render_progress([1.0], 5, 10, box=(100, 16))
    full = ui.render_progress([1.0], 10, 10, box=(100, 16))
    assert not np.array_equal(half, full)


def test_the_progress_panel_survives_a_flat_loss():
    # A single-point history, and a run that is not moving, must not divide by a
    # zero span or produce NaN pixels.
    assert np.isfinite(ui.render_progress([2.0], 1, 10)).all()
    assert np.isfinite(ui.render_progress([2.0] * 5, 5, 10)).all()
    assert np.isfinite(ui.render_progress([0.0], 1, 10)).all()


def test_the_diverging_map_is_signed_and_continuous():
    lut = ui.diverging_lut()
    neutral = lut[NEUTRAL].astype(int)
    below = lut[NEUTRAL - 60].astype(int)
    above = lut[NEUTRAL + 60].astype(int)
    # Below neutral is blue-dominant, above is red-dominant, and the two ends
    # are far apart, so the sign is readable at a glance.
    assert below[0] > below[2]
    assert above[2] > above[0]
    assert np.abs(below - neutral).sum() > 100
    assert np.abs(above - neutral).sum() > 100


# -- the editor ------------------------------------------------------------


def test_the_editor_measures_the_exact_spectrum(image):
    editor = ui.Editor(image, backend=descent.ReferenceBackend())
    assert list(editor.spectrum) == list(area_spectrum(image))
    assert list(editor.scale) == list(jacobian_row_sums(image))


def test_a_bin_with_a_zero_row_sum_cannot_be_edited(editor):
    dead = int(np.flatnonzero(~editor.reachable)[0])
    assert editor.can_edit(dead) is False
    assert editor.set_delta(dead, 1.0) is False
    assert editor.nudge(dead, 1) is False
    assert editor.deltas[dead] == 0.0


def test_a_reachable_bin_accepts_a_delta(editor):
    k = int(np.flatnonzero(editor.reachable)[0])
    assert editor.can_edit(k) is True
    assert editor.set_delta(k, 1.0) is True
    assert editor.deltas[k] == 1.0


def test_nudging_accumulates_whole_pixels(editor):
    k = int(np.flatnonzero(editor.reachable)[0])
    editor.nudge(k, 3)
    assert int(ui.delta_bytes([editor.deltas[k]], 1.0)[0]) == NEUTRAL + 3
    editor.nudge(k, -3)
    assert int(ui.delta_bytes([editor.deltas[k]], 1.0)[0]) == NEUTRAL


def test_nudging_out_of_range_is_refused_not_clamped(editor):
    k = int(np.flatnonzero(editor.reachable)[0])
    editor.nudge(k, 20000)
    # The delta itself is not clamped; only the drawing is. Clamping the value
    # would make a later narrowing of the window forget what was asked for.
    assert editor.deltas[k] > 127
    assert int(ui.delta_bytes(editor.deltas[k], 1.0)[0]) == 255


def test_reset_clears_the_deltas_and_the_selection(editor):
    k = int(np.flatnonzero(editor.reachable)[0])
    editor.set_delta(k, 5.0)
    editor.selected = k
    editor.reset()
    assert not editor.deltas.any()
    assert editor.selected is None


def test_rebaseline_zeroes_the_deltas(editor):
    # The deltas were a request against the image as it was. Once the image
    # moves they describe a different request, so they are dropped rather than
    # silently reinterpreted against a new baseline.
    k = int(np.flatnonzero(editor.reachable)[0])
    editor.set_delta(k, 5.0)
    editor.rebaseline()
    assert not editor.deltas.any()


def test_the_editor_does_not_alias_the_image_it_was_given(image):
    editor = ui.Editor(image, backend=descent.ReferenceBackend())
    editor.image[0, 0] = 99
    assert image[0, 0] == 1


def test_the_editor_uses_the_ntt_backend_when_given_one(image):
    editor = ui.Editor(image, backend=descent.ReferenceBackend())
    ntt = ui.Editor(image, backend=view.ntt_spectrum.NTTBackend())
    assert ntt.spectrum == editor.spectrum
    assert ntt.scale == editor.scale


# -- the window ------------------------------------------------------------


def make_viewer(image, **kwargs):
    return view.Viewer(ui.Editor(image, backend=descent.ReferenceBackend()), **kwargs)


def arrow(name):
    """One of the key codes an arrow maps to, the way a GUI backend would send it."""
    return sorted(view._ARROW[name])[0]


def test_the_window_composes_without_a_display(image):
    viewer = make_viewer(image)
    frame = viewer.compose()
    assert frame.dtype == np.uint8
    assert frame.shape == (viewer.size[1], viewer.size[0], 3)


def test_the_title_reports_the_loss_and_the_bin(image):
    viewer = make_viewer(image)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    viewer.editor.set_delta(k, 1.0)
    viewer.editor.selected = k
    title = viewer.title()
    assert "loss" in title and f"bin {k}" in title


def test_the_title_says_how_many_bins_are_asked_for(image):
    viewer = make_viewer(image)
    assert "0 asked" in viewer.title()
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    viewer.editor.set_delta(k, 1.0)
    assert "1 asked" in viewer.title()


def test_the_title_reports_the_loss_the_run_will_minimise(image):
    # Not a second, different number: the idle loss and the loss a run starts
    # from must be the same value, or the progress graph contradicts the title.
    viewer = make_viewer(image)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    viewer.editor.set_delta(k, 1.0)
    assert viewer.current_loss() == viewer.editor.loss()
    viewer.on_key(13)
    assert viewer.last.loss == pytest.approx(viewer.editor.loss())


def test_q_and_escape_quit_while_editing(image):
    assert make_viewer(image).on_key(ord("q")) is False
    assert make_viewer(image).on_key(27) is False


def test_a_keypress_with_nothing_pressed_does_nothing(image):
    viewer = make_viewer(image)
    assert viewer.on_key(-1) is True
    assert viewer.state == "idle"


def test_enter_starts_a_run_and_escape_cancels_it(image):
    viewer = make_viewer(image, max_steps=50)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    viewer.editor.set_delta(k, 3.0)
    viewer.on_key(13)
    assert viewer.state == "running"
    assert viewer.run is not None
    viewer.on_key(27)
    assert viewer.state == "done"
    assert viewer.cancelled is True


def test_a_run_advances_when_pumped(image):
    viewer = make_viewer(image, max_steps=20, patience=1000)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    viewer.editor.set_delta(k, 3.0)
    viewer.on_key(13)
    before = viewer.last.accepted_steps
    # Not one step per pump: a step descent tries may be rejected, and a
    # rejected step is invisible from outside. What must hold is that pumping
    # makes progress and never reports a loss that is not the one it drew.
    for _ in range(20):
        if viewer.state != "running":
            break
        viewer.pump()
        assert viewer.last.loss == pytest.approx(viewer.last.history[-1])
    assert viewer.last.accepted_steps > before


def test_pumping_to_the_end_reports_patience_and_not_a_cancellation(image):
    viewer = make_viewer(image, max_steps=40, patience=3)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    viewer.editor.set_delta(k, 3.0)
    viewer.on_key(13)
    for _ in range(500):
        viewer.pump()
        if viewer.state != "running":
            break
    assert viewer.state == "done"
    assert viewer.cancelled is False
    assert viewer.last.stopped_early is True


def test_a_run_lowers_the_loss(image):
    viewer = make_viewer(image, max_steps=60, patience=5)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    viewer.editor.set_delta(k, 3.0)
    start = viewer.current_loss()
    viewer.on_key(13)
    while viewer.state == "running":
        viewer.pump()
    assert viewer.last.loss < start
    assert viewer.last.loss == pytest.approx(viewer.last.history[-1])


def test_a_keypress_applies_the_result_and_rebaselines(image):
    viewer = make_viewer(image, max_steps=20)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    viewer.editor.set_delta(k, 3.0)
    viewer.on_key(13)
    while viewer.state == "running":
        viewer.pump()
    result = viewer.last.image.copy()
    viewer.on_key(ord("a"))
    assert viewer.state == "idle"
    assert np.array_equal(viewer.editor.image, result)
    # The deltas are dropped, because they were a request against the old image.
    assert not viewer.editor.deltas.any()


def test_cancelling_keeps_the_steps_already_taken(image):
    viewer = make_viewer(image, max_steps=1000, patience=1000)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    viewer.editor.set_delta(k, 3.0)
    viewer.on_key(13)
    viewer.pump()
    taken = viewer.last.accepted_steps
    assert taken > 0
    viewer.on_key(27)
    assert viewer.last.accepted_steps == taken


def test_the_panel_is_the_grid_while_editing_and_progress_while_running(image):
    viewer = make_viewer(image, max_steps=10)
    box = (viewer.size[0], viewer.size[1] - ui.IMAGE_PANEL_H)
    assert np.array_equal(viewer.panel(), viewer.editor.render_grid(box=box))
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    viewer.editor.set_delta(k, 2.0)
    viewer.on_key(13)
    assert not np.array_equal(viewer.panel(), viewer.editor.render_grid())


def test_a_click_selects_the_bin_under_the_pointer(image):
    viewer = make_viewer(image)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    x, y = ui.cell_rect(k)
    viewer.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y + ui.IMAGE_PANEL_H, 0)
    assert viewer.editor.selected == k


def test_a_click_ignores_the_image_panel(image):
    viewer = make_viewer(image)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    x, _ = ui.cell_rect(k)
    viewer.on_mouse(cv2.EVENT_LBUTTONDOWN, x, 0, 0)
    assert viewer.editor.selected is None


def test_clicking_an_unreachable_bin_says_so_instead_of_ignoring_it(image):
    viewer = make_viewer(image)
    dead = int(np.flatnonzero(~viewer.editor.reachable)[0])
    x, y = ui.cell_rect(dead)
    viewer.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y + ui.IMAGE_PANEL_H, 0)
    assert viewer.editor.selected is None
    assert viewer.alert == dead
    assert "unreachable" in viewer.status


def test_the_alert_stops_after_a_few_frames(image):
    viewer = make_viewer(image)
    box = (viewer.size[0], viewer.size[1] - ui.IMAGE_PANEL_H)
    dead = int(np.flatnonzero(~viewer.editor.reachable)[0])
    x, y = ui.cell_rect(dead)
    viewer.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y + ui.IMAGE_PANEL_H, 0)
    assert viewer.alert == dead
    # While the alert is up the grid carries a red outline on the dead bin.
    during = viewer.panel()
    dx, dy = ui.cell_rect(dead)
    assert np.array_equal(during[dy, dx], ui._ALERT_BGR)
    # It is a timed flash, not a state: the frame it expires on is already the
    # plain grid, so the refusal cannot outlive its explanation.
    viewer.frame = viewer.alert_until
    assert np.array_equal(viewer.panel(), viewer.editor.render_grid(box=box))
    viewer.frame = viewer.alert_until + 1
    assert np.array_equal(viewer.panel(), viewer.editor.render_grid(box=box))


def test_dragging_up_moves_the_bin_above_neutral(image):
    viewer = make_viewer(image)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    x, y = ui.cell_rect(k)
    viewer.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y + ui.IMAGE_PANEL_H, 0)
    viewer.on_mouse(cv2.EVENT_MOUSEMOVE, x, y + ui.IMAGE_PANEL_H - 5,
                    cv2.EVENT_FLAG_LBUTTON)
    assert int(ui.delta_bytes([viewer.editor.deltas[k]], 1.0)[0]) == NEUTRAL + 5


def test_dragging_down_moves_it_below_neutral(image):
    viewer = make_viewer(image)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    x, y = ui.cell_rect(k)
    viewer.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y + ui.IMAGE_PANEL_H, 0)
    viewer.on_mouse(cv2.EVENT_MOUSEMOVE, x, y + ui.IMAGE_PANEL_H + 5,
                    cv2.EVENT_FLAG_LBUTTON)
    assert int(ui.delta_bytes([viewer.editor.deltas[k]], 1.0)[0]) == NEUTRAL - 5


def test_a_drag_is_anchored_so_it_returns_to_where_it_started(image):
    # Absolute from the press, not accumulated per motion event, so a drag out
    # and back is a round trip and not a small permanent offset.
    viewer = make_viewer(image)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    x, y = ui.cell_rect(k)
    start = y + ui.IMAGE_PANEL_H
    viewer.on_mouse(cv2.EVENT_LBUTTONDOWN, x, start, 0)
    viewer.on_mouse(cv2.EVENT_MOUSEMOVE, x, start - 20, cv2.EVENT_FLAG_LBUTTON)
    viewer.on_mouse(cv2.EVENT_MOUSEMOVE, x, start - 20, cv2.EVENT_FLAG_LBUTTON)
    viewer.on_mouse(cv2.EVENT_MOUSEMOVE, x, start, cv2.EVENT_FLAG_LBUTTON)
    assert viewer.editor.deltas[k] == pytest.approx(0.0, abs=1e-12)


def test_moving_without_the_button_down_does_not_drag(image):
    viewer = make_viewer(image)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    x, y = ui.cell_rect(k)
    viewer.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y + ui.IMAGE_PANEL_H, 0)
    viewer.on_mouse(cv2.EVENT_MOUSEMOVE, x, y + ui.IMAGE_PANEL_H - 30, 0)
    assert viewer.editor.deltas[k] == 0.0


def test_the_drag_ends_on_release(image):
    viewer = make_viewer(image)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    x, y = ui.cell_rect(k)
    start = y + ui.IMAGE_PANEL_H
    viewer.on_mouse(cv2.EVENT_LBUTTONDOWN, x, start, 0)
    viewer.on_mouse(cv2.EVENT_LBUTTONUP, x, start, 0)
    assert viewer.drag is None


def test_a_drag_cannot_start_on_an_unreachable_bin(image):
    viewer = make_viewer(image)
    dead = int(np.flatnonzero(~viewer.editor.reachable)[0])
    x, y = ui.cell_rect(dead)
    viewer.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y + ui.IMAGE_PANEL_H, 0)
    assert viewer.drag is None


def test_the_arrows_move_the_selection_over_the_reachable_bins(image):
    viewer = make_viewer(image)
    viewer.on_key(arrow("right"))
    first = viewer.editor.selected
    assert viewer.editor.can_edit(first)
    viewer.on_key(arrow("right"))
    assert viewer.editor.selected != first
    assert viewer.editor.can_edit(viewer.editor.selected)


def test_the_up_and_down_arrows_nudge_the_selected_bin(image):
    viewer = make_viewer(image)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    viewer.editor.selected = k
    viewer.on_key(arrow("up"))
    assert int(ui.delta_bytes([viewer.editor.deltas[k]], 1.0)[0]) == NEUTRAL + 1
    viewer.on_key(arrow("down"))
    assert int(ui.delta_bytes([viewer.editor.deltas[k]], 1.0)[0]) == NEUTRAL


def test_the_window_can_be_halved_and_doubled_from_the_keyboard(image):
    viewer = make_viewer(image)
    viewer.on_key(ord("-"))
    assert viewer.editor.window == 0.5
    viewer.on_key(ord("="))
    assert viewer.editor.window == 1.0


def test_the_selection_never_lands_on_an_unreachable_bin(image):
    viewer = make_viewer(image)
    for _ in range(2 * definition.spectrum_length(image)):
        viewer.on_key(arrow("right"))
        k = viewer.editor.selected
        assert k is None or viewer.editor.can_edit(k)


def test_a_run_cannot_be_started_twice(image):
    viewer = make_viewer(image, max_steps=1000, patience=1000)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    viewer.editor.set_delta(k, 3.0)
    viewer.on_key(13)
    running = viewer.run
    viewer.on_key(13)
    assert viewer.run is running


def test_a_run_with_nothing_asked_for_is_refused(image):
    # Otherwise the objective is empty, every step is judged against a loss of
    # zero, and the run reports patience having done nothing.
    viewer = make_viewer(image, max_steps=10, patience=2)
    viewer.on_key(13)
    assert viewer.state == "idle"
    assert viewer.run is None
    assert "nothing asked for" in viewer.status


def test_a_run_starts_once_something_is_asked_for(image):
    viewer = make_viewer(image, max_steps=1000, patience=1000)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    viewer.editor.set_delta(k, 3.0)
    viewer.on_key(13)
    assert viewer.state == "running"


def test_the_mouse_is_ignored_while_a_run_is_in_flight(image):
    viewer = make_viewer(image, max_steps=1000, patience=1000)
    k = int(np.flatnonzero(viewer.editor.reachable)[0])
    viewer.editor.set_delta(k, 3.0)
    viewer.on_key(13)
    x, y = ui.cell_rect(k)
    viewer.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y + ui.IMAGE_PANEL_H, 0)
    assert viewer.editor.selected is None


def test_main_reports_a_missing_image_without_opening_a_window(tmp_path, monkeypatch):
    monkeypatch.setattr(view.Viewer, "loop", lambda self: pytest.fail("opened a window"))
    assert view.main([str(tmp_path / "absent.png")]) == 2


def test_main_reports_an_undecodable_file_without_opening_a_window(tmp_path, monkeypatch):
    path = tmp_path / "notes.txt"
    path.write_text("not an image")
    monkeypatch.setattr(view.Viewer, "loop", lambda self: pytest.fail("opened a window"))
    assert view.main([str(path)]) == 2


def test_main_loads_and_reduces_the_image(tmp_path, monkeypatch):
    path = str(tmp_path / "big.png")
    assert cv2.imwrite(path, np.full((40, 80), 100, dtype=np.uint8))
    seen = {}

    def fake_loop(self):
        seen["shape"] = self.editor.image.shape
        return 0

    monkeypatch.setattr(view.Viewer, "loop", fake_loop)
    assert view.main([path, "--max-side", "20"]) == 0
    # The longest side, not the width: 80 columns to 20, so 40 rows to 10.
    assert seen["shape"] == (10, 20)


def test_main_keeps_the_image_size_when_max_side_is_zero(tmp_path, monkeypatch):
    path = str(tmp_path / "small.png")
    assert cv2.imwrite(path, np.full((12, 20), 3, dtype=np.uint8))
    seen = {}

    def fake_loop(self):
        seen["shape"] = self.editor.image.shape
        return 0

    monkeypatch.setattr(view.Viewer, "loop", fake_loop)
    assert view.main([path, "--max-side", "0"]) == 0
    assert seen["shape"] == (12, 20)
