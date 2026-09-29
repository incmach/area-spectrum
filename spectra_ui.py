"""The editable area spectrum: deltas, the bytes they draw as, and the panels.

What the user edits is not a spectrum but a *normalized residual*. A bin k of
the spectrum is not directly steerable: one step of one pixel moves bin k by
the Jacobian column at that pixel, whose size depends on k and on the image.
The quantity that behaves the same everywhere is

    delta[k] = (target[k] - area_spectrum(I)[k]) / scale[k]

with scale the per-bin L1 row sum of the spectrum's Jacobian. That is the
normalized residual descent.spectrum_residual_loss minimises, so editing it is
editing the objective directly rather than editing something that is then
scaled by a per-bin factor the user cannot see. The delta is the source of
truth; the target spectrum descent consumes is derived from it, so what is
drawn and what is optimized cannot drift apart.

The delta is unbounded while a byte is not, so it is drawn through a window:

    byte = clip(127 + round(127 * delta / window), 0, 255)

with 127 the neutral byte. A bin at 127 is one the user has not touched, the
display is signed around it, and the window is the half-width that decides how
large a delta saturates the display. Making it a parameter rather than a
constant is what lets the same drawing serve a request to nudge a bin by a
fraction of a count and a request to double it.

Everything in this module is a pure function or a plain state object, and none
of it draws to a screen. The event loop in view.py is the only part that needs
a display, which is what leaves the mapping, the target reconstruction and the
hit testing testable without one -- they are the parts that fail quietly, a
bin in the wrong place or a target clamped where it should not be.
"""
import os

import cv2
import numpy as np

import definition
import descent

# The neutral byte: what a bin the user has not touched draws as, and the pivot
# of the diverging map. 127 rather than 128 so that the byte range splits evenly
# either side of it, 127 down and 128 up.
NEUTRAL = 127

# A bin is drawn as a CELL x CELL block rather than a single pixel. A bin per
# pixel is unreadable and unclickable: at one pixel the pointer cannot reliably
# address the bin it appears to be over, and a thin 1024-wide row is a stripe
# rather than a control. Four pixels is the smallest block that is comfortable
# to hit with a mouse.
CELL = 4

# The row wraps here. Bins are laid out left to right and wrap to the next line
# when they run out of width, so the whole spectrum is on screen at once and no
# navigation is needed to reach a bin.
GRID_WIDTH = 1024
BINS_PER_ROW = GRID_WIDTH // CELL

# Window heights, in pixels, of the fixed panels. The image sits above the
# spectrum and the spectrum is replaced by progress while a descent runs.
IMAGE_PANEL_H = 256
EDIT_PANEL_H = 128

# What a dragged pixel is worth, in delta units, given a window. Defined so that
# dragging n pixels moves the drawn byte by exactly n: a delta of
# n * window / NEUTRAL satisfies 127 * delta / window == n, so the byte is
# 127 + n rather than something near it. The drag is therefore in the same
# units the user sees, and the round trip delta -> byte -> delta is exact for
# whole pixels.
def delta_per_pixel(window):
    return window / NEUTRAL


def load_image(path, max_side=None):
    """Read an image file as a 2D array of non-negative int64.

    Grayscale, because the spectrum is defined for scalar fields and a colour
    image has no single set of pixel values to speak of. int64, because every
    path that consumes it wants exact integers and the spectrum arithmetic is
    int64 end to end.

    max_side, if given, is the longest side to reduce the image to. The area
    spectrum grows with the square of the size and the descent evaluates one
    Jacobian column per step, so the image size is what decides whether a run
    is interactive at all; 32x32 gives 1024 bins and a few hundred steps in
    seconds. Downscaling uses INTER_AREA, which averages over the source rather
    than dropping pixels, so the result is a real image of the new size.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such image: {path}")
    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if data is None:
        # imread returns None for an unreadable format and for a directory, and
        # says which in neither case, so this is the honest report: the file is
        # there and is not an image this build can decode.
        raise ValueError(f"could not decode {path} as an image")
    I = data.astype(np.int64)
    if max_side is not None and max(I.shape) > max_side:
        scale = max_side / max(I.shape)
        size = (max(1, int(round(I.shape[1] * scale))),
                max(1, int(round(I.shape[0] * scale))))
        # Resized as uint8 and widened after, because cv2.resize has no int64
        # path: an int64 image has one channel of a type it will not touch.
        # uint8 is also the right precision to average in, since the source was
        # 8-bit to begin with and nothing here can invent precision.
        I = cv2.resize(data, size, interpolation=cv2.INTER_AREA).astype(np.int64)
    return I


def delta_bytes(deltas, window):
    """The byte each bin draws as: 127 neutral, signed, clipped, uint8.

    Always one-dimensional, so a scalar argument gives a one-element result
    rather than a bare numpy scalar. The callers here are all asking about a
    single bin -- what byte is this one, nudge this one -- and a function whose
    result needs [0] added to it only when it is handed a scalar is a trap
    waiting for the one caller that forgets.

    The round is round-half-to-even on both sides (np.rint and Python's round
    agree), so a delta that lands exactly on a half byte boundary does not drift
    upwards, and a drag that returns to its starting point returns to the same
    bytes.

    Clipping is where the window earns its keep. A delta beyond the window
    saturates at 0 or 255 and the user is told only that it is off the scale;
    that is the honest report, since the drawing cannot show a magnitude it does
    not have room for, and the window is adjustable for exactly that reason.

    The swing is not symmetric: 127 counts below neutral and 128 above, because
    that is how many byte values there are. A delta of -window draws as 0 and a
    delta of +window as 254, and 255 arrives at 128/127 of the window. Widening
    the window to 128/127 of itself would make the ends symmetric and would cost
    the exactness below, which is worth more.
    """
    scaled = NEUTRAL + np.rint(
        NEUTRAL * np.atleast_1d(np.asarray(deltas, dtype=float)) / window)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def delta_for_bytes(n_bytes, window):
    """The delta that draws as NEUTRAL + n_bytes, the inverse of delta_bytes.

    Exact for whole n_bytes rather than approximately, which is what lets a
    drag be a count of pixels: delta_for_bytes(delta_to_byte(d)) == d for any d
    that came from a whole-pixel drag.
    """
    return np.atleast_1d(np.asarray(n_bytes, dtype=float)) * delta_per_pixel(window)


def target_spectrum(spectrum, scale, deltas):
    """The spectrum descent is asked to hit, derived from the deltas.

    target[k] = round(spectrum[k] + delta[k] * scale[k])

    Rounded, because a target is an integer spectrum and the delta is a real
    number: a bin can be asked for a fraction of a count only up to the
    rounding. Not clamped. A negative delta can ask for a target below zero,
    which no non-negative image can have, and the honest thing is to leave the
    request in the target and let the loss report the residual it cannot remove
    -- clamping would silently reinterpret the request as a smaller one, and the
    user would see a run that ends at a loss they never asked for. Bins with
    scale[k] == 0 are unreachable by any step, so their delta is not expressed
    at all and they are left at the current spectrum.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    scale = np.atleast_1d(np.asarray(scale, dtype=np.float64))
    deltas = np.atleast_1d(np.asarray(deltas, dtype=np.float64))
    reachable = scale > 0
    moved = spectrum + deltas * scale
    return np.rint(np.where(reachable, moved, spectrum)).astype(np.int64)


def grid_rows(n_bins):
    """How many wrapped lines n_bins occupy."""
    return max(1, -(-n_bins // BINS_PER_ROW))


def window_size(n_bins):
    """(width, height) of the whole window, in pixels."""
    return GRID_WIDTH, IMAGE_PANEL_H + max(EDIT_PANEL_H, grid_rows(n_bins) * CELL)


def cell_rect(k):
    """(x, y) top-left of bin k's cell, in grid coordinates."""
    row, col = divmod(k, BINS_PER_ROW)
    return col * CELL, row * CELL


def bin_at(x, y, n_bins):
    """Which bin a grid pixel is over, or None if it is not over a bin.

    Floor division rather than a test for exact cell alignment, so a click
    between two cells lands on one of them instead of being discarded: a four
    pixel cell is small enough that being fussy about the last pixel of it would
    feel broken.
    """
    if x < 0 or y < 0 or x >= GRID_WIDTH:
        return None
    row, col = y // CELL, x // CELL
    k = row * BINS_PER_ROW + col
    return k if k < n_bins else None


# The diverging map. Two saturated hues meeting at a near-white neutral, so the
# side of neutral a bin sits on is readable without reading its number. A
# sequential map would encode magnitude well and sign badly, and sign is what
# the user is choosing when they drag up or down. All uint8, so the fills below
# are uint8 and nothing promotes a panel to float on its way to the screen.
_NEUTRAL_RGB = np.array([242, 242, 242], dtype=np.uint8)
_NEGATIVE_BGR = np.array([232, 96, 0], dtype=np.uint8)     # blue
_POSITIVE_BGR = np.array([0, 96, 232], dtype=np.uint8)     # red
_UNREACHABLE_BGR = np.array([56, 56, 56], dtype=np.uint8)
_HATCH_BGR = np.array([30, 30, 30], dtype=np.uint8)
_PANEL_BGR = np.array([24, 24, 24], dtype=np.uint8)
_SELECTED_BGR = np.array([0, 230, 230], dtype=np.uint8)    # yellow
_ALERT_BGR = np.array([0, 0, 255], dtype=np.uint8)         # red, for a refusal


def diverging_lut():
    """A 256-entry BGR table mapping each byte to its colour.

    Built once and indexed per cell, rather than interpolating per cell at draw
    time, so the colour of a bin is a function of its byte alone and can be
    asserted on in a test without a screen.

    The palette above is uint8, so the blend casts to float first. Subtracting
    two uint8 endpoints wraps instead of going negative -- 232 - 242 is 246, not
    -10 -- and a wrapped blend produces a colour brighter than the neutral it
    started from, which is not a colour this map can have.
    """
    neutral = _NEUTRAL_RGB.astype(float)
    negative = _NEGATIVE_BGR.astype(float)
    positive = _POSITIVE_BGR.astype(float)
    lut = np.zeros((256, 3), dtype=np.float64)
    t = (np.arange(256, dtype=np.float64) - NEUTRAL) / NEUTRAL
    neg = t < 0
    lut[neg] = neutral + (negative - neutral) * (-t[neg])[:, None]
    lut[~neg] = neutral + (positive - neutral) * t[~neg][:, None]
    return np.clip(np.rint(lut), 0, 255).astype(np.uint8)


def _hatch(height, width):
    """A diagonal stripe mask, for marking cells that cannot be addressed."""
    ys, xs = np.mgrid[0:height, 0:width]
    return ((xs + ys) // 2) % 2 == 0


def render_grid(deltas, reachable, window, selected=None, alert=None, box=None):
    """Draw the delta row as a wrapped grid of CELL-sized cells, as BGR.

    Unreachable bins (scale == 0) are drawn as a hatched dark cell rather than
    in the diverging colours. They cannot be moved by any pixel step, so a delta
    there is a request that cannot be expressed; drawing them like live bins
    would offer an affordance that does nothing.

    The selection and the alert are outlines rather than fills, so neither hides
    the byte underneath -- the value being edited is the thing the user is
    looking at while dragging.

    box, if given, is the (width, height) the panel must come out as. The grid
    is what sets the height otherwise, and the two disagree: a small spectrum
    wraps to a single short row while the space reserved below the image is
    sized for the progress panel that replaces the grid during a run. Padding to
    the box is what keeps the two panels the same size, so the window does not
    change height when a run starts.
    """
    n_bins = len(deltas)
    rows = grid_rows(n_bins)
    total = rows * BINS_PER_ROW
    height = rows * CELL
    lut = diverging_lut()
    # Cells that can be addressed: within the spectrum, and with a row sum.
    # Padding the wrap and the unreachable bins into one mask means they are
    # drawn by the same path rather than by a special case per region.
    live = np.zeros(total, dtype=bool)
    live[:n_bins] = np.asarray(reachable, dtype=bool)[:n_bins]
    raw = np.zeros(total, dtype=np.uint8)
    raw[:n_bins] = delta_bytes(deltas, window)
    cells = np.where(live[:, None], lut[raw], _UNREACHABLE_BGR)
    cells[~live] = _UNREACHABLE_BGR
    # Every pixel takes the colour of the cell it falls in, as a gather rather
    # than a repeat followed by a reshape. The repeat has to insert the two
    # within-cell axes in exactly the order the final reshape expects, and
    # getting that order wrong does not crash or look obviously wrong -- every
    # cell still has a colour, it is just its neighbour's, and a bin shows the
    # delta of the bin beside it.
    ys = np.arange(rows * CELL) // CELL
    xs = np.arange(BINS_PER_ROW * CELL) // CELL
    canvas = np.ascontiguousarray(
        cells.reshape(rows, BINS_PER_ROW, 3)[np.ix_(ys, xs)])
    dead = ~live.reshape(rows, BINS_PER_ROW)[np.ix_(ys, xs)]
    canvas[dead] = _UNREACHABLE_BGR
    canvas[dead & _hatch(height, GRID_WIDTH)] = _HATCH_BGR
    for k, colour in ((alert, _ALERT_BGR), (selected, _SELECTED_BGR)):
        if k is None or not 0 <= k < n_bins:
            continue
        x, y = cell_rect(k)
        canvas[y:y + 1, x:x + CELL] = colour
        canvas[y + CELL - 1:y + CELL, x:x + CELL] = colour
        canvas[y:y + CELL, x:x + 1] = colour
        canvas[y:y + CELL, x + CELL - 1:x + CELL] = colour
    if box is None:
        return canvas
    width, height = box
    fitted = np.empty((height, width, 3), dtype=np.uint8)
    fitted[:, :] = _PANEL_BGR
    rows = min(height, canvas.shape[0])
    cols = min(width, canvas.shape[1])
    fitted[:rows, :cols] = canvas[:rows, :cols]
    return fitted


def render_image(I, box=(GRID_WIDTH, IMAGE_PANEL_H)):
    """Draw the image, scaled to fit `box` and centred, as BGR.

    Nearest-neighbour by index arithmetic rather than an interpolation, so a
    pixel is drawn as a block of one colour and the user is looking at the actual
    integer values descent is moving. A smooth resample would look better and
    would not be the image.

    Two regimes, one code path. If the image fits at some whole multiple of its
    own size, each pixel becomes a block that multiple. If it does not fit, the
    drawing decimates instead, because clamping the block size to one and
    letting the canvas crop would silently draw only the top-left corner of a
    large image -- and the user would be editing a spectrum of a picture they
    are not looking at. The index mapping below is nearest in both cases: for
    upscaling it repeats each source row and column, and for downscaling it
    steps through them.
    """
    width, height = box
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:, :] = _PANEL_BGR
    H, W = I.shape
    block = min(width // W, height // H)
    if block >= 1:
        out_h, out_w = H * block, W * block
    else:
        out_h, out_w = min(H, height), min(W, width)
    ys = np.minimum((np.arange(out_h) * H) // out_h, H - 1)
    xs = np.minimum((np.arange(out_w) * W) // out_w, W - 1)
    shade = np.clip(np.asarray(I, dtype=np.int64)[np.ix_(ys, xs)], 0, 255)
    top, left = (height - out_h) // 2, (width - out_w) // 2
    canvas[top:top + out_h, left:left + out_w, 0] = shade
    canvas[top:top + out_h, left:left + out_w, 1] = shade
    canvas[top:top + out_h, left:left + out_w, 2] = shade
    # A one-pixel frame, so the extent of the image is visible against the
    # panel rather than floating in it.
    fy0, fy1 = max(0, top - 1), min(height, top + out_h + 1)
    fx0, fx1 = max(0, left - 1), min(width, left + out_w + 1)
    border = np.zeros((fy1 - fy0, fx1 - fx0), dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    canvas[fy0:fy1, fx0:fx1][border] = (96, 96, 96)
    return canvas


def render_progress(history, done_steps, max_steps, box=(GRID_WIDTH, EDIT_PANEL_H),
                    stopped_early=False, cancelled=False):
    """Draw a run in flight: a step bar above a loss curve, as BGR.

    The curve is log10(loss) rather than loss, because the loss falls by orders
    of magnitude over a run and on a linear axis the interesting part -- the
    flattening that says patience is about to end -- is a horizontal line on the
    floor. The bar is steps against max_steps, which is the only honest
    progress measure available: a run that ends on patience never reaches it.
    """
    width, height = box
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:, :] = _PANEL_BGR
    # The bar takes what it needs, but never more than the panel can spare: a box
    # too short for a curve still gets a bar, rather than a curve drawn past the
    # bottom of the panel into an out-of-bounds row.
    bar_h = min(10, max(0, height - 8))
    fraction = 0.0 if max_steps <= 0 else min(done_steps, max_steps) / max_steps
    filled = int(round(width * fraction))
    canvas[4:4 + bar_h, :] = (48, 48, 48)
    colour = _ALERT_BGR if stopped_early or cancelled else _POSITIVE_BGR
    canvas[4:4 + bar_h, :filled] = colour
    if not history:
        return canvas
    top = 4 + bar_h + 6
    plot_h = height - top - 4
    if plot_h < 1:
        return canvas
    losses = np.asarray(history, dtype=float)
    span = np.log10(max(losses.max(), 1e-300)) - np.log10(max(losses.min(), 1e-300))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    ys = np.log10(np.maximum(losses, 1e-300))
    # width, not GRID_WIDTH: the panel is asked for at a size, and a curve
    # plotted on a wider grid than the panel is an out-of-bounds index.
    xs = (np.linspace(0, width - 1, len(losses)) if len(losses) > 1
          else np.zeros(1))
    px = np.clip(xs.astype(int), 0, width - 1)
    py = (top + plot_h - 1 - (ys - (ys.max() - span)) / span * (plot_h - 1)).astype(int)
    py = np.clip(py, top, top + plot_h - 1)
    # Connect with vertical runs, so the curve is a connected polyline without
    # needing a line-drawing routine and without gaps on steep sections.
    for i in range(1, len(px)):
        lo, hi = sorted((py[i - 1], py[i]))
        canvas[lo:hi + 1, px[i]] = colour
    canvas[py, px] = (255, 255, 255)
    return canvas


class Editor:
    """The image, its baseline, and the deltas the user has asked for.

    Holds the baseline the deltas are relative to: the spectrum and the row
    sums of the image as it is *now*. Both are functions of the image and both
    go stale the moment a step is taken, which is why rebaseline() exists and
    why the view calls it when a run ends -- a delta asked of the old baseline
    no longer describes the same request against the new one.
    """

    def __init__(self, I, window=1.0, backend=None, max_value=255):
        self.backend = backend
        self.max_value = max_value
        self.window = float(window)
        self.deltas = np.zeros(definition.spectrum_length(I), dtype=float)
        self.selected = None
        self.image = np.array(I, dtype=np.int64, copy=True)
        self.rebaseline()

    def rebaseline(self):
        """Recompute spectrum and scale from the current image, zero the deltas.

        Zeroing is not cosmetic. The deltas were a request relative to the image
        as it was when they were made; once the image has moved they no longer
        mean the same thing, and keeping them would replay a request the user
        has already been given. Starting the next round from "this image,
        unchanged" is the only state in which a delta means what it says.
        """
        self.spectrum = self._spectrum(self.image)
        self.scale = self._scale(self.image)
        self.deltas = np.zeros(len(self.spectrum), dtype=float)
        self.selected = None

    def _spectrum(self, I):
        if self.backend is None:
            return list(definition.area_spectrum(I))
        return [int(v) for v in self.backend.area_spectrum(I)]

    def _scale(self, I):
        if self.backend is None:
            return [int(v) for v in definition.jacobian_row_sums(I)]
        return [int(v) for v in self.backend.jacobian_row_sums(I)]

    @property
    def n_bins(self):
        return len(self.spectrum)

    @property
    def reachable(self):
        return np.asarray(self.scale) > 0

    @property
    def edited(self):
        """The bins the user has actually asked for."""
        return self.deltas != 0

    @property
    def objective_scale(self):
        """The row sums, zeroed for every bin that was not asked for.

        This is what decides which bins the run is trying to hit, and it is the
        one thing that makes the tool work at all. Any single pixel step moves
        nearly every bin: a step of one unit against row sums in the hundreds of
        thousands shifts bins the user never touched by a fraction of their own
        scale, and there are 143 of those to a 144-bin spectrum. An objective
        over the whole spectrum therefore sees more damage from a step than gain
        from it, accepts nothing, and every run ends on patience without moving.

        Zero is how a bin is excluded, and it is already how a bin is excluded:
        both spectrum_residual_loss and spectrum_gradient skip a bin whose scale
        is zero, by the same test, so the loss and the gradient cannot disagree
        about what counts. The bins the user did not drag are not held in place
        -- they are simply not part of the request, and the run is free to disturb
        them. What is asked for is hit; what was not is allowed to move.

        A bin the user drags back to neutral stops being asked for, because the
        mask follows the delta rather than a separate list of touched bins.
        """
        scale = np.asarray(self.scale, dtype=np.float64)
        return np.where(self.edited, scale, 0.0)

    def target(self):
        """The spectrum to optimize towards, derived from the current deltas."""
        return target_spectrum(self.spectrum, self.scale, self.deltas)

    def can_edit(self, k):
        """Whether bin k can be moved at all.

        scale[k] == 0 means no pixel of the image changes bin k, so a delta
        there asks for something no image can do. Edits are refused rather than
        ignored, because a drag that silently does nothing is indistinguishable
        from a broken one.
        """
        return 0 <= k < self.n_bins and int(self.scale[k]) > 0

    def set_delta(self, k, value):
        """Set bin k's delta outright. False if the bin is unreachable."""
        if not self.can_edit(k):
            return False
        self.deltas[k] = value
        return True

    def nudge(self, k, pixels):
        """Move bin k by `pixels` dragged pixels, from where it is now.

        Additive rather than absolute so a held drag and a repeated keypress
        mean the same thing, and so the value is always within one pixel of a
        whole-pixel drag from the last commit, which keeps the round trip
        delta_for_bytes(delta_bytes(d)) exact for anything a drag produced.
        """
        if not self.can_edit(k):
            return False
        self.deltas[k] += pixels * delta_per_pixel(self.window)
        return True

    def reset(self):
        self.deltas = np.zeros(self.n_bins, dtype=float)
        self.selected = None

    def loss(self):
        """The objective at the current image, for the title bar.

        Over the bins that were asked for, which is what the run minimizes and so
        what the title must report: a number that counted the bins the user did
        not touch would move for reasons they did not cause, and would look like
        the run getting worse while it was getting closer.

        spectrum=self.spectrum passes the baseline already measured, so this is
        the arithmetic of spectrum_residual_loss and not a second spectrum pass:
        the number in the title bar is the exact same number descent will
        minimise, not a re-measurement of it that might disagree.
        """
        return descent.spectrum_loss(self.image, self.target(), self.objective_scale,
                                     backend=self.backend, spectrum=self.spectrum,
                                     max_value=self.max_value)

    def render_grid(self, alert=None, box=None):
        return render_grid(self.deltas, self.reachable, self.window,
                           selected=self.selected, alert=alert, box=box)

    def render_image(self, box=(GRID_WIDTH, IMAGE_PANEL_H)):
        return render_image(self.image, box)
