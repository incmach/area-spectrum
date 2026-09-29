"""The interactive window: the part of the editable spectrum that needs a display.

Everything with a decision in it lives in spectra_ui; this is the loop, the key
map and the mouse arithmetic that turn a click into a delta. The split is what
makes the interesting behaviour testable: what a bin is worth, where it is on
screen and what target a delta asks for are pure functions, and this file only
decides when to ask.

The loop advances a run by one generator step per frame rather than running it
to completion in a background thread. A thread would need a lock around the
image, the editor and the window, and the run is already a generator that hands
back a snapshot per step -- so the frame is the natural place to advance it, and
cancelling is the absence of a next call rather than a flag a thread has to
poll. The cost is that the loop's frame rate is bounded below by the cost of a
step, which is the cost of a Jacobian column; at 32x32 that is tens of
milliseconds, which is fast enough to watch.
"""
import argparse
import sys

import cv2
import numpy as np

import descent
import ntt_spectrum
import spectra_ui as ui

WINDOW_NAME = "area spectrum"
FRAME_MS = 15

# OpenCV reports arrow keys as an unprintable code, and which code depends on
# the GUI backend in use. Both the X11 and Qt sets are listed so the same binary
# works whichever cv2 was built against, rather than arrows silently doing
# nothing on one of them.
_ARROW = {
    "left": {0xFF51, 0x01000012, 81, 2424832},
    "right": {0xFF53, 0x01000013, 83, 2424834},
    "up": {0xFF52, 0x01000014, 82, 2424833},
    "down": {0xFF50, 0x01000015, 80, 2424836},
}
_ARROW_OF = {code: name for name, codes in _ARROW.items() for code in codes}

MOUSE = cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP


def key_arrow(key):
    """Which arrow key a code is, or None."""
    return _ARROW_OF.get(key)


class Viewer:
    """One image, its editable spectrum, and whatever run is in flight."""

    def __init__(self, editor, max_steps=1000, patience=25, width=1):
        self.editor = editor
        self.max_steps = max_steps
        self.patience = patience
        self.width = width
        self.state = "idle"
        self.run = None            # the generator, while running
        self.last = None           # the newest Descent snapshot
        self.cancelled = False
        self.status = ""
        self.alert = None          # a bin to flash, and the frame to stop
        self.alert_until = 0
        self.frame = 0
        self.drag = None           # (bin, anchor_y, anchor_delta) while dragging
        self.size = ui.window_size(editor.n_bins)

    # -- what is on screen -------------------------------------------------

    def current_image(self):
        """The image to draw: the running result, or the one being edited."""
        if self.state in ("running", "done") and self.last is not None:
            return self.last.image
        return self.editor.image

    def current_loss(self):
        if self.state in ("running", "done") and self.last is not None:
            return self.last.loss
        return self.editor.loss()

    def title(self):
        """The status line, in the window title rather than on the image."""
        editor = self.editor
        if self.state == "running":
            head = f"running  step {self.last.accepted_steps}/{self.max_steps}"
        elif self.state == "done":
            how = "cancelled" if self.cancelled else (
                "patience" if self.last.stopped_early else "max steps")
            head = f"done ({how})  step {self.last.accepted_steps}"
        else:
            # Always the count, including zero. Zero asked for is the state a
            # run cannot do anything about, and it is the state a user reaches
            # by accident -- a drag that ended back at neutral, or a bin dragged
            # to neutral again. Saying so is more useful than omitting it.
            head = (f"editing  {self.editor.n_bins} bins, "
                    f"{int(self.editor.edited.sum())} asked")
        parts = [head, f"loss {self.current_loss():.6g}"]
        k = editor.selected
        if k is not None and 0 <= k < editor.n_bins:
            byte = int(ui.delta_bytes([editor.deltas[k]], editor.window)[0])
            parts.append(f"bin {k} byte {byte}")
        parts.append(f"window {editor.window:.4g}")
        if self.status:
            parts.append(self.status)
        elif self.state == "idle":
            parts.append("[enter] run  [esc] quit")
        else:
            parts.append("[esc] cancel" if self.state == "running" else "[key] continue")
        return "   ".join(parts)

    def panel(self):
        """The lower panel: the delta grid while idle, progress during a run."""
        box = (self.size[0], self.size[1] - ui.IMAGE_PANEL_H)
        if self.state == "running":
            return ui.render_progress(self.last.history, self.last.accepted_steps,
                                      self.max_steps, box=box)
        if self.state == "done":
            return ui.render_progress(self.last.history, self.last.accepted_steps,
                                      self.max_steps, box=box,
                                      stopped_early=self.last.stopped_early,
                                      cancelled=self.cancelled)
        return self.editor.render_grid(
            box=box, alert=self.alert if self.frame < self.alert_until else None)

    def compose(self):
        """The whole window as one BGR image, laid out top to bottom."""
        top = ui.render_image(self.current_image(),
                              box=(self.size[0], ui.IMAGE_PANEL_H))
        return np.vstack([top, self.panel()])

    # -- input -------------------------------------------------------------

    def on_mouse(self, event, x, y, flags, _=None):
        """A click selects and drags a bin; a drag is a count of pixels.

        The drag is anchored at the pixel the press landed on rather than
        accumulated per motion event, so the delta is a function of where the
        pointer is and not of how many times the pointer crossed a pixel
        boundary on the way -- no drift, and a drag back to where it started
        returns the bin to where it started.
        """
        if self.state != "idle" or event not in MOUSE:
            return
        k = ui.bin_at(x, y - ui.IMAGE_PANEL_H, self.editor.n_bins)
        if event == cv2.EVENT_MOUSEMOVE:
            if self.drag is not None and flags & cv2.EVENT_FLAG_LBUTTON:
                bin_, anchor_y, anchor = self.drag
                moved = anchor + (anchor_y - y) * ui.delta_per_pixel(self.editor.window)
                self.editor.set_delta(bin_, moved)
            return
        if k is None:
            return
        if not self.editor.can_edit(k):
            self.flash(k, "bin %d is unreachable: no pixel moves it" % k)
            return
        self.editor.selected = k
        self.snap(k)
        self.status = ""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (k, y, self.editor.deltas[k])
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None

    def flash(self, k, message):
        self.alert = k
        self.alert_until = self.frame + 20
        self.status = message

    def snap(self, k):
        """Put bin k's delta back on the whole-byte grid the display uses.

        A no-op for anything a drag or an arrow produced, because both build
        deltas out of whole pixels. It matters after the window has been halved
        or doubled, where one delta now draws as a different byte: snapping on
        grab is what keeps the value the pointer picks up the value that was
        being shown, instead of a sub-pixel remainder of the old scale.
        """
        byte = int(ui.delta_bytes([self.editor.deltas[k]], self.editor.window)[0])
        self.editor.set_delta(
            k, float(ui.delta_for_bytes(byte - ui.NEUTRAL, self.editor.window)[0]))

    def on_key(self, key):
        """The key map. Anything unclaimed is an acknowledgement of a result."""
        if key < 0:
            return True                      # nothing pressed
        if self.state == "running":
            if key == 27:                   # escape, or cancel
                self.finish(cancelled=True)
            return True
        if self.state == "done":
            self.editor.image = self.last.image.copy()
            self.editor.rebaseline()
            self.state = "idle"
            self.last = None
            self.status = "applied"
            return True
        if key in (ord("q"), 27):
            return False                     # quit
        if key in (13, 10, ord("d")):
            self.start()
        elif key == ord("r"):
            self.editor.reset()
            self.status = "reset"
        elif key in (ord("-"), ord("_"), ord("="), ord("+")):
            # Multiply the current window, not an absolute one: a keystroke is
            # a step of the control, and a run of them should walk the scale up
            # and down rather than snap to a value the user did not ask for.
            factor = 0.5 if key in (ord("-"), ord("_")) else 2.0
            self.editor.window = max(1e-9, self.editor.window * factor)
            self.status = ""
        else:
            arrow = key_arrow(key)
            if arrow in ("up", "down"):
                k = self.editor.selected
                if k is None:
                    self.status = "select a bin first"
                elif not self.editor.nudge(k, 1 if arrow == "up" else -1):
                    self.flash(k, "bin %d is unreachable" % k)
            elif arrow in ("left", "right"):
                self.move_selection(1 if arrow == "right" else -1)
        return True

    def move_selection(self, step):
        """Step the selection, staying inside the spectrum and skipping dead bins.

        Skipping unreachable bins is what makes arrowing across the row usable:
        on a small image most bins are unreachable, and stopping on every one of
        them would make moving the selection an exercise in patience.
        """
        n = self.editor.n_bins
        k = self.editor.selected
        k = 0 if k is None else (k + step) % n
        for _ in range(n):
            if self.editor.can_edit(k):
                self.editor.selected = k
                self.status = ""
                return
            k = (k + step) % n

    # -- the run -----------------------------------------------------------

    def start(self):
        """Begin a run against the target the deltas ask for.

        The scale passed is the masked one, so the run optimizes the bins that
        were asked for and is free to move the rest. Passing the full row sums
        would have the run try to hold every bin it was not asked about, which
        costs more than the request is worth and stops it accepting anything.

        Refused when nothing is asked for. The objective would be empty, every
        step would be found not to improve a loss of zero, and the run would
        report patience having done nothing -- technically true and useless to
        look at.
        """
        if not self.editor.edited.any():
            self.status = "nothing asked for: drag a bin first"
            return
        target = self.editor.target().tolist()
        self.run = descent.iter_descent(
            self.editor.image, target,
            scale=[int(s) for s in self.editor.objective_scale],
            max_steps=self.max_steps, patience=self.patience,
            min_value=0, max_value=self.editor.max_value, width=self.width,
            backend=self.editor.backend)
        self.cancelled = False
        self.last = next(self.run)
        self.state = "running"
        self.status = ""

    def pump(self):
        """Advance the run by one step, or finish it. Cheap when idle."""
        if self.state != "running":
            return
        try:
            self.last = next(self.run)
        except StopIteration:
            self.finish(cancelled=False)
            return
        if self.last.stopped_early:
            # A snapshot with stopped_early is the generator's last, so the next
            # call would only raise. Finishing here saves a frame of a dead
            # panel and, more to the point, is the difference between the run
            # being reported as ended by patience and as cancelled.
            self.finish(cancelled=False)

    def finish(self, cancelled):
        """Close the generator and hold the result for the user to look at."""
        if self.run is not None:
            self.run.close()
            self.run = None
        self.cancelled = cancelled
        self.state = "done"

    def loop(self):
        """Draw, poll input, advance the run, until the user quits."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        # AUTOSIZE, so the window cannot be resized under the hit testing and a
        # bin stays where it was drawn.
        shown = None
        while True:
            self.frame += 1
            cv2.imshow(WINDOW_NAME, self.compose())
            title = self.title()
            if title != shown:
                # Only on a change: an unchanged title set every frame is a
                # round-trip to the window manager every frame for nothing, and
                # the loss in the title does not change when the user is only
                # moving the pointer. (cv2's Qt build also logs a missing-bundle
                # font warning periodically regardless of this; that one is Qt's
                # own timer, not something the title causes.)
                cv2.setWindowTitle(WINDOW_NAME, title)
                shown = title
            key = cv2.waitKey(FRAME_MS) & 0xFF
            if not self.on_key(key):
                break
            self.pump()
        cv2.destroyAllWindows()
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="edit an image's area spectrum, then watch descent chase it")
    parser.add_argument("image", help="image file to load")
    parser.add_argument("--max-side", type=int, default=12,
                        help="reduce the image to this longest side (0 keeps it); "
                             "a run costs about the fourth power of the size, so "
                             "this is what decides whether a run is watchable")
    parser.add_argument("--window", type=float, default=0.1,
                        help="delta that saturates the display, and so the value "
                             "one dragged pixel is worth as a fraction of it; "
                             "0.1 puts a delta of 1 within one screen height")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--width", type=int, default=1,
                        help="candidates tried per step")
    parser.add_argument("--reference", action="store_true",
                        help="use the brute-force backend, to check the fast one")
    args = parser.parse_args(argv)
    try:
        I = ui.load_image(args.image, args.max_side or None)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    backend = descent.ReferenceBackend() if args.reference else ntt_spectrum.NTTBackend()
    editor = ui.Editor(I, window=args.window, backend=backend)
    return Viewer(editor, max_steps=args.max_steps, patience=args.patience,
                  width=args.width).loop()


if __name__ == "__main__":
    sys.exit(main())
