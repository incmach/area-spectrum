import argparse
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Interactive 1D DFT editing of an image.")
    parser.add_argument("image", help="Path to the input image (grayscale).")
    args = parser.parse_args()

    # Load image as 8-bit grayscale
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load image.")
        return

    h, w = img.shape
    N = h * w
    flat_img = img.ravel().astype(np.float64)

    # 1D DFT of the raveled image
    dft_coeffs = np.fft.fft(flat_img)   # complex array of length N

    # Display original image
    cv2.imshow("Image", img)

    # Scaling factor for DFT display (each original pixel becomes KxK squares)
    K = 10
    selected_index = None   # index in the raveled DFT array

    def update_dft_display():
        """Update the DFT magnitude window with current coefficients."""
        mag = np.abs(dft_coeffs)
        log_mag = np.log1p(mag)                     # log(1+mag) for dynamic range
        # Normalize to 0..255 and reshape to original dimensions
        mag_norm = cv2.normalize(log_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mag_img = mag_norm.reshape(h, w)
        # Scale each pixel to a KxK square (nearest neighbor to keep block effect)
        scaled = cv2.resize(mag_img, (w * K, h * K), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("DFT", scaled)

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_index, dft_coeffs

        if event == cv2.EVENT_RBUTTONDOWN:
            # Right click: select a pixel from the scaled DFT window
            orig_x = x // K
            orig_y = y // K
            if 0 <= orig_x < w and 0 <= orig_y < h:
                idx = orig_y * w + orig_x
                selected_index = idx
                print(f"Selected pixel at ({orig_x}, {orig_y}) → index {idx}")
            else:
                print("Click outside valid area.")

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Mouse wheel: modify magnitude of the selected DFT coefficient
            if selected_index is None:
                print("No pixel selected. Right‑click first.")
                return

            delta = 10.0      # magnitude change per wheel step
            if flags > 0:     # wheel up → increase magnitude
                inc = delta
                print("Wheel up: increasing magnitude")
            else:             # wheel down → decrease magnitude
                inc = -delta
                print("Wheel down: decreasing magnitude")

            old_val = dft_coeffs[selected_index]
            old_mag = np.abs(old_val)
            new_mag = max(0.0, old_mag + inc)

            if old_mag > 0:
                new_val = new_mag * (old_val / old_mag)   # preserve phase
            else:
                new_val = complex(new_mag, 0.0)

            dft_coeffs[selected_index] = new_val
            update_dft_display()   # refresh the DFT window

        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click: compute IDFT and show amplitude image
            idft = np.fft.ifft(dft_coeffs)
            idft_amp = np.abs(idft)                     # amplitude (magnitude)
            idft_img = idft_amp.reshape(h, w)           # reshape to original size
            # Normalize to 0..255 for display
            idft_norm = cv2.normalize(idft_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow("Image", idft_norm)
            print("IDFT computed and displayed in 'Image' window.")

    # Initial DFT display
    update_dft_display()

    # Attach mouse callback to the DFT window
    cv2.setMouseCallback("DFT", mouse_callback)

    print("Instructions:")
    print("- Right‑click on the DFT window to select a pixel.")
    print("- Mouse wheel up/down to increase/decrease the magnitude of the selected coefficient.")
    print("- Left‑click on the DFT window to compute the IDFT and show the resulting image.")
    print("Press 'q' or ESC to quit.")

    # Main loop
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:   # 'q' or ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()