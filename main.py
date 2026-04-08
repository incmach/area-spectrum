import time
import numpy as np
import cv2
from scipy.fft import fft, fft2, ifft, ifft2
from scipy.ndimage import rotate
from numba import njit, prange, get_num_threads, get_thread_id
import sys
import argparse
import math

# --- FAST NUMBA-ACCELERATED METHODS (O(N * m^2)) ---

@njit(parallel=True, fastmath=True)
def _spectrum_kernel(alpha, m, N):
    # 1. Symmetry: Only need to compute half + 1 bins
    N_half = N // 2 + 1
    hat_f_half = np.zeros(N_half, dtype=np.complex128)

    shift = m // 2 + 1
    side = m + shift + 1

    # 2. Flatten maps into a 3D array to avoid list-of-arrays indexing
    # Shape: [parity, index]
    total_elements = m * (2 * m - 1)
    buf_size = (total_elements + 1) // 2
    maps = np.zeros((2, 2, buf_size), dtype=np.int32)

    for c in range(1-m, m):
        for r in range(m):
            hs = (r + c) // 2 + shift
            hd = (r - c) // 2 + shift
            map_idx = (r + c) % 2
            idx = (r * (2 * m - 1) + (c + m - 1)) // 2
            maps[0, map_idx, idx] = hs * side + hd
            maps[1, map_idx, idx] = hd * side + hs

    # 3. Pre-allocate scratchpads for each thread
    num_threads = get_num_threads()
    # Workspace per thread to avoid allocations in prange
    ws_subalpha = np.zeros((num_threads, m, 2 * m - 1), dtype=np.complex128)
    ws_beta_k = np.zeros((num_threads, side, side), dtype=np.complex128)
    ws_merged = np.zeros((num_threads, m, 2 * m - 1), dtype=np.complex128)

    # 4. Parallel Loop over half spectrum
    for k in prange(N_half):
        tid = get_thread_id()

        # Local views of the workspace
        subalpha = ws_subalpha[tid]
        beta_k = ws_beta_k[tid]
        merged = ws_merged[tid]

        # Efficient column slicing
        for c in range(1-m, m):
            col_idx = (k * c) % N
            for r in range(m):
                subalpha[r, c + m - 1] = alpha[r, col_idx]

        merged.fill(0)

        # Process parities
        for offset in range(2):
            beta_k.fill(0)
            map_idx = (m - 1 + offset) % 2

            # Flattened transfer
            src = subalpha.ravel()
            dst = beta_k.ravel()
            m_split = maps[0, map_idx]
            m_merge = maps[1, map_idx]

            # Populate matrix (Explicit loop is often faster for sparse-scatter in Numba)
            for i in range(buf_size):
                dst[m_split[i]] = src[2*i + offset]

            # Matmul
            beta_k_sq = beta_k @ beta_k
            dst_sq = beta_k_sq.ravel()
            res_merged = merged.ravel()

            # Extract
            for i in range(buf_size):
                res_merged[2*i + offset] = dst_sq[m_merge[i]]

        # Vectorized sum
        hat_f_half[k] = np.sum(subalpha * merged)

    # 5. Reconstruct full spectrum using conjugate symmetry
    hat_f = np.empty(N, dtype=np.complex128)
    hat_f[:N_half] = hat_f_half
    for k in range(N_half, N):
        hat_f[k] = np.conj(hat_f[N - k])

    return hat_f

def compute_area_spectrum(I):
    m, n = I.shape
    N = 2 * m * n
    # Standard Forward FFT (raw sum with exp(-j...))
    alpha = fft(I.astype(np.float64), n=N, axis=1)
    hat_f = _spectrum_kernel(alpha, m, N)
    # The raw spatial sum is exactly the Inverse DFT of the frequency-domain products
    return ifft(hat_f).real

@njit(parallel=True, fastmath=True)
def _gradient_kernel(alpha, hat_G, m, N):
    # 1. Setup dimensions and symmetry
    N_half = N // 2 + 1
    num_threads = get_num_threads()

    shift = m // 2 + 1
    side = m + shift + 1
    total_elements = m * (2 * m - 1)
    buf_size = (total_elements + 1) // 2

    # 2. Pre-allocate Thread-Local Workspaces
    # We use a 3D array for grad_alpha to avoid race conditions during parallel accumulation
    ws_grad_alpha = np.zeros((num_threads, m, N), dtype=np.complex128)
    ws_subalpha = np.zeros((num_threads, m, 2 * m - 1), dtype=np.complex128)
    ws_beta_k = np.zeros((num_threads, side, side), dtype=np.complex128)
    ws_merged = np.zeros((num_threads, m, 2 * m - 1), dtype=np.complex128)

    # Pre-calculate maps (same as forward pass)
    maps = np.zeros((2, 2, buf_size), dtype=np.int32)
    for c in range(1-m, m):
        for r in range(m):
            hs = (r + c) // 2 + shift
            hd = (r - c) // 2 + shift
            map_idx = (r + c) % 2
            idx = (r * (2 * m - 1) + (c + m - 1)) // 2
            maps[0, map_idx, idx] = hs * side + hd
            maps[1, map_idx, idx] = hd * side + hs

    # 3. Parallel Loop over Frequencies (k)
    for k in prange(N_half):
        gk_conj = 3.0 * np.conj(hat_G[k])
        if abs(gk_conj) < 1e-15:
            continue

        tid = get_thread_id()
        subalpha = ws_subalpha[tid]
        beta_k = ws_beta_k[tid]
        merged = ws_merged[tid]
        grad_alpha_local = ws_grad_alpha[tid]

        # Construct subalpha for current k
        for c in range(1-m, m):
            col_idx = (k * c) % N
            for r in range(m):
                subalpha[r, c + m - 1] = alpha[r, col_idx]

        merged.fill(0)
        # Matrix squaring logic to find the gradient components
        for offset in range(2):
            beta_k.fill(0)
            map_idx = (m - 1 + offset) % 2

            # Populate matrix
            src = subalpha.ravel()
            dst = beta_k.ravel()
            m_split = maps[0, map_idx]
            for i in range(buf_size):
                dst[m_split[i]] = src[2*i + offset]

            # Square the matrix (The core O(m^2) operation)
            beta_k_sq = beta_k @ beta_k

            # Extract and merge
            dst_sq = beta_k_sq.ravel()
            res_merged = merged.ravel()
            m_merge = maps[1, map_idx]
            for i in range(buf_size):
                res_merged[2*i + offset] = dst_sq[m_merge[i]]

        # Accumulate the gradient for this k
        # We also account for the conjugate symmetry k -> N-k here
        for c in range(1-m, m):
            target_freq = (k * c) % N
            # For k=0 and k=N/2, we don't double the contribution
            weight = 1.0 if (k == 0 or (N % 2 == 0 and k == N // 2)) else 2.0

            for r in range(m):
                val = gk_conj * merged[r, c + m - 1]
                # Update local frequency bin
                grad_alpha_local[r, target_freq] += weight * val

    # 4. Reduction: Sum all thread-local gradients into one
    final_grad_alpha = np.zeros((m, N), dtype=np.complex128)
    for t in range(num_threads):
        final_grad_alpha += ws_grad_alpha[t]

    return final_grad_alpha

def compute_gradient(I, grad_output):
    m, n = I.shape
    N = 2 * m * n
    alpha = fft(I.astype(np.float64), n=N, axis=1)
    hat_G = fft(grad_output.astype(np.complex128))
    
    grad_alpha = _gradient_kernel(alpha, hat_G, m, N)
    # Backprop through Row-FFT requires the Forward FFT divided by N
    return fft(grad_alpha, axis=1).real[:, :n] / N


# --- NON-PARALLEL REFERENCE METHODS (Absolute Ground Truth) ---

@njit(parallel=False)
def compute_area_spectrum_reference(I):
    m, n = I.shape
    N = 2 * m * n
    f_ref = np.zeros(N)
    for y0 in range(m):
        for x0 in range(n):
            for y1 in range(m):
                for x1 in range(n):
                    for y2 in range(m):
                        for x2 in range(n):
                            s = (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)) % N
                            f_ref[s] += float(I[y0, x0]) * I[y1, x1] * I[y2, x2]
    return f_ref

@njit(parallel=False)
def compute_gradient_reference(I, grad_output):
    m, n = I.shape
    N = 2 * m * n
    grad_I = np.zeros((m, n))
    for y in range(m):
        for x in range(n):
            current_grad = 0.0
            for y1 in range(m):
                for x1 in range(n):
                    for y2 in range(m):
                        for x2 in range(n):
                            s = (x * (y1 - y2) + x1 * (y2 - y) + x2 * (y - y1)) % N
                            current_grad += 3.0 * grad_output[s] * float(I[y1, x1]) * I[y2, x2]
            grad_I[y, x] = current_grad
    return grad_I

# --- VERIFICATION ---

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == '--test':
    jitted = False
    #jitted = True

    np.random.seed(42)
    m = 16
    for logn in range(6,11):
        n = 2**logn
        # Use 3x3 to keep the reference time reasonable
        I = np.random.rand(m, n)
        grad_output = np.random.rand(2 * m * n)
        
        print(f"Dimensions: {m}x{n} | Spectrum Bins: {2*m*n}")
        print("Running calculations...")
        sys.stdout.flush()

        # 1. Forward Pass

        start_ts = time.time()
        f_fast = compute_area_spectrum(I)
        if not jitted:
            f_ref = compute_area_spectrum_reference(I)
            err_f = np.linalg.norm(f_ref - f_fast) / np.linalg.norm(f_ref)
            
            print(f"Forward Match:  {'SUCCESS' if err_f < 1e-12 else 'FAILED'}")
            print(f"  Relative Error: {err_f:.2e}")
        else:
            print(f"Forward-1: {time.time() - start_ts}")
        sys.stdout.flush()
    
        # 2. Gradient Pass
        start_ts = time.time()
        g_fast = compute_gradient(I, grad_output)
        if not jitted:
            g_ref = compute_gradient_reference(I, grad_output)
            err_g = np.linalg.norm(g_ref - g_fast) / np.linalg.norm(g_ref)
            
            print(f"Gradient Match: {'SUCCESS' if err_g < 1e-12 else 'FAILED'}")
            print(f"  Relative Error: {err_g:.2e}")
        else:
            print(f"Gradient-1: {time.time() - start_ts}")
        sys.stdout.flush()

        jitted = True
elif __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactive 1D DFT editing of an image.")
    parser.add_argument("image", help="Path to the input image (grayscale).")
    parser.add_argument("voffset", help="Subimage vertical offset")
    parser.add_argument("height", help="Subimage height")

    args = parser.parse_args()

    # Load image as 8-bit grayscale
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load image.")
        exit()

    voffset = int(args.voffset)
    height = int(args.height)
    
    img = img[voffset:voffset+height,:].copy()

    cv2.imshow("Image", img)

    as_img = compute_area_spectrum(img)
    as_img = as_img[:math.prod(img.shape)].reshape(*img.shape)
    print(len(np.unique(as_img)))
    as_img = cv2.normalize(as_img, None, 0.0001, 0.9999, cv2.NORM_MINMAX)
    print(len(np.unique(as_img)))
    as_img = np.where(as_img < 0.001, 1000*as_img, 1)
    as_img = cv2.normalize(as_img, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow("Area spectrum", as_img)

    # Main loop
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:   # 'q' or ESC
            break

    cv2.destroyAllWindows()

    # o. cv2.imread(args[1])
    # o. cv2.imshow()
    # o. compute_area_spectrum
    # o. cv2.imshow() spectrum reshaped as (m-1)*(n-1)
    # o. handle mouse wheel events on spectrum image
    # o. on a roll change the target value accordingly
    # o. on a click, do gradient descent starting at spatial
    # o. show the resulting image and the resulting spectrum
    pass 
