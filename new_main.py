import functools
import itertools as it
import math
import numpy as np
import sympy
import galois
import concurrent.futures
import multiprocessing
from typing import Any

_volume_fs = dict()
def double_volume(*vs):
    if len(vs) in _volume_fs:
        f = _volume_fs[len(vs)]
    else:
        vs_vars = [ sympy.symbols(' '.join(f'v_{i}_{j}' for j in range(len(vs[i])))) for i in range(len(vs)) ]
        formula = abs(sympy.det(sympy.Matrix([ [ vij - v0j for vij, v0j in zip(vi, vs_vars[0]) ] for vi in vs_vars[1:]])))
        f = sympy.lambdify(list(it.chain.from_iterable(vs_vars)), formula)
        _volume_fs[len(vs)] = f
    return f(*it.chain.from_iterable(vs))

def compute_area_spectrum_by_definition(I):
    result = math.prod(I.shape)*[int(0)]
    grid_nodes = it.product(*[range(n) for n in I.shape])
    simplices = it.product(grid_nodes, repeat = len(I.shape)+1)
    for s in simplices:
        result[double_volume(*s)] += math.prod(int(I[v]) for v in s)
    return result

def get_ntt_primes(shape, dtype):
    """
    Calculates multiple smallest fitting primes with a product large enough 
    to prevent arithmetic overflow during the NTT correlation phase.
    """
    if math.prod(shape) == 0:
        return []
        
    rows, cols = shape
    padded_size = (2 * rows - 1) * (2 * cols - 1)
    max_val = math.prod(shape) * (np.iinfo(dtype).max ** 3)
    
    primes = []
    prod = 1
    p = 2
    while prod <= max_val:
        p = galois.next_prime(p)
        if (p - 1) % padded_size == 0:
            primes.append(p)
            prod *= p
            
    return primes

_g_primes = []
_g_GFs = []
_g_padded_Is = []
_g_windows = []         # <--- NEW: Precomputed sliding windows
_g_ntt_I_revs = []
_g_crt_coeffs = []
_g_M = [1]
_g_dxs = []
_g_dys = []
_g_image_size = [0]

# Global precomputed DFT matrices for O(1) batched transforms
_g_W_R = []
_g_W_C = []
_g_iW_R = []
_g_iW_C = []
_g_N_inv = []

def process_batch(batch):
    """Worker function to process a batch of d_12 elements."""
    tc_section_primes = np.zeros((len(_g_primes),len(batch),) + _g_padded_Is[0].shape, dtype = np.int64)
    
    # 1. Precalculate shifts and slicing indices for the entire batch ONCE
    H, W = _g_padded_Is[0].shape
    
    dys = np.array([d[0] for d in batch])
    dxs = np.array([d[1] for d in batch])
    
    # Map the shifts to their starting slice indices on a 2x2 tiled array
    sys = H - (dys % H)
    sxs = W - (dxs % W)
    
    # Calculate section of triple correlation in each GF(p) for the entire batch
    for i, p in enumerate(_g_primes):
        GF = _g_GFs[i]
        p_I = _g_padded_Is[i]
        
        W_R, W_C = _g_W_R[i], _g_W_C[i]
        iW_R, iW_C = _g_iW_R[i], _g_iW_C[i]
        N_inv = _g_N_inv[i]
        
        # --- PRECOMPUTED ZERO-COPY BATCH SLICING ---
        # Fetch exact memory views instantly via advanced indexing
        rolled_batch = GF(_g_windows[i][sys, sxs])
        
        # p_I broadcasts over the batched dimension
        J = p_I * rolled_batch
        
        # Purely vectorized batched 2D NTT via matrix multiplication
        ntt_J = W_R @ J @ W_C
        
        # Purely vectorized batched 2D INTT via matrix multiplication
        tc_section_p = (iW_R @ (ntt_J * _g_ntt_I_revs[i]) @ iW_C) * N_inv
        
        # Convert Galois array to standard Python ints wrapped in numpy object array
        tc_section_primes[i,:] = tc_section_p.astype(np.int64)
        
    # Restore actual triple correlation via CRT across the entire batch
    M = _g_M[0]
    tc_section = np.tensordot(_g_crt_coeffs, tc_section_primes, axes = 1) % M
    
    # Batch Binning
    dy = dys[:, None, None]
    dx = dxs[:, None, None]
    
    # Broadcast bin calculation to shape (Batch, 2R-1, 2C-1)
    bin_idxs = abs(dy * _g_dxs[0].reshape(1, 1, -1) - dx * _g_dys[0].reshape(1, -1, 1))
    
    bins = np.ravel(bin_idxs)
    img_size = _g_image_size[0]
    actual_bins = bins < img_size
    
    bins = bins[actual_bins]
    values = tc_section.ravel()[actual_bins]
    
    # Return basic python lists to eliminate the heavy IPC object-pickling overhead
    return bins.tolist(), values.tolist()

def compute_area_spectrum_via_ntt_triple_correlation(image, primes, batch_size=8, max_workers=None):
    if image.size == 0 or not primes: #[cite: 1]
        return [] #[cite: 1]
    
    # Pad the image to shape (2R-1, 2C-1)
    padded_I = np.pad(image, tuple((0, n-1) for n in image.shape), constant_values=0) #[cite: 1]
    rows, cols = image.shape #[cite: 1]
    rows_pad, cols_pad = padded_I.shape #[cite: 1]
    
    M = math.prod(primes) #[cite: 1]
    crt_coeffs = [] #[cite: 1]
    
    GFs = [] #[cite: 1]
    padded_Is = [] #[cite: 1]
    windows_list = []      # <--- NEW: Local tracker for windows
    ntt_I_revs = [] #[cite: 1]
    
    W_R_list, W_C_list = [], [] #[cite: 1]
    iW_R_list, iW_C_list = [], [] #[cite: 1]
    N_inv_list = [] #[cite: 1]
    
    padded_I_rev = np.roll(np.flip(padded_I, axis=(0, 1)), (1, 1), axis=(0, 1)) #[cite: 1]
    
    # Precompute fields, matrices, and arrays
    for p in primes: #[cite: 1]
        Mi = M // p #[cite: 1]
        yi = pow(Mi, -1, p) #[cite: 1]
        crt_coeffs.append(Mi * yi) #[cite: 1]
        
        GF = galois.GF(p) #[cite: 1]
        GFs.append(GF) #[cite: 1]
        
        # 1. Precompute batched DFT matrices for pure C-level vectorization
        alpha_R = GF.primitive_element ** ((GF.order - 1) // rows_pad) #[cite: 1]
        pow_R = (np.arange(rows_pad)[:, None] * np.arange(rows_pad)[None, :]) % rows_pad #[cite: 1]
        W_R = alpha_R ** pow_R #[cite: 1]
        iW_R = (alpha_R ** -1) ** pow_R #[cite: 1]
        
        alpha_C = GF.primitive_element ** ((GF.order - 1) // cols_pad) #[cite: 1]
        pow_C = (np.arange(cols_pad)[:, None] * np.arange(cols_pad)[None, :]) % cols_pad #[cite: 1]
        W_C = alpha_C ** pow_C #[cite: 1]
        iW_C = (alpha_C ** -1) ** pow_C #[cite: 1]
        
        W_R_list.append(W_R) #[cite: 1]
        W_C_list.append(W_C) #[cite: 1]
        iW_R_list.append(iW_R) #[cite: 1]
        iW_C_list.append(iW_C) #[cite: 1]
        
        N_inv = GF(rows_pad * cols_pad) ** -1 #[cite: 1]
        N_inv_list.append(N_inv) #[cite: 1]
        
        # 2. Map images to field and pre-calculate NTT for reverse image
        p_I = GF(padded_I%p if p <= np.iinfo(image.dtype).max else padded_I) #[cite: 1]
        padded_Is.append(p_I) #[cite: 1]
        
        # <--- NEW: Pre-tile and create sliding windows for O(1) rolling
        tiled_p_I = np.tile(p_I, (2, 2))
        windows = np.lib.stride_tricks.sliding_window_view(tiled_p_I, p_I.shape)
        windows_list.append(windows)
        # --->
        
        p_I_rev = GF(padded_I_rev%p if p <= np.iinfo(image.dtype).max else padded_I_rev) #[cite: 1]
        ntt_I_rev = W_R @ p_I_rev @ W_C #[cite: 1]
        ntt_I_revs.append(ntt_I_rev) #[cite: 1]
    
    # Update module-level globals so child processes inherit state efficiently via copy-on-write
    _g_primes[:] = primes #[cite: 1]
    _g_GFs[:] = GFs #[cite: 1]
    _g_padded_Is[:] = padded_Is #[cite: 1]
    _g_windows[:] = windows_list   # <--- NEW: Hoist window views to globals
    _g_ntt_I_revs[:] = ntt_I_revs #[cite: 1]
    _g_crt_coeffs[:] = crt_coeffs #[cite: 1]
    _g_M[0] = M #[cite: 1]
        
    _g_W_R[:] = W_R_list
    _g_W_C[:] = W_C_list
    _g_iW_R[:] = iW_R_list
    _g_iW_C[:] = iW_C_list
    _g_N_inv[:] = N_inv_list
    
    _g_dys.clear()
    _g_dys.append(np.concatenate((np.arange(0, rows), np.arange(1-rows, 0))))
    
    _g_dxs.clear()
    _g_dxs.append(np.concatenate((np.arange(0, cols), np.arange(1-cols, 0))))
    
    _g_image_size[0] = image.size
    
    result = np.zeros(image.size, dtype=np.int64)
    d_12_iterator = it.product(*(range(1-n, n) for n in image.shape))
           
    if max_workers is None or max_workers > 1:
        ctx = multiprocessing.get_context('fork')
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, #mp_context=ctx
                                                   ) as executor:
            futures = []
            
            while True:
                batch = list(it.islice(d_12_iterator, batch_size))
                if not batch:
                    break
                futures.append(executor.submit(process_batch, batch))
                
            for future in concurrent.futures.as_completed(futures):
                bins, values = future.result()
                np.add.at(result, bins, values)
    else:
        while True:
            batch = list(it.islice(d_12_iterator, batch_size))
            if not batch:
                break
            bins, values = process_batch(batch)
            np.add.at(result, bins, values)

    return list(result)

if __name__ == '__main__':
    if False:
        for cas in [compute_area_spectrum_by_definition]:
            assert(cas(np.ones((0,0), dtype = np.uint8)) == [])
            assert(cas(np.ones((1,1), dtype = np.uint8)) == [ 1 ])
            assert(cas(np.ones((2,1), dtype = np.uint8)) == [ 8, 0 ])
            assert(cas(np.ones((1,2), dtype = np.uint8)) == [ 8, 0 ])
            assert(cas(np.array([[1,1],[1,1]], dtype = np.uint8)) == [ 40, 24, 0, 0 ])
            assert(cas(np.ones((3,2), dtype = np.uint8)) == [ 108, 72, 36, 0, 0, 0 ])
            assert(cas(np.ones((2,3), dtype = np.uint8)) == [ 108, 72, 36, 0, 0, 0 ])
            stretched_as = cas(np.array([
                [      0,  32601,      0,     15,      0,      3,      0,       4,      0],
                [      5,      0,      6,      0,      7,      0,     17,       0,      9],
                [      0,     10,      0,     11,      0,     12,      0,      13,      0],
                [     14,      0,     15,      0,     16,      0,     19,       0,2**32+1] ], dtype=np.uint64))
            unstretched_as = cas(np.array([
                [      0,      5,  32601,      0,      0,      0],
                [     14,     10,      6,     15,      0,      0],
                [      0,     15,     11,      7,      3,      0],
                [      0,      0,     16,     12,     17,      4],
                [      0,      0,      0,     19,     13,      9],
                [      0,      0,      0,      0,2**32+1,      0]], dtype=np.uint64))
            assert(stretched_as[::2] == unstretched_as[:len(unstretched_as)//2])
            assert(stretched_as[1::2] == unstretched_as[len(unstretched_as)//2:] == [0]*(len(unstretched_as)//2))
        
    if False:
        np.random.seed(39)
        for m in range(1,9):
            for n in range(1,9):
                print(f'{m},{n}')
                image = np.random.randint(0,256,(m,n),dtype=np.uint8)
                primes = get_ntt_primes(image.shape, image.dtype)
                assert(compute_area_spectrum_via_ntt_triple_correlation(image, primes) == compute_area_spectrum_by_definition(image))
    
    import time
    shape = (16,16)
    primes = get_ntt_primes(shape, np.uint8)
    image = np.random.randint(0,256,shape,dtype=np.uint8)
    compute_area_spectrum_via_ntt_triple_correlation(image, primes, batch_size = 16, max_workers = 1)
    total = 0
    
    print('...')
    for  i in range(1, 101):
        if i % 10 == 0:
            print(i)
        image = np.random.randint(0,256,shape,dtype=np.uint8)
        start = time.perf_counter()
        compute_area_spectrum_via_ntt_triple_correlation(image, primes, batch_size = 16, max_workers = 1)
        total += time.perf_counter() - start
    print(total)
