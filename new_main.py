import functools
import itertools as it
import math
import numpy as np
import sympy
import galois
import concurrent.futures
import multiprocessing
from typing import Any
import time

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
    
    max_val = 6 * math.prod(shape) * (int(np.iinfo(dtype).max) ** 3)
    
    primes = []
    prod = 1
    p = 2
    while prod <= max_val:
        p = galois.next_prime(p)
        if (p - 1) % padded_size == 0:
            primes.append(p)
            prod *= p
            
    return primes


@functools.lru_cache(maxsize=32)
def _get_ntt_precomputations(shape, dtype_str, primes_tuple):
    """
    Caches field computations, DFT arrays, and inversion grids that are 
    strictly dependent on the frame's structural dimensions.
    """
    rows, cols = shape
    rows_pad, cols_pad = 2 * rows - 1, 2 * cols - 1
    
    M = math.prod(primes_tuple)
    crt_coeffs = []
    GFs = []
    W_R_list, W_C_list = [], []
    iW_R_list, iW_C_list = [], []
    N_inv_list = []
    
    for p in primes_tuple:
        Mi = M // p
        yi = pow(Mi, -1, p)
        crt_coeffs.append(Mi * yi)
        
        GF = galois.GF(p)
        GFs.append(GF)
        
        alpha_R = GF.primitive_element ** ((GF.order - 1) // rows_pad)
        pow_R = (np.arange(rows_pad)[:, None] * np.arange(rows_pad)[None, :]) % rows_pad
        W_R = alpha_R ** pow_R
        iW_R = (alpha_R ** -1) ** pow_R
        
        alpha_C = GF.primitive_element ** ((GF.order - 1) // cols_pad)
        pow_C = (np.arange(cols_pad)[:, None] * np.arange(cols_pad)[None, :]) % cols_pad
        W_C = alpha_C ** pow_C
        iW_C = (alpha_C ** -1) ** pow_C
        
        W_R_list.append(W_R)
        W_C_list.append(W_C)
        iW_R_list.append(iW_R)
        iW_C_list.append(iW_C)
        
        N_inv = GF(rows_pad * cols_pad) ** -1
        N_inv_list.append(N_inv)

    dys_13 = np.concatenate((np.arange(0, rows), np.arange(1-rows, 0)))
    dxs_13 = np.concatenate((np.arange(0, cols), np.arange(1-cols, 0)))
    
    return (M, crt_coeffs, GFs, 
            W_R_list, W_C_list, iW_R_list, iW_C_list, N_inv_list,
            dys_13, dxs_13)


def compute_area_spectrum_via_ntt_triple_correlation(image, primes, batch_size=8):
    if image.size == 0 or not primes:
        return []
    
    rows, cols = image.shape
    
    # Extract cached invariants instantly via lru_cache
    (M, crt_coeffs, GFs, 
     W_R_list, W_C_list, iW_R_list, iW_C_list, N_inv_list,
     dys_13, dxs_13) = _get_ntt_precomputations(image.shape, str(image.dtype), tuple(primes))
    
    padded_I = np.pad(image, tuple((0, n-1) for n in image.shape), constant_values=0)
    padded_I_rev = np.roll(np.flip(padded_I, axis=(0, 1)), (1, 1), axis=(0, 1))
    
    padded_Is = []
    windows_list = []
    ntt_I_revs = []
    max_val = np.iinfo(image.dtype).max
    
    # Iteration remains strictly for dynamic, frame-dependent projections
    for i, p in enumerate(primes):
        GF = GFs[i]
        
        p_I = GF(padded_I % p if p <= max_val else padded_I)
        padded_Is.append(p_I)
        
        tiled_p_I = np.tile(p_I, (2, 2))
        windows = np.lib.stride_tricks.sliding_window_view(tiled_p_I, p_I.shape)
        windows_list.append(windows)
        
        p_I_rev = GF(padded_I_rev % p if p <= max_val else padded_I_rev)
        ntt_I_rev = W_R_list[i] @ p_I_rev @ W_C_list[i]
        ntt_I_revs.append(ntt_I_rev)

    d13_y = dys_13.reshape(1, -1, 1)
    d13_x = dxs_13.reshape(1, 1, -1)

    def process_batch(batch):
        tc_section_primes = np.zeros((len(primes), len(batch),) + padded_Is[0].shape, dtype=np.int64)
        
        H, W = padded_Is[0].shape
        
        dys = np.array([d[0] for d in batch])
        dxs = np.array([d[1] for d in batch])
        
        sys = H - (dys % H)
        sxs = W - (dxs % W)
        
        for i, p in enumerate(primes):
            GF = GFs[i]
            p_I = padded_Is[i]
            
            W_R, W_C = W_R_list[i], W_C_list[i]
            iW_R, iW_C = iW_R_list[i], iW_C_list[i]
            N_inv = N_inv_list[i]
            
            rolled_batch = GF(windows_list[i][sys, sxs])
            J = p_I * rolled_batch
            
            ntt_J = W_R @ J @ W_C
            tc_section_p = (iW_R @ (ntt_J * ntt_I_revs[i]) @ iW_C) * N_inv
            
            tc_section_primes[i,:] = tc_section_p.astype(np.int64)
            
        tc_section = np.tensordot(crt_coeffs, tc_section_primes, axes=1) % M
        
        d12_y = dys[:, None, None]
        d12_x = dxs[:, None, None]
        
        lex_greater = (d13_y > d12_y) | ((d13_y == d12_y) & (d13_x > d12_x))
        lex_equal = (d13_y == d12_y) & (d13_x == d12_x)
        lex_valid = lex_greater | lex_equal
        
        d12_is_zero = (d12_y == 0) & (d12_x == 0)
        
        weights_arr = np.full(tc_section.shape, 6, dtype=np.int64)
        weights_arr = np.where(d12_is_zero, 3, weights_arr)
        weights_arr = np.where(lex_equal, 3, weights_arr)
        weights_arr = np.where(d12_is_zero & lex_equal, 1, weights_arr)
        
        bin_idxs = abs(d12_y * dxs_13.reshape(1, 1, -1) - d12_x * dys_13.reshape(1, -1, 1))
        
        actual_bins = bin_idxs < image.size
        
        valid_mask = lex_valid & actual_bins
        
        bins = bin_idxs[valid_mask]
        values = (tc_section * weights_arr)[valid_mask]
        
        return bins, values

    result = np.zeros(image.size, dtype=np.int64)
    
    def generate_d12():
        for dy in range(1 - rows, rows):
            for dx in range(1 - cols, cols):
                if dy > 0 or (dy == 0 and dx >= 0):
                    yield (dy, dx)

    d_12_iterator = generate_d12()
    
    total_d12 = (math.prod(2*n - 1 for n in image.shape) + 1) // 2
    total_batches = math.ceil(total_d12 / batch_size)

    counter = 0
    start = time.perf_counter()
    while True:
        batch = list(it.islice(d_12_iterator, batch_size))
        if not batch:
            break

        bins, values = process_batch(batch)
        result += np.bincount(bins, weights=values, minlength=result.size).astype(np.int64)

        counter += 1
        if False and counter % 10 == 0:
            print(f'{counter}/{total_batches} batches done: {time.perf_counter() - start}')

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
    
    shape = (32, 32)
    batch_size = 64
    primes = get_ntt_primes(shape, np.uint8)
    image = np.random.randint(0, 256, shape, dtype=np.uint8)
    compute_area_spectrum_via_ntt_triple_correlation(image, primes, batch_size = batch_size)
    total = 0
    
    print('...')
    for i in range(1, 11):
        print(i)
        image = np.random.randint(0,256,shape,dtype=np.uint8)
        start = time.perf_counter()
        compute_area_spectrum_via_ntt_triple_correlation(image, primes, batch_size = batch_size)
        total += time.perf_counter() - start
    print(f'Total computed in {total:.4f} seconds')
