import functools
import itertools as it
import math
import numpy as np
import sympy
import galois
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

def ntt2d(arr: Any) -> Any:
    """Vectorized 2D NTT using numpy.apply_along_axis over the last two axes."""
    GF = type(arr)
    rows_ntt = GF(np.apply_along_axis(galois.ntt, -1, arr))
    return GF(np.apply_along_axis(galois.ntt, -2, rows_ntt))

def intt2d(arr: Any) -> Any:
    """Vectorized 2D INTT using numpy.apply_along_axis over the last two axes."""
    GF = type(arr)
    cols_intt = GF(np.apply_along_axis(galois.intt, -2, arr))
    return GF(np.apply_along_axis(galois.intt, -1, cols_intt))

def compute_area_spectrum_via_ntt_triple_correlation(image, batch_size=64):
    if image.size == 0:
        return []
    
    # Pad the image to shape (2R-1, 2C-1)
    padded_I = np.pad(image, tuple((0, n-1) for n in image.shape), constant_values=0)
    rows, cols = image.shape
    
    # 1. P Calculation: Bound the max possible value by (sum of all pixels)^3
    max_val = image.size*(np.iinfo(image.dtype).max**3)
    
    # Select multiple smallest fitting ps with a large enough product
    primes = []
    prod = 1
    p = 2
    while prod <= max_val:
        p = galois.next_prime(p)
        if (p - 1) % padded_I.size == 0:
            primes.append(p)
            prod *= p
            
    # Precompute CRT coefficients for vectorized restoration later
    M = prod
    crt_coeffs = []
    for p in primes:
        Mi = M // p
        yi = pow(Mi, -1, p)
        crt_coeffs.append(Mi * yi)
        
    # Pre-instantiate fields and images to prevent redundant processing
    GFs = [galois.GF(p) for p in primes]
    padded_Is = [GF(padded_I%p if p <= np.iinfo(image.dtype).max else padded_I) for GF,p in zip(GFs,primes)]
    
    # 2. Cross-Correlation: cyclically reverse padded_I to establish I(-u)
    padded_I_rev = np.roll(np.flip(padded_I, axis=(0, 1)), (1, 1), axis=(0, 1))
    ntt_I_revs = [ntt2d(GF(padded_I_rev%p if p <= np.iinfo(image.dtype).max else padded_I_rev)) for GF,p in zip(GFs,primes)]
    
    # Initialize the result array using object dtype for arbitrary precision
    result = np.zeros(image.size, dtype=object)
    
    # 3. Match NTT's standard un-shifted layout output mapping.
    dys = np.concatenate((np.arange(0, rows), np.arange(1-rows, 0)))
    dxs = np.concatenate((np.arange(0, cols), np.arange(1-cols, 0)))
    
    # Initialize batch iterator
    d_12_iterator = it.product(*(range(1-n, n) for n in image.shape))
    
    while True:
        batch = list(it.islice(d_12_iterator, batch_size))
        if not batch:
            break
            
        tc_section_primes = []
        
        # Calculate section of triple correlation in each GF(p) for the entire batch
        for i, p in enumerate(primes):
            GF = GFs[i]
            p_I = padded_Is[i]
            
            # Stack rolled images to create a 3D batch array of shape (Batch, 2R-1, 2C-1)
            rolled_batch = GF(np.stack([np.roll(p_I, d, (0, 1)) for d in batch]))
            
            # p_I broadcasts over the batched dimension
            J = p_I * rolled_batch
            
            # ntt_I_revs[i] broadcasts over the batched dimension
            tc_section_p = intt2d(ntt2d(J) * ntt_I_revs[i])
            
            # Convert Galois array to standard Python ints wrapped in numpy object array
            tc_section_primes.append(np.array(tc_section_p.tolist(), dtype=object))
            
        # Restore actual triple correlation via CRT across the entire batch
        tc_section = sum(tc_section_primes[i] * crt_coeffs[i] for i in range(len(primes))) % M
        
        # Batch Binning
        dy = np.array([d[0] for d in batch])[:, None, None]
        dx = np.array([d[1] for d in batch])[:, None, None]
        
        # Broadcast bin calculation to shape (Batch, 2R-1, 2C-1)
        bin_idxs = abs(dy * dxs.reshape(1, 1, -1) - dx * dys.reshape(1, -1, 1))
        
        bins = np.ravel(bin_idxs)
        actual_bins = bins < len(result)
        bins = bins[actual_bins]
        values = tc_section.ravel()[actual_bins]
         
        np.add.at(result, bins, values)

    return list(result)

if True:
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
    np.random.seed(39)
    for m in range(1,9):
        for n in range(1,9):
            print(f'{m},{n}')
            image = np.random.randint(0,256,(m,n),dtype=np.uint8)
            assert(compute_area_spectrum_via_ntt_triple_correlation(image) == compute_area_spectrum_by_definition(image))