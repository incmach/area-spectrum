import itertools as it
import math
import time

import numpy as np
import galois

import joblib
from multiprocessing import shared_memory

TEST = True

def recompute_area_spectrum_at_zero(I, spectrum):
    spectrum[0] = math.comb(sum(sum(int(v) for v in row) for row in I), 3) - sum(spectrum[1:])

#TODO joblib
def compute_spectrum_naive(I):
    result = 2*math.prod(I.shape)*[int(0)]
    for vs in it.product(it.product(*[range(n) for n in I.shape]), repeat = len(I.shape) + 1):
        signed_area = int(np.round(np.linalg.det(
            [ [ vs[i+1][j] - vs[0][j] for j in range(len(I.shape)) ] for i in range(len(I.shape)) ])))
        if signed_area == 0:
            continue
        result[signed_area] += math.prod(int(I[v]) for v in vs)
    for i, v in enumerate(result):
        result[i] = v//3
    if result:
        result = result[:math.prod(I.shape)]
        recompute_area_spectrum_at_zero(I, result)
    return result
        
if TEST:
    assert(compute_spectrum_naive(np.zeros((0, 0), dtype = np.uint8)) == [ ])
    assert(len(compute_spectrum_naive(np.zeros((1, 1), dtype = np.uint8))) == 1)
    assert(len(compute_spectrum_naive(np.ones((1, 1), dtype = np.uint8))) == 1)
    assert(compute_spectrum_naive(np.ones((1, 2), dtype = np.uint8))[1:] == [ 0 ])
    assert(compute_spectrum_naive(np.ones((2, 1), dtype = np.uint8))[1:] == [ 0 ])
    assert(compute_spectrum_naive(np.ones((2, 2), dtype = np.uint8))[1:] == [ 4, 0, 0 ])
    assert(compute_spectrum_naive(np.ones((3, 3), dtype = np.uint8))[1:] == [ 32, 32, 4, 8, 0, 0, 0, 0 ])
    stretched_as = compute_spectrum_naive(np.array([
        [      0,  32601,      0,     15,      0,      3,      0,       4,      0],
        [      5,      0,      6,      0,      7,      0,     17,       0,      9],
        [      0,     10,      0,     11,      0,     12,      0,      13,      0],
        [     14,      0,     15,      0,     16,      0,     19,       0,2**32+1] ], dtype=np.uint64))
    
    unstretched_as = compute_spectrum_naive(np.array([
        [      0,      5,  32601,      0,      0,      0],
        [     14,     10,      6,     15,      0,      0],
        [      0,     15,     11,      7,      3,      0],
        [      0,      0,     16,     12,     17,      4],
        [      0,      0,      0,     19,     13,      9],
        [      0,      0,      0,      0,2**32+1,      0]], dtype=np.uint64))
    assert(len(stretched_as) == len(unstretched_as))
    for i in range(1, len(unstretched_as)):
        if i*2 < len(stretched_as):
            assert(stretched_as[i*2] == unstretched_as[i])
        else:
            assert(unstretched_as[i] == 0)
        if i%2 == 1:
            assert(stretched_as[i] == 0)

def compute_max_spectrum(image_shape, max_pixel_value):
    multiplier = max_pixel_value**3
    result = [ multiplier*v for v in compute_spectrum_naive(np.ones(image_shape, dtype = np.uint8)) ]
    return result

if TEST: 
    assert(compute_max_spectrum((4,4), 255)[1:] == compute_spectrum_naive(255*np.ones((4,4), dtype=np.uint8))[1:])

def aggregate_area_spectrum_ntt_per_row_triplets(NTT_I):
    rows, double_spectrum_size = NTT_I.shape
    NTT_R = np.zeros_like(NTT_I[0])
    for ys in it.product(range(rows), repeat = 3):
        complement = [
            NTT_I[ ys[y_i], [ k*(ys[(y_i+1)%3]-ys[(y_i+2)%3])%double_spectrum_size for k in range(double_spectrum_size) ] ]
            for y_i in range(3)
        ]
        summand = math.prod(complement)
        if set(ys) == {0,1,3}:
            print(f'{ys}: {summand}')
        NTT_R += math.prod(complement)
    return NTT_R

def aggregate_area_spectrum_ntt_per_ordered_row_triplets(NTT_I):
    rows, double_spectrum_size = NTT_I.shape
    NTT_R = np.zeros_like(NTT_I[0])
    for y_1 in range(rows):
        for y_2 in range(y_1, rows):
            for y_3 in range(y_2, rows):
                ys = [ y_1, y_2, y_3 ]
                complement = [
                    NTT_I[ ys[y_i], [ k*(ys[(y_i+1)%3]-ys[(y_i+2)%3])%double_spectrum_size for k in range(double_spectrum_size) ] ]
                    for y_i in range(3)
                ]
                summand = 3*math.prod(complement)
                NTT_R += summand
                if y_1 < y_2 < y_3:
                    reverse_summand = np.zeros_like(summand)
                    reverse_summand[0] = summand[0]
                    reverse_summand[1:] = np.flip(summand[1:])
                    NTT_R += reverse_summand
    return NTT_R

factors_idxs_cache = dict()
def get_factors_idxs(shape):
    rows, L = shape
    if shape in factors_idxs_cache:
        shm = shared_memory.SharedMemory(factors_idxs_cache[shape])
    else:
        shm = shared_memory.SharedMemory(create = True, size = L*(2*rows-1)*4)
    result = np.ndarray((2*rows-1, L), dtype = np.int32, buffer = shm.buf)
    if shape not in factors_idxs_cache:
        for d in it.chain(range(rows), range(1-rows,0)):
            result[d] = np.arange(L)*d%L
        factors_idxs_cache[shape] = shm.name
    return result, shm

#TODO clean up shm handling
def aggregate_area_spectrum_ntt_per_ordered_diff_pairs_TODO(NTT_I):
    rows, double_spectrum_size = NTT_I.shape
    dt = NTT_I.dtype
    p = NTT_I._order

    NTT_I_T_shm = shared_memory.SharedMemory(create = True, size = NTT_I.nbytes)
    NTT_I_T_shm_name = NTT_I_T_shm.name
    NTT_I_T = np.ndarray((double_spectrum_size, rows), dtype = dt, buffer = NTT_I_T_shm.buf)
    NTT_I_T[:,:] = NTT_I.T

    get_factors_idxs(NTT_I.shape)

    def compute_area_spectrum_summand(d_12):
        NTT_I_T_shm = shared_memory.SharedMemory(NTT_I_T_shm_name)
        GF = galois.GF(p)
        NTT_I_T = GF(np.ndarray((double_spectrum_size, rows), dtype = dt, buffer = NTT_I_T_shm.buf))
        all_factors_idxs, closeable = get_factors_idxs((rows, double_spectrum_size))
        result = np.zeros_like(NTT_I_T[:,0])
        for d_23 in range(1-rows-d_12, 1):
            y_max = rows+d_12+d_23
            summand = 3*np.sum(
                      math.prod(NTT_I_T[all_factors_idxs[i],lower:upper] for (i, lower, upper) in
                      [ (d_23, 0, y_max), (-d_12-d_23, -d_12, y_max-d_12), (d_12, -d_12-d_23, rows) ]),
                axis = 1)
            if d_12 != 0 and d_23 != 0:
                summand[0] *= 2
                summand[1:] += np.flip(summand[1:])
            result += summand
        closeable.close()
        NTT_I_T_shm.close()
        return result


    summands = joblib.Parallel(n_jobs=4, return_as = 'generator')(
            joblib.delayed(compute_area_spectrum_summand)(d_12)
            for d_12 in range(1-rows, 1))
    
    
    result = np.zeros_like(NTT_I[0])
    for summand in summands:
        result += summand

    NTT_I_T_shm.close()
    NTT_I_T_shm.unlink()

    return result

factors_1_2_idxs_cache = dict()
factors_3_idxs_cache = dict()

def aggregate_area_spectrum_ntt_per_ordered_diff_pairs(NTT_I):
    NTT_I_T = NTT_I.T
    rows, double_spectrum_size = NTT_I.shape
    NTT_R = np.zeros_like(NTT_I[0])

    for d_12 in range(-(rows-1), 1):
        if (double_spectrum_size, d_12) in factors_3_idxs_cache:
            factors_3_idxs = factors_3_idxs_cache[(double_spectrum_size, d_12)]
        else:
            factors_3_idxs = [k*d_12%double_spectrum_size for k in range(double_spectrum_size) ]
            factors_3_idxs_cache[(double_spectrum_size, d_12)] = factors_3_idxs
        factors_3 = NTT_I_T[ factors_3_idxs, : ]
        for d_23 in range(-(rows-1) - d_12, 1):
            y_max = rows+d_12+d_23

            if (double_spectrum_size, d_12, d_23) in factors_1_2_idxs_cache:
                factors_1_idxs, factors_2_idxs = factors_1_2_idxs_cache[(double_spectrum_size, d_12, d_23)]
            else:
                factors_1_idxs = [k*d_23%double_spectrum_size for k in range(double_spectrum_size) ]
                factors_2_idxs = [k*(-d_12-d_23)%double_spectrum_size for k in range(double_spectrum_size) ]
                factors_1_2_idxs_cache[(double_spectrum_size, d_12, d_23)] = (factors_1_idxs, factors_2_idxs)

            factors_1 = NTT_I_T[ factors_1_idxs, :y_max ]
            factors_2 = NTT_I_T[ factors_2_idxs, -d_12:y_max-d_12 ]
            
            summand = 3*np.sum(
                factors_1*factors_2*factors_3[:,-d_12-d_23:y_max-d_12-d_23],
                axis = 1)

            NTT_R += summand
            if d_12 != 0 and d_23 != 0:
                NTT_R[0] += summand[0]
                NTT_R[1:] += np.flip(summand[1:])

    return NTT_R

#TODO CRT for orders over 2**20
precomputed_primes = dict()
def compute_area_spectrum_ntt_simple(I, aggregator = aggregate_area_spectrum_ntt_per_ordered_diff_pairs, p = None):
    spectrum_size = math.prod(I.shape)
    double_spectrum_size = 2*spectrum_size
    if p is None:
        p = math.comb(sum(int(v) for row in I for v in row), 3)
    if p in precomputed_primes:
        p = precomputed_primes[p]
    else:
        while True:
            p = galois.next_prime(p)
            if (p-1)%double_spectrum_size == 0:
                break 
            p += 1
    GF = galois.GF(p)
    rows = I.shape[0]
    NTT_I = GF([ galois.ntt(GF(I[r]), double_spectrum_size) for r in range(I.shape[0]) ])
    NTT_R = aggregator(NTT_I)
    result = [ int(v) for v in galois.intt(NTT_R)[:spectrum_size]/GF(3) ]
    recompute_area_spectrum_at_zero(I, result)
    return result

if TEST:
    np.random.seed(38)
    I = np.random.randint(0, 16, size = (8, 64), dtype = np.uint8)

    #start = time.time()
    #reference = compute_spectrum_naive(I)
    #print(time.time() - start)
    #print(reference)
    
    max_binary_as = compute_area_spectrum_ntt_simple(np.ones(I.shape, dtype = np.uint8))
    max_as_value = int(np.max(I))**3*max(max_binary_as[1:])

    print(f'max AS value is {max_as_value}')
    
    print('unrefactored pass:')
    for i in range(2):
        start = time.perf_counter()
        result0 = compute_area_spectrum_ntt_simple(I, aggregate_area_spectrum_ntt_per_ordered_diff_pairs, max_as_value)
        print(time.perf_counter() - start)

    try:
        print('refactored pass:')
        for i in range(2):
            start = time.perf_counter()
            result = compute_area_spectrum_ntt_simple(I, aggregate_area_spectrum_ntt_per_ordered_diff_pairs_TODO, max_as_value)
            print(time.perf_counter() - start)
    finally:
        for it in factors_idxs_cache:
            shm = shared_memory.SharedMemory(factors_idxs_cache[it])
            shm.close()
            shm.unlink()
    assert(result0 == result)
