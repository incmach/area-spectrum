import functools as ft
import itertools as it
import math
import time

import numpy as np
import galois

TEST = True

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
        result[0] = math.comb(sum(sum(int(v) for v in row) for row in I), 3) - sum(result[1:])
    *_, gcd = it.accumulate(result[1:], galois.gcd, initial = 0)
    if gcd > 0:
        print(gcd)
        print([v//gcd for v in result])
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

def compute_area_spectrum_ntt_simple(I, p):
    # 1. Compute ntt per row
    spectrum_size = math.prod(I.shape)
    double_spectrum_size = 2*spectrum_size
    while True:
        p = galois.next_prime(p)
        if (p-1)%(2*spectrum_size)== 0:
            break 
        p += 1
    GF = galois.GF(p)
    rows = I.shape[0]
    NTT_I = GF([ galois.ntt(GF(I[r]), double_spectrum_size) for r in range(I.shape[0]) ]).T
    NTT_R = GF(np.zeros(double_spectrum_size, dtype=np.uint8))
    for ys in it.product(range(rows), repeat = 3):
        if ys[1] == ys[2] == 0:
            print(f'starting {ys[0]}')
        complement = [
                NTT_I[ [ k*(ys[(y_i+1)%3]-ys[(y_i+2)%3])%(spectrum_size*2) for k in range(double_spectrum_size) ], y_i ]
                for y_i in range(3)
            ]
        NTT_R += GF(1)/GF(3)*math.prod(complement)
    return [ int(v) for v in galois.intt(NTT_R)[:spectrum_size] ]

if TEST:
    I = np.random.randint(0, 16, size = (5,5), dtype = np.uint8)
    reference = compute_spectrum_naive(I)
    print(reference)
    computed = compute_area_spectrum_ntt_simple(I, max(reference[1:])+1)
    print(computed)
