import itertools as it
import math
import time

import numpy as np
import galois

TEST = True

#TODO a couple of tests for it
#TODO joblib
def compute_spectrum_naive(I):
    result = 2*math.prod(I.shape)*[0]
    for vs in it.product(it.product(*[range(n) for n in I.shape]), repeat = len(I.shape) + 1):
        signed_area = int(np.round(np.linalg.det(
            [ [ vs[i+1][j] - vs[0][j] for j in range(len(I.shape)) ] for i in range(len(I.shape)) ])))
        result[signed_area] += math.prod(int(I[v]) for v in vs)
    for i, v in enumerate(result):
        result[i] = v//3
    return result[:math.prod(I.shape)]
        
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
    print(stretched_as[::2])
    print(unstretched_as)
    assert(stretched_as[::2] == unstretched_as[:len(stretched_as)//2])
    assert(unstretched_as[len(stretched_as)//2:] == (len(unstretched_as)-len(stretched_as)//2)*[0])
def compute_max_spectrum(image_shape, spectrum_size, max_pixel_value):
    m, n = image_shape
    I = np.ones(image_shape, dtype = np.uint8)
    N = 2*spectrum_size
    S = np.zeros((N,), dtype = np.uint64)
    for y1, x1, y2, x2, y3, x3 in itertools.product(range(m), range(n), repeat = 3):
        if y2 == x2 == x3 == y3 == 0:
            print(f'{x1}x{y1}')
        S[x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)] += np.uint64(I[y1,x1])*I[y2,x2]*I[y3,x3]
    S //= 6
    S *= max_pixel_value**3
    return S
 
if __name__ == "__main__" and not TEST:
    image_shape = (16, 32)
    spectrum_size = math.prod(image_shape)
    max_pixel_value = 255
    # we should later take spectrum[1:] to exclude 0s item which is trivially computed from other elements
    p = int(np.max(compute_max_spectrum(image_shape, spectrum_size, max_pixel_value)))
    while True:
        p = galois.next_prime(p)
        if (p-1)%(2*spectrum_size)== 0:
            break 
        p += 1
    print(f'p == 2^{math.log2(p):.2f}')
    GF = galois.GF(p)
    J = np.random.randint(0, max_pixel_value+1, size = image_shape, dtype = np.uint8)
    J = GF(J)
    NTTJT = GF([galois.ntt(J[r], 2*spectrum_size) for r in range(image_shape[0])]).T

    NTTAS = GF(np.zeros(2*spectrum_size, dtype=np.uint8))
    # iterate through triplets of rows
    for ys in itertools.product(range(image_shape[0]), repeat = 3):
        if ys[1] == ys[2] == 0:
            print(f'starting {ys[0]}')
        complement = [
                NTTJT[ [ k*(ys[(y_i+1)%3]-ys[(y_i+2)%3])%(spectrum_size*2) for k in range(spectrum_size*2) ], y_i ]
                for y_i in range(3)
            ]
        NTTAS += GF(1)/GF(6)*math.prod(complement)
    AS = galois.intt(NTTAS)
    print(AS)

    print(f'done');

