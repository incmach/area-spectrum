import itertools
import math
import time

import numpy as np
import galois

def compute_max_spectrum(image_shape, spectrum_size, max_pixel_value):
    m, n = image_shape
    I = np.ones(image_shape, dtype = np.uint8)
    N = 2*spectrum_size
    S = np.zeros((N,), dtype = np.uint64)
    # this only works for reference, for max spectrum we should just take a sufficiently large prime
    for y1, x1, y2, x2, y3, x3 in itertools.product(range(m), range(n), repeat = 3):
        if y2 == x2 == x3 == y3 == 0:
            print(f'{x1}x{y1}')
        S[x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)] += np.uint64(I[y1,x1])*I[y2,x2]*I[y3,x3]
    S //= 6
    S *= max_pixel_value**3
    return S
 
if __name__ == "__main__":
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
