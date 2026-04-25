import time
import numpy as np
import galois
import itertools

# --- VERIFICATION ---

if __name__ == "__main__":
    m = 8
    n = 32
    N = 2*m*n
    J = np.random.randint(0, 256, size = (m, n), dtype = np.uint8)
    I = 255*np.ones((m, n), dtype = np.uint8)
    reference = np.zeros((2*m*n,), dtype = np.uint64)
    for y1, x1, y2, x2, y3, x3 in itertools.product(range(m), range(n), repeat = 3):
        reference[x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)] += np.uint64(I[y1,x1])*I[y2,x2]*I[y3,x3]
    reference //= 6
    maximal = np.max(reference)
    print(np.argmax(reference))
    p = int(maximal)
    while True:
        p = galois.next_prime(p)
        if (p-1)%(2*m*n) == 0:
            break 
        p += 1
    GF = galois.GF(p)
    J = GF(J)
    NTTJ = GF([galois.ntt(J[r], N) for r in range(m)])
    print(NTTJ)
