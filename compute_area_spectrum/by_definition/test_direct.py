import numpy as np

from .direct import f

assert(f(np.ones((0,0), dtype = np.uint8)) == [])
assert(f(np.ones((1,1), dtype = np.uint8)) == [ 1 ])
assert(f(np.ones((2,1), dtype = np.uint8)) == [ 8, 0 ])
assert(f(np.ones((1,2), dtype = np.uint8)) == [ 8, 0 ])
assert(f(np.ones((2,2), dtype = np.uint8)) == [ 40, 24, 0, 0 ])
assert(f(np.ones((3,2), dtype = np.uint8)) == [ 108, 72, 36, 0, 0, 0 ])
assert(f(np.ones((2,3), dtype = np.uint8)) == [ 108, 72, 36, 0, 0, 0 ])
stretched_as = f(np.array([
    [      0,  32601,      0,     15,      0,      3,      0,       4,      0],
    [      5,      0,      6,      0,      7,      0,     17,       0,      9],
    [      0,     10,      0,     11,      0,     12,      0,      13,      0],
    [     14,      0,     15,      0,     16,      0,     19,       0,2**32+1] ], dtype=np.uint64))
unstretched_as = f(np.array([
    [      0,      5,  32601,      0,      0,      0],
    [     14,     10,      6,     15,      0,      0],
    [      0,     15,     11,      7,      3,      0],
    [      0,      0,     16,     12,     17,      4],
    [      0,      0,      0,     19,     13,      9],
    [      0,      0,      0,      0,2**32+1,      0]], dtype=np.uint64))
assert(stretched_as[::2] == unstretched_as[:len(unstretched_as)//2])
assert(stretched_as[1::2] == unstretched_as[len(unstretched_as)//2:] == [0]*(len(unstretched_as)//2))
