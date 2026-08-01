import functools
import itertools as it
import math
import numpy as np
import sympy
import galois

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

def ntt2d(I, p):
    GF = galois.GF(p)
    per_row = GF([ galois.ntt(r, modulus = p) for r in I ])
    return GF([ galois.ntt(col) for col in per_row.T ]).T

def intt2d(I, p):
    GF = galois.GF(p)
    per_col = GF([ galois.intt(col) for col in I.T ]).T
    return GF([ galois.intt(row) for row in per_col ])[::-1]

#triple correlation
def compute_area_spectrum_via_ntt_triple_correlation(image):
    if image.size == 0:
        return [ ]
    padding = [ n//2 for n in image.shape ] 
    padded_I = np.pad(image, tuple((n, n) for n in padding), constant_values = 0)
    p = image.size*(255**3)
    p = galois.next_prime(p)
    while (p-1)%padded_I.size != 0:
        p = galois.next_prime(p+1)
    GF = galois.GF(p)
    result = np.zeros(image.size, dtype = int)
    for d_12 in it.product(*(range(n) for n in image.shape)):
        J = padded_I*np.roll(padded_I, d_12, (0, 1))
        tc_section = intt2d(ntt2d(J, p)*ntt2d(padded_I, p), p)[tuple(slice(0,n) for p, n in zip(padding, image.shape))]*3
        tc_section[0,0] //= 3

            
        ys, xs = [ np.arange(n) for n in image.shape ]
        dy, dx = d_12
        bin_idxs = abs(dy*xs.reshape(1, -1) - dx*ys.reshape(-1, 1))
        # components of both d_12 and d_23 are non-negative. 
        values = np.ravel(tc_section)
        bins = np.ravel(bin_idxs)
         
        result += np.bincount(bins, weights = values, minlength = image.size).astype(int)
        print()
        print(f'{d_12}:')
        print('padded_I')
        print(padded_I)
        print("J")
        print(J)
        print("tc_section")
        print(tc_section)
        print("bin_idxs")
        print(bin_idxs)
        print('result')
        print(result)

    return list(result)

#TODO test_new_main.py

if True:
    for cas in compute_area_spectrum_via_ntt_triple_correlation, compute_area_spectrum_by_definition:
        assert(cas(np.ones((0,0), dtype = np.uint8)) == [])
        assert(cas(np.ones((1,1), dtype = np.uint8)) == [ 1 ])
        v = cas(np.ones((2,1), dtype = np.uint8))
        if v != [ 8, 0 ]:
            print(v)
            assert(False)
        assert(cas(np.ones((1,2), dtype = np.uint8)) == [ 8, 0 ])
        assert(cas(np.ones((2,2), dtype = np.uint8)) == [ 40, 24, 0, 0 ])
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
