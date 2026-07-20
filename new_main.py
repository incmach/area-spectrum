import functools
import itertools as it
import math
import numpy as np
import sympy

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

def compute_spectrum_by_definition(I):
    result = math.prod(I.shape)*[int(0)]
    grid_nodes = it.product(*[range(n) for n in I.shape])
    simplices = it.product(grid_nodes, repeat = len(I.shape)+1)
    for s in simplices:
        result[double_volume(*s)] += math.prod(int(I[v]) for v in s)
    return result

if True:
    assert(compute_spectrum_by_definition(np.ones((0,0), dtype = np.uint8)) == [])
    assert(compute_spectrum_by_definition(np.ones((1,1), dtype = np.uint8)) == [ 1 ])
    assert(compute_spectrum_by_definition(np.ones((2,1), dtype = np.uint8)) == [ 8, 0 ])
    assert(compute_spectrum_by_definition(np.ones((1,2), dtype = np.uint8)) == [ 8, 0 ])
    assert(compute_spectrum_by_definition(np.ones((2,2), dtype = np.uint8)) == [ 40, 24, 0, 0 ])
    assert(compute_spectrum_by_definition(np.ones((3,2), dtype = np.uint8)) == [ 108, 72, 36, 0, 0, 0 ])
    assert(compute_spectrum_by_definition(np.ones((2,3), dtype = np.uint8)) == [ 108, 72, 36, 0, 0, 0 ])
    stretched_as = compute_spectrum_by_definition(np.array([
        [      0,  32601,      0,     15,      0,      3,      0,       4,      0],
        [      5,      0,      6,      0,      7,      0,     17,       0,      9],
        [      0,     10,      0,     11,      0,     12,      0,      13,      0],
        [     14,      0,     15,      0,     16,      0,     19,       0,2**32+1] ], dtype=np.uint64))
    unstretched_as = compute_spectrum_by_definition(np.array([
        [      0,      5,  32601,      0,      0,      0],
        [     14,     10,      6,     15,      0,      0],
        [      0,     15,     11,      7,      3,      0],
        [      0,      0,     16,     12,     17,      4],
        [      0,      0,      0,     19,     13,      9],
        [      0,      0,      0,      0,2**32+1,      0]], dtype=np.uint64))
    assert(stretched_as[::2] == unstretched_as[:len(unstretched_as)//2])
    assert(stretched_as[1::2] == unstretched_as[len(unstretched_as)//2:] == [0]*(len(unstretched_as)//2))
 
# parallelisable ordered triplets of points ("by definition sped up")
# tests
# triple correlation
# tests
# parallelisable ordered pairs of diffs via NTT
# tests
