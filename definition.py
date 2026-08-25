from functools import cache
import itertools
import math
import numpy as np
import sympy

@cache
def volume_function(d):
    vs = [ [ sympy.Symbol(f'v_{i}_{j}') for j in range(d) ] for i in range(d+1) ] 
    m = sympy.Matrix([ [ v_j - v_0_j for v_0_j, v_j in zip(vs[0],v) ] for v in vs[1:] ])
    return sympy.lambdify(tuple(itertools.chain(vs)), abs(m.det()))

def volume(vs):
    f = volume_function(len(vs)-1)
    return f(*vs)

def area_spectrum(I):
    result = [0]*I.size
    for vs in itertools.product(itertools.product(*(range(n) for n in I.shape)), repeat = len(I.shape)+1):
        result[volume(vs)] += math.prod(int(I[v]) for v in vs)
    return result;
