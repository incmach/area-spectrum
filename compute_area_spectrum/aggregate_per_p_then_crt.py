import uuid
import numpy as np
import math
import galois
import joblib
import multiprocessing

precomputed_primes = dict()
precomputed_GFs = dict()
def get_min_ps(p, double_spectrum_size, q):
    if (p, double_spectrum_size, q) in precomputed_primes:
        return precomputed_primes[(p, double_spectrum_size, q)]
    ps = []
    if double_spectrum_size != 0:
        while math.prod(ps) < p:
            q = galois.next_prime(q)
            if (q-1)%double_spectrum_size == 0:
                ps.append(q)
            q += 1
    ps = tuple(ps)
    precomputed_primes[(p, double_spectrum_size, q)] = ps
    for p in ps:
        if p not in precomputed_GFs:
            precomputed_GFs[p] = galois.GF(p)

    return ps

_images = dict()
def call_per_p_f(per_p_f, p, key):
    return per_p_f(p, _images[key])

def f(I, per_p_f, p = None, max_p = None, use_crt = True):
    rows, cols = I.shape
    spectrum_size = math.prod(I.shape)
    double_spectrum_size = 2*spectrum_size
    if p is None:
        p = (np.iinfo(I.dtype).max*rows*cols)**3
    ps = get_min_ps(p, double_spectrum_size, 1 if use_crt else p)
    if max_p is not None and len(ps) > 0 and ps[-1] > max_p:
        raise RuntimeError(f'not enough primes <= {max_p} for max value {p} and ntt size {double_spectrum_size}: got {ps}')

    if len(ps) <= 16:
        results = [ per_p_f(p, I) for p in ps ]
    else:    
        key = uuid.uuid4()
        _images[key] = I
        results = joblib.Parallel(n_jobs=max(min(len(ps), 16), 1),
                                  backend = 'multiprocessing',
                                  context = multiprocessing.get_context('fork'))(
                joblib.delayed(call_per_p_f)(per_p_f, p, key)
                for p in ps)
        del _images[key]

    result = [ ]
    for r in zip(*results):
        result.append(galois.crt(r, ps) if len(ps) > 1 else r[0])

    return result


