import itertools as it
import functools as ft
import math
import time

import numpy as np
import galois

import multiprocessing
multiprocessing.set_start_method('fork')
import joblib
with joblib.parallel_config(backend='multiprocessing', make_default=True):
    pass

#TODO computing a field for a prime seems very resource-heavy. That's a problem when parallelising. We can do related multiprocessing ourselves to avoid recomputation. The good news is that our fastest methods must get even faster
#TODO rewrite AS as sum of triple products, not combinations
#TODO clean up

TEST = False
random_seed = [ 37 ]

def TEST_compare_methods(reference_method, method, sizes):
    random_seed[0] += 1
    np.random.seed(random_seed[0])
    ref_timer = 0
    method_timer = 0
    for size_i, size in enumerate(sizes):
    #for size in it.product(*(range(n+1) for n in max_size)):
        for t in [ np.uint8 ]:
            print(f'progress: {size} ({size_i}/{len(sizes)}), type {t}')
            I = np.random.randint(np.iinfo(t).min, np.iinfo(t).max+1, size = size, dtype = t)

            start = time.perf_counter()
            reference = reference_method(I)
            ref_timer += time.perf_counter() - start

            start = time.perf_counter()
            computed = method(I)
            method_timer += time.perf_counter() - start

            if reference != computed:
                print(f'{random_seed}, {size} ({size_i}/{len(sizes)}), {t}:')
                print(f'{computed}')
                print(f'!=')
                print(f'reference {reference}')
                assert(False)
    print(f'{method_timer/len(sizes)}/reference {ref_timer/len(sizes)}')

def double_volume(*vs):
    v0 = vs[0]
    vs = vs[1:]
    #TODO this must be exact but looks non-exact, make it look exact e.g. via assertions
    return abs(int(np.round(np.linalg.det(
        [ 
            [ vij - v0j for vij, v0j in zip(vi, v0) ]
        for vi in vs ]))))

def get_zero_areas_count(spectrum):
    if len(spectrum) == 0:
        return 0
    tail_sum = sum(spectrum[1:])
    total_points_cube = spectrum[0] + tail_sum
    #TODO make look exact
    total_points_aprx = int(np.round(total_points**(1/3)))
    return math.comb(total_points_aprx, 3) - tail_sum

def compute_spectrum_by_definition(I):
    result = math.prod(I.shape)*[int(0)]
    grid_nodes = it.product(*[range(n) for n in I.shape])
    simplices = it.product(grid_nodes, repeat = len(I.shape)+1)
    for s in simplices:
        result[double_volume(*s)] += math.prod(int(I[v]) for v in s)
    return result

if TEST:
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
    
def compute_spectrum_by_definition_ordered_parallel(I):
    rows, cols = I.shape

    def points_after(v0):
        y0, x0 = v0
        for x in range(x0+1, cols):
            yield (y0, x)
        for y in range(y0+1, rows):
            for x in range(cols):
                yield (y, x)

    def compute_summand(v0):
        result = math.prod(I.shape)*[int(0)]
        result[0] += int(I[v0])**3 # v0 == v1 == v2
        for v2 in points_after(v0):
            result[0] += 3*int(I[v0])**2*int(I[v2]) # v0 == v1 < v2
        for v1 in points_after(v0):
            result[0] += 3*int(I[v0])*int(I[v1])**2 # v0 < v1 == v2
            for v2 in points_after(v1):
                area = double_volume(v0, v1, v2)
                result[area] += 6*math.prod(int(I[v]) for v in [ v0, v1, v2 ])

        return result

    summands = joblib.Parallel(n_jobs=16, return_as = 'generator')(
            joblib.delayed(compute_summand)(v0)
            for v0 in it.product(range(rows), range(cols)))
    result = math.prod(I.shape)*[int(0)]
    for s in summands:
        for i, v in enumerate(result):
            result[i] += s[i]

    return result

if TEST:
    print('compute_spectrum_by_definition_ordered_parallel')
    TEST_compare_methods(compute_spectrum_by_definition, compute_spectrum_by_definition_ordered_parallel, (4, 8))

precomputed_primes = dict()
precomputed_GFs = dict()
def get_min_ps(p, double_spectrum_size, q):
    if (p, double_spectrum_size, q) in precomputed_primes:
        return precomputed_primes[(p, double_spectrum_size, q)]
    ps = []
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

wip_Is = [  ]
def compute_spectrum_modulo_p_by_ntt(p, wip_I_idx, aggregator):
    GF = precomputed_GFs[p]
    I = wip_Is[wip_I_idx]
    spectrum_size = math.prod(I.shape)
    double_spectrum_size = 2*spectrum_size
    rows = I.shape[0]
    NTT_I = GF([
        galois.ntt(GF(I[r] if p > np.iinfo(I.dtype).max else I[r]%p), double_spectrum_size)
        for r in range(I.shape[0]) ])

    NTT_R = aggregator(NTT_I)
    R = galois.intt(NTT_R)
    if len(R) > 0:
        result = [ int(R[0]) ]
        for u, v in zip(R[1:spectrum_size], np.flip(R[1-spectrum_size:])):
            result.append(int(u) + int(v)) 
            
    return result


def compute_spectrum_by_ntt(I, aggregator, p = None, max_p = None, use_crt = True):
    rows, cols = I.shape
    spectrum_size = math.prod(I.shape)
    double_spectrum_size = 2*spectrum_size
    if p is None:
        p = sum(int(v) for row in I for v in row)**3
    ps = get_min_ps(p, double_spectrum_size, 1 if use_crt else p)
    if max_p is not None and len(ps) > 0 and ps[-1] > max_p:
        raise RuntimeError(f'not enough primes <= {max_p} for max value {p} and ntt size {double_spectrum_size}: got {ps}')

    if False:
        results = [ f(p) for p in ps ]
    else:    
        wip_I_idx = len(wip_Is)
        wip_Is.append(I)
        results = joblib.Parallel(n_jobs=16, backend = 'multiprocessing')(
                joblib.delayed(compute_spectrum_modulo_p_by_ntt)(p, wip_I_idx, aggregator)
                for p in ps)
        del wip_Is[wip_I_idx]

    result = [ ]
    for r in zip(*results):
        result.append(galois.crt(r, ps) if len(ps) > 1 else r[0])

    return result

def aggregate_area_spectrum_ntt_per_row_triplets(NTT_I):
    rows, double_spectrum_size = NTT_I.shape
    NTT_R = np.zeros_like(NTT_I[0])
    for ys in it.product(range(rows), repeat = 3):
        complement = [
            NTT_I[ ys[y_i], [ k*(ys[(y_i+1)%3]-ys[(y_i+2)%3])%double_spectrum_size for k in range(double_spectrum_size) ] ]
            for y_i in range(3)
        ]
        NTT_R += math.prod(complement)
    return NTT_R

if True:
    print('compute_spectrum_by_ntt(aggregate_area_spectrum_ntt_per_row_triplets)')
    TEST_compare_methods(compute_spectrum_by_definition_ordered_parallel,
                         lambda I: compute_spectrum_by_ntt(I, aggregate_area_spectrum_ntt_per_row_triplets, (255*8*32)**3, 2**16, True),
                         [(8, 32)])
    print('compute_spectrum_by_ntt(aggregate_area_spectrum_ntt_per_row_triplets, use_crt = True)')
    TEST_compare_methods(compute_spectrum_by_definition_ordered_parallel,
                         lambda I: compute_spectrum_by_ntt(I, aggregate_area_spectrum_ntt_per_row_triplets, (255*8*32)**3, 2**16, True),
                         [(8, 32)]*16)
exit()

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
                summand = math.prod(complement)
                if y_1 < y_3:
                    summand *= 3
                    if y_1 < y_2 < y_3:
                        summand[0] *= 2
                        summand[1:] += np.flip(summand[1:])
                NTT_R += summand
    return NTT_R

if TEST:
    print('compute_spectrum_by_ntt(aggregate_area_spectrum_ntt_per_ordered_row_triplets)')
    TEST_compare_methods(compute_spectrum_by_definition_ordered_parallel,
                         lambda I: compute_spectrum_by_ntt(I, aggregate_area_spectrum_ntt_per_ordered_row_triplets, None, None, False),
                         (4, 8))
    print('compute_spectrum_by_ntt(aggregate_area_spectrum_ntt_per_ordered_row_triplets, use_crt = True)')
    TEST_compare_methods(compute_spectrum_by_definition_ordered_parallel,
                         lambda I: compute_spectrum_by_ntt(I, aggregate_area_spectrum_ntt_per_ordered_row_triplets, None, 2**20, True),
                         (4, 8))

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
            
            summand = np.sum(
                factors_1*factors_2*factors_3[:,-d_12-d_23:y_max-d_12-d_23],
                axis = 1)
            if d_12 != 0 or d_23 != 0:
                summand *= 3
                if d_12 != 0 and d_23 != 0:
                    summand[0] *= 2
                    summand[1:] += np.flip(summand[1:])

            NTT_R += summand

    return NTT_R

if TEST:
    print('compute_spectrum_by_ntt(aggregate_area_spectrum_ntt_per_ordered_diff_pairs)')
    TEST_compare_methods(compute_spectrum_by_definition_ordered_parallel,
                         lambda I: compute_spectrum_by_ntt(I, aggregate_area_spectrum_ntt_per_ordered_diff_pairs, None, None, False),
                         (4, 8))
    print('compute_spectrum_by_ntt(aggregate_area_spectrum_ntt_per_ordered_diff_pairs, use_crt = True)')
    TEST_compare_methods(compute_spectrum_by_definition_ordered_parallel,
                         lambda I: compute_spectrum_by_ntt(I, aggregate_area_spectrum_ntt_per_ordered_diff_pairs, None, 2**20, True),
                         (4, 8))

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
def aggregate_area_spectrum_ntt_per_ordered_diff_pairs_parallel(NTT_I):

    rows, double_spectrum_size = NTT_I.shape
    dt = NTT_I.dtype
    if dt not in [ np.uint8, np.uint16, np.uint32, np.uint64 ]:
        raise RuntimeError('Unable to marshall the ntt with given datatype. Please use CRT')
    p = NTT_I._order

    NTT_I_T_shm = shared_memory.SharedMemory(create = True, size = NTT_I.nbytes)
    NTT_I_T_shm_name = NTT_I_T_shm.name
    NTT_I_T = np.ndarray((double_spectrum_size, rows), dtype = dt, buffer = NTT_I_T_shm.buf)
    NTT_I_T[:,:] = NTT_I.T

    def compute_area_spectrum_summand(d_12):
        NTT_I_T_shm = shared_memory.SharedMemory(NTT_I_T_shm_name)
        GF = galois.GF(p)
        NTT_I_T = GF(np.ndarray((double_spectrum_size, rows), dtype = dt, buffer = NTT_I_T_shm.buf))
        all_factors_idxs, closeable = get_factors_idxs((rows, double_spectrum_size))
        result = np.zeros_like(NTT_I_T[:,0])
        for d_23 in range(1-rows-d_12, 1):
            y_max = rows+d_12+d_23
            summand = np.sum(
                      math.prod(NTT_I_T[all_factors_idxs[i],lower:upper] for (i, lower, upper) in
                      [ (d_23, 0, y_max), (-d_12-d_23, -d_12, y_max-d_12), (d_12, -d_12-d_23, rows) ]),
                axis = 1)
            if d_12 != 0 or d_23 != 0:
                summand *= 3
                if d_12 != 0 and d_23 != 0:
                    summand[0] *= 2
                    summand[1:] += np.flip(summand[1:])
            result += summand
        closeable.close()
        NTT_I_T_shm.close()
        return result


    summands = joblib.Parallel(n_jobs=16, return_as = 'generator')(
            joblib.delayed(compute_area_spectrum_summand)(d_12)
            for d_12 in range(1-rows, 1))
    
    
    result = np.zeros_like(NTT_I[0])
    for summand in summands:
        result += summand

    NTT_I_T_shm.close()
    NTT_I_T_shm.unlink()

    return result

if TEST:
    print('compute_spectrum_by_ntt(aggregate_area_spectrum_ntt_per_ordered_diff_pairs_parallel)')
    TEST_compare_methods(compute_spectrum_by_definition_ordered_parallel,
                         lambda I: compute_spectrum_by_ntt(I, aggregate_area_spectrum_ntt_per_ordered_diff_pairs_parallel, None, None, False),
                         (3, 3))
    print('compute_spectrum_by_ntt(aggregate_area_spectrum_ntt_per_ordered_diff_pairs_parallel, use_crt = True)')
    TEST_compare_methods(compute_spectrum_by_definition_ordered_parallel,
                         lambda I: compute_spectrum_by_ntt(I, aggregate_area_spectrum_ntt_per_ordered_diff_pairs_parallel, None, 2**30, True),
                         (4, 8))

