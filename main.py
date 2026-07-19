import itertools as it
import functools as ft
import math
import time

import numpy as np
import galois

import multiprocessing
import joblib

from compute_area_spectrum.by_definition.common import double_volume
from compute_area_spectrum.by_definition.direct import f as compute_spectrum_by_definition
from compute_area_spectrum.by_definition.parallel_in_triangles import f as compute_spectrum_by_definition_ordered_parallel
from compute_area_spectrum.aggregate_per_p_then_crt import f as compute_spectrum_by_ntt

from compute_area_spectrum.test_common import TEST_method

#TODO clean up
#TODO ideally, each method definition should be a formulaic chain of functions

TEST = False

def compute_spectrum_in_GF_by_ntt(aggregator, GF, I):
    p = GF.order
    spectrum_size = math.prod(I.shape)
    double_spectrum_size = 2*spectrum_size
    rows = I.shape[0]
    NTT_I = GF([
        galois.ntt(GF(I[r] if p > np.iinfo(I.dtype).max else I[r]%p), double_spectrum_size)
        for r in range(I.shape[0])
    ])

    NTT_R = aggregator(NTT_I)
    R = galois.intt(NTT_R)
    if len(R) > 0:
        result = [ int(R[0]) ]
        for u, v in zip(R[1:spectrum_size], np.flip(R[1-spectrum_size:])):
            result.append(int(u) + int(v)) 
            
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
    I | enumerate | triplets | group-by v | v, sum(value)
    min_ps(I) | for p in . (
        NTT(row) for row in I | enumerate | triplets | triple-cross-correlation-projection for t in . | sum | INTT
    ) | CRT
    print('compute_spectrum_by_ntt(aggregate_area_spectrum_ntt_per_row_triplets)')

    m = lambda use_crt, I : compute_spectrum_by_ntt(I,
                                                    ft.partial(compute_spectrum_in_GF_by_ntt,
                                                               aggregate_area_spectrum_ntt_per_row_triplets),
                                                    None, None, use_crt)
    TEST_method(ft.partial(m, False),
                [(8,8)],
                38,
                compute_spectrum_by_definition_ordered_parallel)
    m(False, np.ones((8,16), dtype = np.uint8))
    print('precomputed non-crt')
    m(True, np.ones((8,16), dtype = np.uint8))
    print('precomputed crt')
    TEST_method(ft.partial(m, True),
                [(8,16)]*256,
                39,
                ft.partial(m, False))

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

