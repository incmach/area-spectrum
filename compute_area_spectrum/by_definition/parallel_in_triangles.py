import numpy as np
import math
import itertools as it
import multiprocessing
import joblib
import uuid

from .common import double_volume

_images = dict()

def _points_after(v0, shape):
    rows, cols = shape
    y0, x0 = v0
    for x in range(x0+1, cols):
        yield (y0, x)
    for y in range(y0+1, rows):
        for x in range(cols):
            yield (y, x)

def _compute_summand_with_first_poiht(image_key, v0):
    I = _images[image_key]
    result = math.prod(I.shape)*[int(0)]
    result[0] += int(I[v0])**3 # v0 == v1 == v2
    for v2 in _points_after(v0, I.shape):
        result[0] += 3*int(I[v0])**2*int(I[v2]) # v0 == v1 < v2
    for v1 in _points_after(v0, I.shape):
        result[0] += 3*int(I[v0])*int(I[v1])**2 # v0 < v1 == v2
        for v2 in _points_after(v1, I.shape):
            area = double_volume(v0, v1, v2)
            result[area] += 6*math.prod(int(I[v]) for v in [ v0, v1, v2 ])

    return result

def f(I):
    rows, cols = I.shape
    key = uuid.uuid4()
    _images[key] = I
    summands = joblib.Parallel(n_jobs=min(max(rows*cols//8, 1), 16),
                               backend = 'multiprocessing',
                               context = multiprocessing.get_context('fork'))(
            joblib.delayed(_compute_summand_with_first_poiht)(key, v0)
            for v0 in it.product(range(rows), range(cols)))
    result = math.prod(I.shape)*[int(0)]
    for s in summands:
        for i, v in enumerate(result):
            result[i] += s[i]

    return result
