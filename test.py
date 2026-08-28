import numpy as np
import definition

cas = definition.area_spectrum
gas = definition.area_spectrum_int_gradient

def test_cas_and_zero_gradient(image, result):
    assert(cas(image) == result)
    assert(gas(image, result) == np.zeros_like(image).tolist())

test_cas_and_zero_gradient(np.ones((0,0), dtype = np.uint8), [])
test_cas_and_zero_gradient(np.ones((1,1), dtype = np.uint8), [ 1 ])
test_cas_and_zero_gradient(np.ones((2,1), dtype = np.uint8), [ 8, 0 ])
test_cas_and_zero_gradient(np.ones((1,2), dtype = np.uint8), [ 8, 0 ])
test_cas_and_zero_gradient(np.ones((2,2), dtype = np.uint8), [ 40, 24, 0, 0 ])
test_cas_and_zero_gradient(np.ones((3,2), dtype = np.uint8), [ 108, 72, 36, 0, 0, 0 ])
test_cas_and_zero_gradient(np.ones((2,3), dtype = np.uint8), [ 108, 72, 36, 0, 0, 0 ])
test_cas_and_zero_gradient(np.array([
    [      0,  32601,      0,     15,      0,      3,      0,       4,      0],
    [      5,      0,      6,      0,      7,      0,     17,       0,      9],
    [      0,     10,      0,     11,      0,     12,      0,      13,      0],
    [     14,      0,     15,      0,     16,      0,     19,       0,2**32+1] ], dtype=np.uint64),
    [79229976470091439004081799388, 0, 16015572677574090, 0, 16040389087768854, 0, 28607394755137830, 0, 23573211754915986, 0, 4213414667217552, 0, 15999208760009820, 0, 15967692078232416, 0, 21517904344770, 0, 15971969895599256, 0, 1030854160662, 0, 515420721996, 0, 11769636322922604, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
test_cas_and_zero_gradient(np.array([
    [      0,      5,  32601,      0,      0,      0],
    [     14,     10,      6,     15,      0,      0],
    [      0,     15,     11,      7,      3,      0],
    [      0,      0,     16,     12,     17,      4],
    [      0,      0,      0,     19,     13,      9],
    [      0,      0,      0,      0,2**32+1,      0]], dtype=np.uint64),
    [79229976470091439004081799388, 16015572677574090, 16040389087768854, 28607394755137830, 23573211754915986, 4213414667217552, 15999208760009820, 15967692078232416, 21517904344770, 15971969895599256, 1030854160662, 515420721996, 11769636322922604, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

print(cas(np.array([[1,0],[0,1]], dtype = np.uint8)))
print(cas(np.array([[1,1],[0,1]], dtype = np.uint8)))
print(gas(np.array([[1,0],[0,1]], dtype = np.uint8), [ 21, 6, 0, 0 ]))
