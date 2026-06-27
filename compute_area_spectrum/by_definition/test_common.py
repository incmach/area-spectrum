import numpy as np
import time

from . import direct

def TEST_method(method, sizes, random_seed = 38, reference_method = direct.f):
    np.random.seed(random_seed)
    ref_timer = 0
    method_timer = 0
    for size_i, size in enumerate(sizes):
        for t in [ np.uint8 ]:
            print(f'progress: {size} ({size_i+1}/{len(sizes)}), type {t}')
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

