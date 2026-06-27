import itertools as it
from .test_common import TEST_method 
from .parallel_in_triangles import f

TEST_method(f, list(it.product(*(range(n+1) for n in (8,8)))))
