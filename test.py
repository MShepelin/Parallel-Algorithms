from scipy.sparse import rand
import numpy as np

matrix = rand(10, 100, density=0.2, format='csr', dtype=np.int8)
matrix.data[:] = 1

from parallelrank import find_rank

rank = find_rank(matrix.indices, matrix.indptr, matrix.shape[0], matrix.shape[1])

print("Found rank", rank)