from scipy.sparse import rand
import numpy as np
import tensorflow as tf
from parallelrank import find_rank

matrix = rand(100, 100, density=0.2, format='csr', dtype=np.int8)
matrix.data[:] = 1

rank_custom = find_rank(matrix.indptr, matrix.indices, matrix.shape[1], matrix.shape[0])

rank = tf.linalg.matrix_rank(matrix.todense(), tol=1e-10).numpy()
print("Rank by tensorflow", rank)
print("Rank by custom algorithm", rank_custom)