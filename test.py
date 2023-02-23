from scipy.sparse import rand
import numpy as np
import tensorflow as tf
from parallelrank import find_rank

from scipy.sparse import csr_matrix

row = np.array( [4, 0, 1, 2, 5, 6, 8, 1, 3, 4, 4, 7, 2, 7, 8, 7, 9, 0, 2, 8])
col = np.array( [0, 1, 1, 1, 1, 1, 1, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 9, 9, 9])
data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
matrix = csr_matrix((data, (col, row)), shape=(10, 10))

#matrix = rand(10, 10, density=0.2, format='csr', dtype=np.int8)
#matrix.data[:] = 1

rank = tf.linalg.matrix_rank(matrix.todense(), tol=1e-5).numpy()
numpy_rank = np.linalg.matrix_rank(matrix.todense(), tol=1e-5)
rank_custom = find_rank(matrix.indptr, matrix.indices, matrix.shape[1], matrix.shape[0])

print("Rank by tensorflow", rank)
print("Rank by numpy", numpy_rank)
print("Rank by custom algorithm", rank_custom)
print(matrix.todense())
