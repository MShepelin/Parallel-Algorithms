from parallelrank import find_rank
from scipy.sparse import load_npz
from time import perf_counter
import numpy as np

matrix = load_npz("data/side_50000/matrix_1.npz")
print(len(matrix.indptr), len(matrix.indices), flush=True)

length = 50000
matrix = matrix[:length, :length]

print(len(matrix.indptr), len(matrix.indices), flush=True)

time_start = perf_counter()
rank = find_rank(matrix.indptr, matrix.indices, matrix.shape[1], matrix.shape[0])
time_stop = perf_counter()

print("Rank computation time in seconds (HUGE matrix):", time_stop-time_start)
np.save("data/algorithm_time_v0.3_{}.npy".format(length), np.array([time_stop-time_start]))
