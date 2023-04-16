from parallelrank import find_rank
from scipy.sparse import load_npz
from time import perf_counter

matrix = load_npz("tests/matrix_10x1000_1.npz")
time_start = perf_counter()
rank = find_rank(matrix.indptr, matrix.indices, matrix.shape[1], matrix.shape[0])
time_stop = perf_counter()

print("Rank computation time in seconds (small matrix):", time_stop-time_start)
assert rank == 1