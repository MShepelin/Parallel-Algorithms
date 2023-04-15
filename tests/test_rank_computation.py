from parallelrank import find_rank
from scipy.sparse import load_npz
from time import perf_counter

class TestRankComputation:
    def test_matrix_zero(self):
        matrix = load_npz("tests/matrix_zeros.npz")
        time_start = perf_counter()
        rank = find_rank(matrix.indptr, matrix.indices, matrix.shape[1], matrix.shape[0])
        time_stop = perf_counter()
        
        print("Rank computation time in seconds (matrix full of zeros):", time_stop-time_start)
        assert rank == 0
        
    def test_small_matrix(self):
        matrix = load_npz("tests/matrix_10x1000_1.npz")
        time_start = perf_counter()
        rank = find_rank(matrix.indptr, matrix.indices, matrix.shape[1], matrix.shape[0])
        time_stop = perf_counter()
        
        print("Rank computation time in seconds (small matrix):", time_stop-time_start)
        assert rank == 1
        
    def test_medium_matrix(self):
        matrix = load_npz("tests/matrix_1000x1000_93.npz")
        time_start = perf_counter()
        rank = find_rank(matrix.indptr, matrix.indices, matrix.shape[1], matrix.shape[0])
        time_stop = perf_counter()
        
        print("Rank computation time in seconds (medium matrix):", time_stop-time_start)
        assert rank == 93
        
    def test_large_matrix(self):
        matrix = load_npz("tests/matrix_10000x1000_10.npz")
        time_start = perf_counter()
        rank = find_rank(matrix.indptr, matrix.indices, matrix.shape[1], matrix.shape[0])
        time_stop = perf_counter()
        
        print("Rank computation time in seconds (large matrix):", time_stop-time_start)
        assert rank == 10
